"""
eval/run_eval.py — §14 3-arm evaluation harness (architecture §2b, §7).

Runs the SAME held-out customer population (identical customer_ids,
identical latent archetype parameters -- built once, wrapped into three
independent CustomerBehaviorModel sets so each arm's rollout doesn't share
mutable cache state) through three policy arms:

  1. No-Intervention  -- always "none"
  2. Naive-Uniform    -- a fixed reminder cadence once overdue, regardless of risk
  3. Promise Integrity Agent -- Stage 4's policy.eiv.select_action(), driven
     by Stage 3's real trained models.propensity_model.PropensityModel

Population is held out from Stage 3's training set by construction: a
different seed, a fair/unweighted 7-archetype mix (Stage 3's population is
deliberately skewed -- see models/train.py's TRAINING_ARCHETYPE_WEIGHTS
docstring, which explicitly says eval must NOT reuse that skew), and a
distinct "eval-" customer_id prefix that can never collide with Stage 3's
"train-" prefix.

Ground-truth discipline (architecture §2b): the Agent arm's policy uses the
trained PropensityModel only to CHOOSE actions. Every arm's dollar outcome
comes from the simulator's own realized closure (SimInvoice.status/paid_day,
authoritatively tracked by SimulationEngine) -- which is itself always the
simulator's true potential outcome for the action actually taken, since
CustomerBehaviorModel.realize(action, day) is defined as
generate_potential_outcomes(...)[action]; there is no separate "model
belief" anywhere in that path. Table 2 goes further and uses an explicit
COUNTERFACTUAL potential outcome (generate_potential_outcomes(...)["none"]
at the moment of escalation, for the action NOT taken) to score escalation
precision -- this is the one place a genuine "what would have happened
otherwise" query is required, and it's answered directly from the
simulator's ground truth, never from the model.

Amount generation is a pure hash-of-invoice_id function (amount_for_invoice
below), not a shared sequentially-advancing RNG stream -- three arms
naturally diverge in when/whether invoices reissue (different interventions
close invoices on different days), so a shared stream would desync and
assign different dollar amounts to "the same" invoice_id across arms
depending on draw order. Hashing removes that risk entirely: any arm asking
about invoice_id X gets the same amount, independent of call order.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.state_machine import sync_dispute_state
from backend.db import Communication, Dispute, Invoice, Promise
from features.feature_engine import build_feature_vector, compute_prs
from models.propensity_model import PropensityModel
from models.train import make_db_session, seed_customers, sync_realized_day
from policy.constraints import PolicyConfig, PolicyState
from policy.eiv import ACTION_COST, select_action
from simulator.archetypes import ARCHETYPE_NAMES, sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel, CustomerLatentState
from simulator.engine import SimInvoice, SimulationEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_SEED = 4242          # distinct from Stage 3's SEED=7; "eval-" id prefix guarantees disjointness regardless
N_CUSTOMERS = 300         # fair, unweighted 7-archetype mix (round-robin) -- NOT Stage 3's skewed training mix
N_DAYS = 150
DECISION_CADENCE_DAYS = 6  # matches Stage 3's cadence, for the Agent arm's decision points
NAIVE_CADENCE_DAYS = 7     # Naive-Uniform: same reminder, every N days, once overdue, regardless of risk
REFERENCE_START = dt.datetime(2025, 1, 1)
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]

PRS_BAND_EDGES = [0.4, 0.6]  # matches the low/mid/high bins from the Stage 5 diagnostic
PRS_BAND_LABELS = ["low (0.0-0.4)", "mid (0.4-0.6)", "high (0.6-1.0)"]

CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def amount_for_invoice(invoice_id: str) -> float:
    seed = int.from_bytes(hashlib.sha256(f"amount|{invoice_id}".encode()).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed)
    return round(8000.0 * float(np.exp(rng.normal(0, 0.5))), 2)


def prs_band(prs_score: float) -> str:
    if prs_score < PRS_BAND_EDGES[0]:
        return PRS_BAND_LABELS[0]
    if prs_score < PRS_BAND_EDGES[1]:
        return PRS_BAND_LABELS[1]
    return PRS_BAND_LABELS[2]


# ---------------------------------------------------------------------------
# Held-out population
# ---------------------------------------------------------------------------

def build_eval_latents(seed: int, n_customers: int) -> list[CustomerLatentState]:
    """Built ONCE; each arm wraps these same latent values in its own fresh
    CustomerBehaviorModel instances (see run_eval_harness) so cache state
    stays isolated per arm while the underlying customer identity/behavior
    parameters are byte-for-byte identical -- the correctness property the
    3-arm comparison depends on, checked directly in tests/test_eval_harness.py.
    """
    rng = np.random.default_rng(seed)
    latents = []
    for i in range(n_customers):
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
        customer_id = f"eval-{i:04d}-{archetype}"
        latents.append(sample_customer_latent(customer_id, archetype, rng))
    return latents


# ---------------------------------------------------------------------------
# Policy arms
# ---------------------------------------------------------------------------

def no_intervention_policy(invoice: SimInvoice, day: int, context: dict) -> str:
    return "none"


def naive_uniform_policy(invoice: SimInvoice, day: int, context: dict) -> str:
    if day < invoice.due_day:
        return "none"
    if (day - invoice.due_day) % NAIVE_CADENCE_DAYS == 0:
        return "soft_reminder"
    return "none"


def sync_dispute_state_eval(session, engine: SimulationEngine, day: int) -> None:
    """Stage 13: the Agent arm's counterpart to agent/state_machine.py's
    own per-cycle sync_dispute_state call -- same function, called once per
    open invoice per day, so the disputer archetype's dispute lifecycle is
    identical whether an invoice is being driven by run_agent_cycle (the
    live loop / Stage 9 batch) or this eval harness. The other two arms
    (No-Intervention, Naive-Uniform) never sync anything to the DB at all
    (see run_no_intervention/run_naive_uniform) and don't need this either,
    since disputes are a policy-relevant DB concept, not a dollar-outcome one.
    """
    as_of = REFERENCE_START + dt.timedelta(days=day)
    for inv in engine.invoices:
        if inv.status != "open" or day < inv.issue_day:
            continue
        db_invoice = session.get(Invoice, inv.invoice_id)
        if db_invoice is None:
            continue  # not yet synced this cycle -- sync_new_invoices_eval runs first, so this shouldn't happen
        sync_dispute_state(db_invoice, engine.customers[inv.customer_id], session, as_of, REFERENCE_START)


def sync_new_invoices_eval(session, engine: SimulationEngine, synced_ids: set[str]) -> None:
    for inv in engine.invoices:
        if inv.invoice_id in synced_ids:
            continue
        session.add(Invoice(
            invoice_id=inv.invoice_id, customer_id=inv.customer_id, amount=amount_for_invoice(inv.invoice_id),
            issue_date=REFERENCE_START + dt.timedelta(days=inv.issue_day),
            due_date=REFERENCE_START + dt.timedelta(days=inv.due_day), status="open",
        ))
        synced_ids.add(inv.invoice_id)
    session.commit()


def make_agent_policy(session, model: PropensityModel, config: PolicyConfig,
                       decision_records: list, escalation_records: list):
    """Real Stage 4 select_action() (hard constraints -> EIV over Stage 3's
    real PropensityModel). dispute_flag/dispute_resolved are now derived
    from real DB state (Stage 13: sync_dispute_state_eval populates
    Invoice.dispute_flag/the Dispute table each day before this runs) --
    rule 1 can genuinely fire here. no_contact_requested remains
    hardcoded False: Stage 1 still has no opt-out concept anywhere in the
    simulator, so unlike dispute_flag there's no real per-customer signal
    to derive it from here (rule 2 is exercised via eval/scenarios.py at
    the agent-loop level instead, where AgentContext.no_contact_customer_ids
    exists for exactly this purpose).
    """
    def policy_fn(invoice: SimInvoice, day: int, context: dict) -> str:
        if day % DECISION_CADENCE_DAYS != 0:
            return "none"

        as_of = REFERENCE_START + dt.timedelta(days=day)
        features = build_feature_vector(invoice.invoice_id, "none", as_of=as_of, session=session)

        db_invoice = session.get(Invoice, invoice.invoice_id)
        unresolved_dispute = (
            session.query(Dispute)
            .filter(Dispute.invoice_id == invoice.invoice_id, Dispute.resolved.is_(False))
            .first()
        )

        contacts_recent = (
            session.query(Communication)
            .filter(
                Communication.invoice_id == invoice.invoice_id,
                Communication.timestamp > as_of - dt.timedelta(days=3),
                Communication.timestamp <= as_of,
            ).count()
        )
        total_contacts = (
            session.query(Communication)
            .filter(Communication.invoice_id == invoice.invoice_id, Communication.dispatched_by == "agent")
            .count()
        )

        invoice_amount = amount_for_invoice(invoice.invoice_id)
        state = PolicyState(
            invoice_amount=invoice_amount,
            dispute_flag=bool(db_invoice.dispute_flag) if db_invoice else False,
            dispute_resolved=unresolved_dispute is None,
            no_contact_requested=False,
            active_promise_flag=features["active_promise_flag"],
            days_until_promised_date=features["days_until_promised_date"],
            broken_promise_streak=features["broken_promise_streak"],
            contacts_in_last_3_days=contacts_recent,
            total_automated_contacts_this_invoice=total_contacts,
            prs_score=features["prs_score"],
        )

        action, rationale_codes, human_approval_required = select_action(
            invoice_amount, features, model, state, config
        )

        decision_records.append({
            "invoice_id": invoice.invoice_id, "customer_id": invoice.customer_id, "day": day,
            "action": action, "rationale_codes": rationale_codes, "prs_score": features["prs_score"],
        })
        if action == "human_escalation":
            escalation_records.append({
                "invoice_id": invoice.invoice_id, "customer_id": invoice.customer_id,
                "day": day, "context": dict(context),
            })
        return action

    return policy_fn


# ---------------------------------------------------------------------------
# Per-arm rollout
# ---------------------------------------------------------------------------

def run_no_intervention(latents: list[CustomerLatentState]) -> SimulationEngine:
    customers = [CustomerBehaviorModel(latent) for latent in latents]
    engine = SimulationEngine(customers, seed=EVAL_SEED)
    for day in range(N_DAYS):
        engine.step(day, no_intervention_policy)
    return engine


def run_naive_uniform(latents: list[CustomerLatentState]) -> SimulationEngine:
    customers = [CustomerBehaviorModel(latent) for latent in latents]
    engine = SimulationEngine(customers, seed=EVAL_SEED)
    for day in range(N_DAYS):
        engine.step(day, naive_uniform_policy)
    return engine


def run_agent(latents: list[CustomerLatentState], model: PropensityModel):
    customers = [CustomerBehaviorModel(latent) for latent in latents]
    engine = SimulationEngine(customers, seed=EVAL_SEED)
    session = make_db_session()
    seed_customers(session, customers, np.random.default_rng(EVAL_SEED))

    synced_ids: set[str] = set()
    decision_records: list[dict] = []
    escalation_records: list[dict] = []
    policy_fn = make_agent_policy(session, model, CONFIG, decision_records, escalation_records)

    for day in range(N_DAYS):
        sync_new_invoices_eval(session, engine, synced_ids)
        sync_dispute_state_eval(session, engine, day)
        engine.step(day, policy_fn)
        sync_realized_day(session, engine, day)

    customers_by_id = {c.latent.customer_id: c for c in customers}
    return engine, session, decision_records, escalation_records, customers_by_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def arm_dollar_metrics(engine: SimulationEngine, customer_ids: set[str] | None = None) -> dict:
    invoices = engine.invoices if customer_ids is None else [i for i in engine.invoices if i.customer_id in customer_ids]
    realized = engine.realized_log if customer_ids is None else [r for r in engine.realized_log if r.customer_id in customer_ids]

    total_recovered = sum(amount_for_invoice(inv.invoice_id) for inv in invoices if inv.status == "paid")
    cost = sum(ACTION_COST.get(rec.action, 0.0) for rec in realized if rec.action != "none")
    return {"total_recovered": total_recovered, "cost": cost, "net_recovery": total_recovered - cost}


def build_table1(metrics_by_arm: dict[str, dict]) -> pd.DataFrame:
    no_int = metrics_by_arm["No-Intervention"]["net_recovery"]
    naive = metrics_by_arm["Naive-Uniform"]["net_recovery"]
    rows = []
    for arm, m in metrics_by_arm.items():
        rows.append({
            "Arm": arm,
            "Total ₹ Recovered": round(m["total_recovered"], 2),
            "Cost": round(m["cost"], 2),
            "Net Recovery": round(m["net_recovery"], 2),
            "Incremental vs. No-Intervention": round(m["net_recovery"] - no_int, 2),
            "Incremental vs. Naive": round(m["net_recovery"] - naive, 2),
        })
    return pd.DataFrame(rows)


def customer_prs_bands(decision_records: list[dict]) -> dict[str, str]:
    """The Agent arm's own LATEST observed prs_score per customer (decision
    records are appended in day-ascending order, so last-write-wins gives
    the latest), used as the PRS-band assignment applied uniformly to all
    three arms for that customer in Table 1b. Only arm that computes real
    PRS via build_feature_vector; the other two have no feature vectors at
    all, so this is the only consistent source for the band cut.
    """
    latest_prs: dict[str, float] = {}
    for rec in decision_records:
        latest_prs[rec["customer_id"]] = rec["prs_score"]
    return {cid: prs_band(p) for cid, p in latest_prs.items()}


def build_table1b(bands_by_customer: dict[str, str], engines_by_arm: dict[str, SimulationEngine]) -> pd.DataFrame:
    rows = []
    for band in PRS_BAND_LABELS:
        band_customers = {cid for cid, b in bands_by_customer.items() if b == band}
        metrics = {arm: arm_dollar_metrics(engine, band_customers) for arm, engine in engines_by_arm.items()}
        no_int = metrics["No-Intervention"]["net_recovery"]
        naive = metrics["Naive-Uniform"]["net_recovery"]
        for arm, m in metrics.items():
            rows.append({
                "PRS Band": band, "n_customers": len(band_customers), "Arm": arm,
                "Total ₹ Recovered": round(m["total_recovered"], 2),
                "Cost": round(m["cost"], 2),
                "Net Recovery": round(m["net_recovery"], 2),
                "Incremental vs. No-Intervention": round(m["net_recovery"] - no_int, 2),
                "Incremental vs. Naive": round(m["net_recovery"] - naive, 2),
            })
    return pd.DataFrame(rows)


def build_table2(escalation_records: list[dict], customers_by_id: dict[str, CustomerBehaviorModel]) -> tuple[pd.DataFrame, float]:
    """§14: "% of human-escalated invoices where
    true_would_have_paid_without_further_automation == False" -- computed
    via the COUNTERFACTUAL potential outcome for action="none" at the exact
    (invoice, day, context) the escalation decision was made, not from any
    realized/model-believed value.
    """
    rows = []
    for rec in escalation_records:
        customer_model = customers_by_id[rec["customer_id"]]
        potential = customer_model.generate_potential_outcomes(rec["invoice_id"], rec["day"], rec["context"])
        would_have_paid_anyway = bool(potential["none"].will_pay_within_N)
        rows.append({
            "invoice_id": rec["invoice_id"], "customer_id": rec["customer_id"], "day": rec["day"],
            "true_would_have_paid_without_further_automation": would_have_paid_anyway,
        })
    df = pd.DataFrame(rows)
    if len(df):
        precision = float((~df["true_would_have_paid_without_further_automation"]).mean())
    else:
        precision = float("nan")
    return df, precision


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart1_net_recovery(table1: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#9e9e9e", "#5b8def", "#2e7d32"]
    bars = ax.bar(table1["Arm"], table1["Net Recovery"], color=colors[: len(table1)])
    ax.set_ylabel("Net Recovery (₹)")
    ax.set_title("Net Recovery by Policy Arm")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, table1["Net Recovery"]):
        ax.annotate(f"₹{value:,.0f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4 if value >= 0 else -14), ha="center", fontsize=9)
    plt.xticks(rotation=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart2_prs_trajectory(session, customer_ids: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_days = list(range(0, N_DAYS, 5))
    for cid in customer_ids:
        prs_values = [compute_prs(cid, as_of=REFERENCE_START + dt.timedelta(days=d), session=session) for d in sample_days]
        line, = ax.plot(sample_days, prs_values, marker="o", markersize=3, label=cid)

        promises = (
            session.query(Promise)
            .join(Invoice, Promise.invoice_id == Invoice.invoice_id)
            .filter(Invoice.customer_id == cid, Promise.kept.isnot(None))
            .all()
        )
        for p in promises:
            d = (p.made_on - REFERENCE_START).days
            prs_at_d = compute_prs(cid, as_of=p.made_on, session=session)
            marker = "^" if p.kept else "v"
            color = "green" if p.kept else "red"
            ax.scatter([d], [prs_at_d], color=color, marker=marker, s=90, zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Simulated day")
    ax.set_ylabel("PRS")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("PRS trajectory (▲ promise kept, ▼ promise broken)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_illustrative_customers(session, latents: list[CustomerLatentState]) -> list[str]:
    """2-3 customers with visibly different archetypes AND actual promise
    history in this run (so the chart has annotations to show)."""
    wanted_archetypes = ["model_citizen", "serial_promiser", "degrading"]
    picks = []
    for archetype in wanted_archetypes:
        candidates = [l.customer_id for l in latents if l.archetype == archetype]
        best, best_n = None, -1
        for cid in candidates:
            n = session.query(Promise).join(Invoice, Promise.invoice_id == Invoice.invoice_id).filter(
                Invoice.customer_id == cid, Promise.kept.isnot(None)
            ).count()
            if n > best_n:
                best, best_n = cid, n
        if best is not None:
            picks.append(best)
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building held-out eval population: {N_CUSTOMERS} customers, seed={EVAL_SEED}, "
          f"fair unweighted archetype mix (id prefix 'eval-', disjoint from Stage 3's 'train-')\n")
    latents = build_eval_latents(EVAL_SEED, N_CUSTOMERS)

    print("Loading Stage 3's trained PropensityModel...")
    model = PropensityModel()

    print("Running arm 1/3: No-Intervention...")
    engine_none = run_no_intervention(latents)
    print("Running arm 2/3: Naive-Uniform...")
    engine_naive = run_naive_uniform(latents)
    print("Running arm 3/3: Promise Integrity Agent...")
    engine_agent, agent_session, decision_records, escalation_records, customers_by_id = run_agent(latents, model)

    engines_by_arm = {"No-Intervention": engine_none, "Naive-Uniform": engine_naive, "Promise Integrity Agent": engine_agent}

    metrics_by_arm = {arm: arm_dollar_metrics(engine) for arm, engine in engines_by_arm.items()}
    table1 = build_table1(metrics_by_arm)
    print("\n=== Table 1: 3-arm comparison ===")
    print(table1.to_string(index=False))
    table1.to_csv(RESULTS_DIR / "table1_three_arm_comparison.csv", index=False)

    bands = customer_prs_bands(decision_records)
    n_unbanded = N_CUSTOMERS - len(bands)
    table1b = build_table1b(bands, engines_by_arm)
    print(f"\n=== Table 1b: 3-arm comparison by PRS band ({n_unbanded} customers had no Agent-arm "
          f"decision recorded -- e.g. paid before any decision cadence day -- and are excluded) ===")
    print(table1b.to_string(index=False))
    table1b.to_csv(RESULTS_DIR / "table1b_by_prs_band.csv", index=False)

    table2, precision = build_table2(escalation_records, customers_by_id)
    print(f"\n=== Table 2: Escalation precision ===")
    print(f"{len(table2)} escalation decisions across {table2['customer_id'].nunique() if len(table2) else 0} distinct customers "
          f"(a customer whose broken_promise_streak/contact count stays over threshold gets re-flagged every "
          f"decision-cadence day -- there's no 'stop processing once escalated' state, so these are not all "
          f"independent business events; see the first-escalation-only figure below for a more conservative read).")
    if len(table2):
        print(table2.head(10).to_string(index=False), f"\n... ({len(table2) - 10} more rows in the saved CSV)" if len(table2) > 10 else "")
        first_only = table2.sort_values("day").groupby("customer_id").first()
        first_precision = float((~first_only["true_would_have_paid_without_further_automation"]).mean())
        print(f"\nEscalation precision, all {len(table2)} escalation decisions: {precision:.1%}")
        print(f"Escalation precision, first escalation per customer only (n={len(first_only)}): {first_precision:.1%}")
    else:
        print("No escalations occurred in this run.")
    table2.to_csv(RESULTS_DIR / "table2_escalation_precision.csv", index=False)

    chart1_net_recovery(table1, RESULTS_DIR / "chart1_net_recovery.png")
    print(f"\nSaved Chart 1 -> {RESULTS_DIR / 'chart1_net_recovery.png'}")

    illustrative = pick_illustrative_customers(agent_session, latents)
    chart2_prs_trajectory(agent_session, illustrative, RESULTS_DIR / "chart2_prs_trajectory.png")
    print(f"Saved Chart 2 -> {RESULTS_DIR / 'chart2_prs_trajectory.png'} (customers: {illustrative})")


if __name__ == "__main__":
    main()
