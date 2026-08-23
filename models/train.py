"""
models/train.py — logging policy + training data generation + S-learner
GBM training (contract §7, architecture §2b/§6).

Bridges the two Stage-1/Stage-2 leaf modules that were built independently:
simulator/ (pure in-memory rollout, no DB) and features/feature_engine.py
(reads only from backend.db). Generating realistic training rows means
syncing the simulator's realized events into a DB as the rollout advances,
so build_feature_vector() can be called exactly the way the live agent
would call it. That bridge lives here, not in simulator/ or features/,
since both of those must stay frozen, dependency-light leaf modules per the
Module interfaces section -- this script is explicitly allowed to be the
one place that reaches across modules (same category as agent/ and eval/).

Known Stage-1 gap, reported honestly rather than hidden: the simulator
(simulator/behavior_model.py) doesn't model disputes, partial payments, or
a "promise still pending" state -- PotentialOutcome has no fields for any
of these. That means `dispute_rate`, `partial_payment_rate`,
`active_promise_flag`, and `days_until_promised_date` are constant
(0.0 / 0.0 / False / -1) across every row of this synthetic dataset. The
GBM literally cannot learn anything from those four columns yet. This is a
real limitation of the training data, not a bug in this script -- flagged
in the training report rather than silently produced.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base, Communication, Customer, Invoice, Payment, Promise
from features.feature_engine import FEATURE_COLUMNS, TARGET, build_feature_vector
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel
from simulator.engine import SimInvoice, SimulationEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 7
N_CUSTOMERS = 750          # bumped from 320 (then 600): stratified coverage of higher-lateness
                            # buckets (see TRAINING_ARCHETYPE_WEIGHTS) needs more raw decision volume
N_DAYS = 250                # bumped from 150 (then 220): the slowest archetypes need this much
                            # runway to actually reach relative_lateness > 1.5 before the window ends
DECISION_CADENCE_DAYS = 6  # a real decision (and training row) is only made every K days per
                            # invoice; every other day the logging policy returns "none" cheaply
                            # without touching the DB. Rollout stays daily (matches Stage 1's
                            # engine.step() semantics and lets spontaneous events fire every day);
                            # only the *logged decision points* are subsampled. This keeps the
                            # dataset in the "low thousands of rows" range the architecture doc
                            # (§2b) explicitly expects, instead of one row per invoice per day.
HEURISTIC_PROB = 0.70       # §7: ~70% heuristic, ~30% uniformly random across eligible actions
DEV_FRACTION = 0.20         # customer-level split (architecture §6)
REFERENCE_START = dt.datetime(2025, 1, 1)  # simulator day 0 maps to this calendar date

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]

# §7: intervention_type's possible values explicitly exclude human_escalation
# ("not a 'pay probability' decision, always policy-forced") -- the logging
# policy therefore never assigns it, and it is not a model feature value.
LOGGED_ACTIONS = ["none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend", "plan_proposal"]

# Archetype mix for TRAINING DATA GENERATION only -- deliberately skewed
# toward archetypes that spend more time overdue (higher avg_days_to_pay /
# lower keep rates), so decision points actually land in the higher
# relative_lateness buckets often enough to get adequate coverage there
# (see the lateness-bucket stratification below). §4's archetype table is
# otherwise an equal 7-way mix; that's still what Stage 6's eval harness
# must use -- eval measures recovery on a realistic customer base, and
# would be dishonest if run against this skewed population instead. This
# weighting is local to this file's population-builder and never touches
# simulator/archetypes.py's own (unweighted) ARCHETYPE_NAMES.
TRAINING_ARCHETYPE_WEIGHTS = {
    "model_citizen": 0.05,
    "reliable_always_late": 0.10,
    "cash_flow_strained_genuine": 0.15,
    "disputer": 0.15,
    "degrading": 0.15,
    "serial_promiser": 0.20,
    "non_responsive": 0.20,
}

# Finer relative_lateness buckets than risk_tier's single near_term/aging
# split, used only for the stratified-coverage mechanism and its report
# table below. Bucket 0 (<=0.3) is the dominant, already-saturated region;
# stratified forcing only ever targets buckets 1-4.
LATENESS_BUCKET_EDGES = [0.3, 0.6, 1.0, 1.5]
LATENESS_BUCKET_LABELS = ["<=0.3", "(0.3,0.6]", "(0.6,1.0]", "(1.0,1.5]", ">1.5"]
MIN_CELL_COUNT = 45          # minimum rows per (lateness bucket >= 1, action) cell
STRATIFIED_FORCE_PROB = 0.97  # how often an under-filled cell is deliberately targeted, once found


def lateness_bucket_index(relative_lateness: float) -> int:
    for i, edge in enumerate(LATENESS_BUCKET_EDGES):
        if relative_lateness <= edge:
            return i
    return len(LATENESS_BUCKET_EDGES)


RISK_TIER_LATENESS_THRESHOLD = 1.0

# §6/architecture §2 give N=7 (near-term) / N=21 (aging) as an illustrative
# "e.g." example, not literal locked values. Empirically, on THIS simulator,
# those values produce a degenerate target: N=7 gives a 0.4% positive rate
# (15/4008 near-term rows), unusable for training. Root cause, traced via
# scripts probing generate_potential_outcomes(): a "hit" (will_pay_within_N
# firing True) samples `days_to_pay` centered near Stage 1's own internal
# horizon (DEFAULT_HORIZON_DAYS=21, behavior_model.py) rather than near 0 --
# an emergent property of the fixed-target-day + edge-fuzz mechanism (the
# Bernoulli fires as soon as the target day enters the [day, day+21) window,
# at which point days_to_pay = effective_payment_day - day is itself close
# to 21, not close to 0). So the realized gap-to-actual-payment clusters
# around 20-30 days, not 0-7. Sweeping N against the actual gap distribution
# (see PR discussion) showed a sharp jump at N=30 (near-term: 12% positive
# at N=21 -> 47% at N=30) -- recalibrated to N=30/45 below, which keeps
# "aging gets the wider window" (30 < 45) but produces a workable target
# instead of a degenerate one on this specific simulator.
RISK_TIER_HORIZON = {"near_term": 30, "aging": 45}
PLAN_HEURISTIC_FLOOR = 0.5                          # mirrors PLAN_ELIGIBILITY_FLOOR (§9)

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "propensity_model.joblib"
TRAINING_DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "training_data.csv"


# ---------------------------------------------------------------------------
# Simulator <-> DB bridge
# ---------------------------------------------------------------------------

def make_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed_customers(session, customers: list[CustomerBehaviorModel], rng: np.random.Generator) -> None:
    for c in customers:
        session.add(Customer(
            customer_id=c.latent.customer_id,
            name=c.latent.customer_id,
            archetype=c.latent.archetype,          # ground truth, stored but never read by feature_engine/model
            segment=str(rng.choice(SEGMENTS)),      # deliberately independent of archetype -- see module docstring
            credit_terms_days=30,
            onboarding_date=REFERENCE_START - dt.timedelta(days=365),
        ))
    session.commit()


def sync_new_invoices(session, engine: SimulationEngine, synced_ids: set[str], rng: np.random.Generator) -> None:
    """§3's SimInvoice carries no dollar amount (the behavior model never
    uses one). Amounts are synthesized here, at DB-sync time, log-normal
    per customer with per-invoice noise -- needed for amount_tier and,
    later, EIV, but invented at this bridge layer rather than in the
    simulator, which stays untouched.
    """
    for inv in engine.invoices:
        if inv.invoice_id in synced_ids:
            continue
        customer_scale = 8000.0 * float(np.exp(rng.normal(0, 0.5)))
        amount = round(customer_scale * float(np.exp(rng.normal(0, 0.25))), 2)
        session.add(Invoice(
            invoice_id=inv.invoice_id,
            customer_id=inv.customer_id,
            amount=amount,
            issue_date=REFERENCE_START + dt.timedelta(days=inv.issue_day),
            due_date=REFERENCE_START + dt.timedelta(days=inv.due_day),
            status="open",
        ))
        synced_ids.add(inv.invoice_id)
    session.commit()


def sync_realized_day(session, engine: SimulationEngine, day: int) -> None:
    """Writes today's realized events (from engine.realized_log, filtered to
    `day`) into Communication/Promise/Payment rows and updates Invoice
    status -- mirrors what a live agent's dispatch/verify steps would write.
    """
    today_records = [r for r in engine.realized_log if r.day == day]
    for rec in today_records:
        as_of_date = REFERENCE_START + dt.timedelta(days=day)
        outcome = rec.outcome

        if rec.action != "none":
            session.add(Communication(
                comm_id=f"comm-{rec.invoice_id}-{day}",
                invoice_id=rec.invoice_id,
                channel="email",
                timestamp=as_of_date,
                message_type=rec.action,
                message_text="",
                dispatched_by="agent",
                response_received=outcome.will_respond,
                response_text=None,
            ))

        if outcome.will_promise:
            session.add(Promise(
                promise_id=f"prom-{rec.invoice_id}-{day}",
                invoice_id=rec.invoice_id,
                promised_date=as_of_date + dt.timedelta(days=7),
                promised_amount=None,
                made_on=as_of_date,
                extraction_confidence=0.9,
                kept=outcome.promise_kept,   # Stage 1 always resolves promises the same day -- see module docstring
            ))

        if outcome.will_pay_within_N:
            invoice = session.get(Invoice, rec.invoice_id)
            payment_date = as_of_date + dt.timedelta(days=outcome.days_to_pay or 0)
            session.add(Payment(
                payment_id=f"pay-{rec.invoice_id}",
                invoice_id=rec.invoice_id,
                amount_paid=invoice.amount,
                payment_date=payment_date,
                partial_flag=False,   # Stage 1 doesn't model partial payments -- see module docstring
            ))
            invoice.status = "paid"

    session.commit()


# ---------------------------------------------------------------------------
# Logging policy (§7)
# ---------------------------------------------------------------------------

def heuristic_action(features: dict) -> str:
    """A deliberately simple risk-based heuristic -- NOT the production
    policy engine (that's policy/constraints.py + policy/eiv.py, a later
    stage). Escalates contact intensity with relative_lateness, and only
    proposes a payment plan for customers whose PRS clears a floor.
    """
    lateness = features["relative_lateness"]
    prs = features["prs_score"]
    if lateness < 0.3:
        return "none"
    if lateness < 0.8:
        return "soft_reminder"
    if lateness < 1.5:
        return "firm_reminder"
    if lateness < 2.2:
        return "channel_escalation" if prs < PLAN_HEURISTIC_FLOOR else "firm_reminder"
    return "plan_proposal" if prs >= PLAN_HEURISTIC_FLOOR else "channel_escalation"


def make_logging_policy(session, pending_rows: dict, rng: np.random.Generator):
    """Returns a policy_fn(invoice, day, context) -> Action for
    SimulationEngine.run(). Three-way action selection, in priority order:

      1. Stratified coverage forcing: if this decision's relative_lateness
         bucket is >= 1 (i.e. not the dominant near-zero-lateness region)
         and it still has an under-filled (bucket, action) cell -- fewer
         than MIN_CELL_COUNT rows logged so far, including "none" -- target
         one of those under-filled actions with probability
         STRATIFIED_FORCE_PROB. This is the direct fix for a diagnosed
         problem: pure 70/30 heuristic/random exploration left some
         higher-lateness (bucket, action) cells at 0-6 rows, because so few
         decision points land there at all and the heuristic branch skews
         hard toward contact actions once lateness is high -- "none" in
         particular was starved exactly where EIV needs P(pay | none)
         estimated well.
      2. Otherwise, ~HEURISTIC_PROB follows heuristic_action().
      3. Otherwise, uniformly random across the full LOGGED_ACTIONS set --
         unrestricted, which is what prevents "risky customers only ever
         see aggressive interventions in the training data" (architecture
         §2b's confounding trap) in the dominant low-lateness region.

    Every decision made (on a cadence day) is recorded into `pending_rows`,
    keyed by (invoice_id, day), for the training loop to pick up after
    realize().
    """
    bucket_action_counts: dict[tuple[int, str], int] = defaultdict(int)

    def policy_fn(invoice: SimInvoice, day: int, context: dict) -> str:
        if day % DECISION_CADENCE_DAYS != 0:
            return "none"

        as_of = REFERENCE_START + dt.timedelta(days=day)
        # intervention_type doesn't affect any other field build_feature_vector
        # computes -- query once with a placeholder, then overwrite it below
        # with whichever action actually gets chosen. Avoids a second DB round-trip.
        features = build_feature_vector(invoice.invoice_id, "none", as_of=as_of, session=session)
        risk_tier = "near_term" if features["relative_lateness"] < RISK_TIER_LATENESS_THRESHOLD else "aging"
        bucket = lateness_bucket_index(features["relative_lateness"])

        under_filled = [a for a in LOGGED_ACTIONS if bucket_action_counts[(bucket, a)] < MIN_CELL_COUNT]

        if bucket >= 1 and under_filled and rng.random() < STRATIFIED_FORCE_PROB:
            # Target the single most under-filled action specifically, not
            # a uniform pick among all under-filled ones -- a uniform pick
            # dilutes attention across up to 6 simultaneously-behind cells,
            # which is what left "none" undershooting even with forcing on.
            action = min(under_filled, key=lambda a: bucket_action_counts[(bucket, a)])
        elif rng.random() < HEURISTIC_PROB:
            action = heuristic_action(features)
        else:
            action = str(rng.choice(LOGGED_ACTIONS))

        bucket_action_counts[(bucket, action)] += 1
        features["intervention_type"] = action
        pending_rows[(invoice.invoice_id, day)] = {
            "features": features,
            "risk_tier": risk_tier,
            "lateness_bucket": LATENESS_BUCKET_LABELS[bucket],
            "horizon_days": RISK_TIER_HORIZON[risk_tier],
            "customer_id": invoice.customer_id,
        }
        return action

    return policy_fn


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def build_training_population(seed: int, n_customers: int) -> list[CustomerBehaviorModel]:
    """Weighted toward slower/riskier archetypes -- see
    TRAINING_ARCHETYPE_WEIGHTS's docstring for why this is safe only for
    training-data generation, never for eval."""
    latent_rng = np.random.default_rng(seed)
    archetype_rng = np.random.default_rng(seed + 1)
    names = list(TRAINING_ARCHETYPE_WEIGHTS.keys())
    weights = np.array([TRAINING_ARCHETYPE_WEIGHTS[n] for n in names])
    weights = weights / weights.sum()

    customers = []
    for i in range(n_customers):
        archetype = str(archetype_rng.choice(names, p=weights))
        customer_id = f"train-{i:04d}-{archetype}"
        latent = sample_customer_latent(customer_id, archetype, latent_rng)
        customers.append(CustomerBehaviorModel(latent))
    return customers


def generate_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    customers = build_training_population(SEED, N_CUSTOMERS)

    session = make_db_session()
    seed_customers(session, customers, rng)

    engine = SimulationEngine(customers, seed=SEED)
    synced_invoice_ids: set[str] = set()
    pending_rows: dict[tuple[str, int], dict] = {}
    policy_fn = make_logging_policy(session, pending_rows, rng)

    for day in range(N_DAYS):
        sync_new_invoices(session, engine, synced_invoice_ids, rng)
        engine.step(day, policy_fn)
        sync_realized_day(session, engine, day)

    # Target labeling: use each invoice's REALIZED closure (invoice.paid_day,
    # authoritatively tracked by the engine across the whole rollout), not
    # the single-day PotentialOutcome.will_pay_within_N sampled at decision
    # time. Those two are not interchangeable here: generate_potential_outcomes()
    # tests, independently on every calendar day, whether a fixed per-invoice
    # target payment day falls inside a fresh [day, day+horizon) window --
    # correct in aggregate (validated in Stage 1), but that per-day test is
    # its own independent coin flip, decoupled from whichever day actually
    # ends up closing the invoice during the sequential rollout. Concretely:
    # for a reliable_always_late customer with baseline_payment_day~30, a
    # probe showed will_pay_within_21 independently sampled False on every
    # day from 0-20, then True at day 22 -- so an invoice that (correctly)
    # closes on day ~22 would still get "paid_within_21=False" labeled at a
    # day-18 decision point 4 days earlier, purely from bad luck on an
    # unrelated coin flip. Using the invoice's own actual paid_day instead
    # asks the well-posed question directly: did *this* invoice, on *its*
    # real trajectory, close within N days of *this* decision point.
    invoice_by_id = {inv.invoice_id: inv for inv in engine.invoices}
    rows = []
    for (invoice_id, day), pending in pending_rows.items():
        invoice = invoice_by_id[invoice_id]
        n = pending["horizon_days"]
        paid_within_n = bool(
            invoice.status == "paid" and invoice.paid_day is not None and (invoice.paid_day - day) <= n
        )
        row = dict(pending["features"])
        row[TARGET] = paid_within_n
        row["customer_id"] = pending["customer_id"]
        row["risk_tier"] = pending["risk_tier"]
        row["lateness_bucket"] = pending["lateness_bucket"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Split, train, evaluate
# ---------------------------------------------------------------------------

CATEGORICAL_COLUMNS = ["segment", "intervention_type"]


def customer_level_split(df: pd.DataFrame, dev_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_ids = df["customer_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(customer_ids)
    n_dev = max(1, int(len(customer_ids) * dev_fraction))
    dev_ids = set(customer_ids[:n_dev])
    train_ids = set(customer_ids[n_dev:])
    assert train_ids.isdisjoint(dev_ids)
    train_df = df[df["customer_id"].isin(train_ids)].reset_index(drop=True)
    dev_df = df[df["customer_id"].isin(dev_ids)].reset_index(drop=True)
    assert set(train_df["customer_id"]).isdisjoint(set(dev_df["customer_id"]))
    return train_df, dev_df


def prep_categoricals(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    df["active_promise_flag"] = df["active_promise_flag"].astype(int)
    return df


def train_model(train_df: pd.DataFrame, categories: dict[str, list]) -> lgb.LGBMClassifier:
    X = prep_categoricals(train_df, categories)[FEATURE_COLUMNS]
    y = train_df[TARGET].astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=15,
        random_state=SEED, min_child_samples=10, verbosity=-1,
    )
    model.fit(X, y, categorical_feature=CATEGORICAL_COLUMNS)
    return model


def evaluate_model(model: lgb.LGBMClassifier, dev_df: pd.DataFrame, categories: dict[str, list]) -> tuple[float, pd.DataFrame]:
    X_dev = prep_categoricals(dev_df, categories)[FEATURE_COLUMNS]
    y_dev = dev_df[TARGET].astype(int)
    proba = model.predict_proba(X_dev)[:, 1]

    auc = roc_auc_score(y_dev, proba) if y_dev.nunique() > 1 else float("nan")

    cal_df = pd.DataFrame({"y_true": y_dev.values, "y_pred": proba})
    n_bins = min(10, cal_df["y_pred"].nunique())
    cal_df["bin"] = pd.qcut(cal_df["y_pred"], n_bins, duplicates="drop")
    calibration = cal_df.groupby("bin", observed=True).agg(
        n=("y_true", "size"), mean_predicted=("y_pred", "mean"), mean_actual=("y_true", "mean"),
    )
    return auc, calibration


def joint_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["risk_tier"], df["intervention_type"])


def lateness_bucket_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    ordered = pd.Categorical(df["lateness_bucket"], categories=LATENESS_BUCKET_LABELS, ordered=True)
    return pd.crosstab(ordered, df["intervention_type"]).reindex(columns=LOGGED_ACTIONS)


def check_bucket_coverage(df: pd.DataFrame) -> list[str]:
    """Returns a list of (bucket, action) cells among buckets >= 1 still
    below MIN_CELL_COUNT, for honest reporting -- not an assertion, since
    the rarest bucket for the rarest action may legitimately stay just
    under the target even with stratified forcing turned up.
    """
    table = lateness_bucket_coverage_table(df)
    shortfalls = []
    for bucket in LATENESS_BUCKET_LABELS[1:]:
        for action in LOGGED_ACTIONS:
            n = int(table.loc[bucket, action]) if bucket in table.index else 0
            if n < MIN_CELL_COUNT:
                shortfalls.append(f"{bucket} x {action}: {n} rows (< {MIN_CELL_COUNT})")
    return shortfalls


def save_artifact(model: lgb.LGBMClassifier, categories: dict[str, list]) -> None:
    import joblib
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "categories": categories, "feature_columns": FEATURE_COLUMNS}, ARTIFACT_PATH)


def action_distribution_check(df: pd.DataFrame) -> pd.Series:
    dist = df["intervention_type"].value_counts(normalize=True)
    assert dist.max() < 0.90, f"logging policy is degenerate: {dist.idxmax()} is {dist.max():.1%} of assignments"
    return dist


def main() -> None:
    print(f"Generating training data: {N_CUSTOMERS} customers, {N_DAYS} days, "
          f"decision cadence={DECISION_CADENCE_DAYS}d, seed={SEED}\n")
    df = generate_training_data()
    print(f"Generated {len(df)} rows across {df['customer_id'].nunique()} customers.\n")

    TRAINING_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_CSV, index=False)
    print(f"Saved raw training data -> {TRAINING_DATA_CSV}\n")

    print("=== Logging-policy action distribution (all rows) ===")
    dist = action_distribution_check(df)
    print(dist.to_string(), "\n")

    print("=== Joint distribution: risk_tier x intervention_type (counts) ===")
    joint = joint_distribution_table(df)
    print(joint.to_string(), "\n")
    print("=== Joint distribution: risk_tier x intervention_type (row %) ===")
    print((joint.div(joint.sum(axis=1), axis=0) * 100).round(1).to_string(), "\n")

    print(f"=== Bucket coverage: relative_lateness bucket x intervention_type (counts, target >= {MIN_CELL_COUNT} for bucket>=1) ===")
    bucket_coverage = lateness_bucket_coverage_table(df)
    print(bucket_coverage.to_string(), "\n")
    shortfalls = check_bucket_coverage(df)
    if shortfalls:
        print(f"Cells still below target ({len(shortfalls)}):")
        for s in shortfalls:
            print(f"  - {s}")
    else:
        print("Every (bucket, action) cell for bucket >= 1 meets the minimum.")
    print()

    categories = {c: sorted(df[c].unique()) for c in CATEGORICAL_COLUMNS}
    train_df, dev_df = customer_level_split(df, DEV_FRACTION, SEED)
    print(f"Split: {len(train_df)} train rows / {train_df['customer_id'].nunique()} customers, "
          f"{len(dev_df)} dev rows / {dev_df['customer_id'].nunique()} customers\n")

    print(f"paid_within_N base rate -- train: {train_df[TARGET].mean():.3f}, dev: {dev_df[TARGET].mean():.3f}\n")

    model = train_model(train_df, categories)
    auc, calibration = evaluate_model(model, dev_df, categories)

    print(f"=== Dev-set AUC: {auc:.4f} ===\n")
    print("=== Dev-set calibration (deciles of predicted probability) ===")
    print(calibration.to_string(), "\n")

    save_artifact(model, categories)
    print(f"Saved model artifact -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
