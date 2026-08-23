"""
eval/scenarios.py — Table 3 (contract §14): failure-injection scenario
pass/fail table. 4 of the original 9 §9 policy-engine rules plus Stage 8's
extraction-refusal case (the hallucination risk architecture §20 calls out
explicitly), each run through the REAL Stage 7 agent loop
(agent.state_machine.run_agent_cycle / process_customer_reply) against a
purpose-built invoice/customer DB state -- not policy/constraints.py's
evaluate_constraints() called in isolation with a hand-built PolicyState
(that's what tests/test_policy_engine.py already covers).

Every scenario asserts the SPECIFIC rationale_code it claims to test, not a
coarser behavioral proxy like "the invoice wasn't contacted" -- the same
class of gap Stage 9's audit-completeness checker was built to catch: a
scenario could superficially pass for the wrong reason (a different rule
coincidentally producing a similar-looking outcome, e.g. FREQUENCY_CAP
suppressing contact instead of the dispute rule actually firing) if the
assertion only checks the outcome and not the mechanism.

--- A Stage-12 fix was required first ---
agent/state_machine.py's _build_policy_state previously hardcoded
dispute_resolved=True and no_contact_requested=False unconditionally -- a
documented, deliberate Stage-1 scope simplification (the identical
simplification is separately documented in eval/run_eval.py's own comment)
-- meaning rules 1 (DISPUTE_UNRESOLVED) and 2 (NO_CONTACT_HONORED) could
never fire through the real agent loop, no matter what DB state a scenario
constructed. Fixed as part of this stage, per explicit user direction:
dispute_resolved is now derived from the frozen §2 schema's own
(previously-unread) Dispute table, and no_contact_requested from a new
AgentContext.no_contact_customer_ids set, mirroring Stage 10's
live_slice_invoice_ids pattern exactly -- no schema change, no invented DB
column, and every pre-existing test still passes unchanged (both fields
default to "off", same as before).

Run standalone:  python3 eval/scenarios.py
Writes eval/results/table3_failure_injection.csv (Table 3 proper) and
eval/results/table3_failure_injection_trails.json (the full real AuditLog
trail behind each scenario, for independent verification that the
rationale_code in the trail actually matches what the scenario claims).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

from agent.state_machine import AgentContext, process_customer_reply, run_agent_cycle
from agent.tools import AuditLogTool, PropensityModelTool
from backend.db import AuditLog, Base, Customer, Dispute, Invoice, Promise
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class ScenarioResult:
    name: str
    rule_under_test: str          # the exact rationale_code(s) this scenario must observe to count as a real pass
    expected: str
    observed: str
    passed: bool
    audit_trail: list[dict] = field(default_factory=list)
    notes: str = ""


def _seq(log_id: str) -> int:
    return int(log_id.rsplit("-", 1)[-1])


def _fresh_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _events(session, invoice_id: str) -> list[AuditLog]:
    rows = session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()
    rows.sort(key=lambda r: _seq(r.log_id))
    return rows


def _find_step(events: list[AuditLog], step: str) -> AuditLog | None:
    matches = [e for e in events if e.step == step]
    return matches[0] if matches else None


def _routine_act_event(events: list[AuditLog], decide_event: AuditLog | None) -> AuditLog | None:
    """The per-cycle "act" summary event always carries the same
    rationale_code the decide event authorized (rationale_joined ==
    rationale_joined) -- this is how it's distinguished from any EXTRA act
    event a cycle might also write (e.g. _record_promise's
    SPONTANEOUS_PROMISE_CAPTURED), rather than just grabbing the first
    step=="act" row, which could silently grab the wrong one."""
    if decide_event is None:
        return None
    for e in events:
        if e.step == "act" and e.rationale_code == decide_event.rationale_code:
            return e
    return None


def _serialize(e: AuditLog) -> dict:
    return {
        "log_id": e.log_id, "seq": _seq(e.log_id), "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        "step": e.step, "decision": e.decision, "executed_action": e.executed_action,
        "rationale_code": e.rationale_code, "constraint_triggered": e.constraint_triggered,
        "human_approval_required": bool(e.human_approval_required),
    }


def _seed_customer(session, customer_id: str, archetype: str, onboarding_days_ago: int = 100) -> CustomerBehaviorModel:
    rng = np.random.default_rng(abs(hash(customer_id)) % (2**32))
    latent = sample_customer_latent(customer_id, archetype, rng)
    session.add(Customer(
        customer_id=customer_id, name=customer_id, archetype=archetype, segment="SMB",
        credit_terms_days=15, onboarding_date=REFERENCE_START - dt.timedelta(days=onboarding_days_ago),
    ))
    return CustomerBehaviorModel(latent)


def _make_ctx(session, day: int, customer_model: CustomerBehaviorModel, no_contact_customer_ids=None) -> AgentContext:
    return AgentContext(
        day=day, reference_start=REFERENCE_START, session=session, customer_model=customer_model,
        propensity_model_tool=PropensityModelTool(PropensityModel()), policy_config=CONFIG,
        no_contact_customer_ids=no_contact_customer_ids or set(),
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Dispute-stop (rule 1: DISPUTE_UNRESOLVED)
# ---------------------------------------------------------------------------

def scenario_dispute_stop() -> ScenarioResult:
    session = _fresh_session()
    customer_id, invoice_id = "scn-dispute-01", "inv-scn-dispute-01-0"
    model = _seed_customer(session, customer_id, "model_citizen")
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=20_000.0,
        issue_date=REFERENCE_START - dt.timedelta(days=20), due_date=REFERENCE_START - dt.timedelta(days=5),
        status="open", dispute_flag=True, dispute_reason="quality dispute",
    ))
    session.add(Dispute(
        dispute_id=f"disp-{invoice_id}", invoice_id=invoice_id,
        raised_date=REFERENCE_START - dt.timedelta(days=10), reason="quality dispute", resolved=False,
    ))
    session.commit()

    ctx = _make_ctx(session, day=0, customer_model=model)
    run_agent_cycle(invoice_id, ctx)

    events = _events(session, invoice_id)
    decide, act = _find_step(events, "decide"), None
    act = _routine_act_event(events, decide)
    invoice = session.get(Invoice, invoice_id)

    passed = bool(
        decide is not None and decide.rationale_code == "DISPUTE_UNRESOLVED" and decide.decision == "human_escalation"
        and act is not None and act.rationale_code == "DISPUTE_UNRESOLVED" and act.executed_action == "human_escalation"
        and invoice.status == "human_queue"
    )
    return ScenarioResult(
        name="Dispute-stop",
        rule_under_test="DISPUTE_UNRESOLVED (rule 1)",
        expected="An unresolved dispute (Dispute.resolved=False) on the invoice forces decide=human_escalation with "
                 "rationale_code=DISPUTE_UNRESOLVED (checked first, ahead of every other rule), invoice.status -> human_queue.",
        observed=f"decide: decision={decide.decision if decide else None}, rationale_code={decide.rationale_code if decide else None} | "
                 f"act: executed_action={act.executed_action if act else None}, rationale_code={act.rationale_code if act else None} | "
                 f"invoice.status={invoice.status}",
        passed=passed,
        audit_trail=[_serialize(e) for e in events],
    )


# ---------------------------------------------------------------------------
# Scenario 2 — No-contact-request honored (rule 2: NO_CONTACT_HONORED)
# ---------------------------------------------------------------------------

def scenario_no_contact_honored() -> ScenarioResult:
    session = _fresh_session()
    customer_id, invoice_id = "scn-nocontact-01", "inv-scn-nocontact-01-0"
    model = _seed_customer(session, customer_id, "serial_promiser")
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=18_000.0,
        issue_date=REFERENCE_START - dt.timedelta(days=25), due_date=REFERENCE_START - dt.timedelta(days=10),
        status="open", dispute_flag=False,
    ))
    session.commit()

    ctx = _make_ctx(session, day=0, customer_model=model, no_contact_customer_ids={customer_id})
    run_agent_cycle(invoice_id, ctx)

    events = _events(session, invoice_id)
    decide = _find_step(events, "decide")
    act = _routine_act_event(events, decide)
    invoice = session.get(Invoice, invoice_id)

    passed = bool(
        decide is not None and decide.rationale_code == "NO_CONTACT_HONORED" and decide.decision == "none"
        and act is not None and act.rationale_code == "NO_CONTACT_HONORED" and act.executed_action == "none"
        and invoice.status == "open"  # forced "none" is not terminal -- the invoice stays open, just untouched
    )
    return ScenarioResult(
        name="No-contact-request honored",
        rule_under_test="NO_CONTACT_HONORED (rule 2)",
        expected="A customer flagged no_contact_requested forces decide=none with rationale_code=NO_CONTACT_HONORED "
                 "(permanent, overrides every rule below it), act executes \"none\" (the literal authorized no-op, "
                 "not a withheld/pending action) -- invoice stays open, not escalated.",
        observed=f"decide: decision={decide.decision if decide else None}, rationale_code={decide.rationale_code if decide else None} | "
                 f"act: executed_action={act.executed_action if act else None}, rationale_code={act.rationale_code if act else None} | "
                 f"invoice.status={invoice.status}",
        passed=passed,
        audit_trail=[_serialize(e) for e in events],
    )


# ---------------------------------------------------------------------------
# Scenario 3 — Broken-promise-streak escalation (rule 4: PROMISE_STREAK_EXCEEDED)
# ---------------------------------------------------------------------------

def scenario_broken_promise_streak_escalation() -> ScenarioResult:
    session = _fresh_session()
    customer_id, invoice_id = "scn-streak-01", "inv-scn-streak-01-0"
    model = _seed_customer(session, customer_id, "serial_promiser")
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=22_000.0,
        issue_date=REFERENCE_START - dt.timedelta(days=40), due_date=REFERENCE_START - dt.timedelta(days=25),
        status="open",
    ))
    # Two most-recent-by-made_on promises both kept=False -> broken_promise_streak=2 (>=2 fires rule 4).
    # made_on/promised_date are all in the past relative to as_of=REFERENCE_START, so neither promise
    # reads as "active" (no COOLING_PERIOD_ACTIVE from rule 3 masking rule 4).
    session.add(Promise(
        promise_id=f"prom-{invoice_id}-a", invoice_id=invoice_id,
        promised_date=REFERENCE_START - dt.timedelta(days=20), made_on=REFERENCE_START - dt.timedelta(days=27),
        promised_amount=None, extraction_confidence=0.9, kept=False, broken_reason="not_kept_by_grace_deadline",
    ))
    session.add(Promise(
        promise_id=f"prom-{invoice_id}-b", invoice_id=invoice_id,
        promised_date=REFERENCE_START - dt.timedelta(days=10), made_on=REFERENCE_START - dt.timedelta(days=17),
        promised_amount=None, extraction_confidence=0.9, kept=False, broken_reason="not_kept_by_grace_deadline",
    ))
    session.commit()

    ctx = _make_ctx(session, day=0, customer_model=model)
    run_agent_cycle(invoice_id, ctx)

    events = _events(session, invoice_id)
    decide = _find_step(events, "decide")
    act = _routine_act_event(events, decide)
    invoice = session.get(Invoice, invoice_id)

    passed = bool(
        decide is not None and decide.rationale_code == "PROMISE_STREAK_EXCEEDED" and decide.decision == "human_escalation"
        and act is not None and act.rationale_code == "PROMISE_STREAK_EXCEEDED" and act.executed_action == "human_escalation"
        and invoice.status == "human_queue"
    )
    return ScenarioResult(
        name="Broken-promise-streak escalation",
        rule_under_test="PROMISE_STREAK_EXCEEDED (rule 4)",
        expected="2 consecutive broken promises (broken_promise_streak >= 2) forces decide=human_escalation with "
                 "rationale_code=PROMISE_STREAK_EXCEEDED, invoice.status -> human_queue.",
        observed=f"decide: decision={decide.decision if decide else None}, rationale_code={decide.rationale_code if decide else None} | "
                 f"act: executed_action={act.executed_action if act else None}, rationale_code={act.rationale_code if act else None} | "
                 f"invoice.status={invoice.status}",
        passed=passed,
        audit_trail=[_serialize(e) for e in events],
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Ambiguous reply: extraction correctly refuses to fabricate
# (architecture §20's named hallucination risk; Stage 8's no_commitment_detected /
# confidence-floor fallback)
# ---------------------------------------------------------------------------

def scenario_ambiguous_reply_refuses_to_fabricate() -> ScenarioResult:
    session = _fresh_session()
    customer_id, invoice_id = "scn-ambiguous-01", "inv-scn-ambiguous-01-0"
    model = _seed_customer(session, customer_id, "cash_flow_strained_genuine")
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=15_000.0,
        issue_date=REFERENCE_START - dt.timedelta(days=15), due_date=REFERENCE_START,
        status="open",
    ))
    session.commit()

    ctx = _make_ctx(session, day=0, customer_model=model)
    reply_text = "I got your message, I'll think about it."  # payment-adjacent tone, no date/amount/genuine commitment -- must not be fabricated into one
    expected_codes = {"NO_COMMITMENT_DETECTED", "EXTRACTION_BELOW_CONFIDENCE_FLOOR"}

    try:
        from agent.llm.extract import LLMExtractTool
        process_customer_reply(
            invoice_id, customer_id, reply_text, ctx, REFERENCE_START,
            owed_amount=15_000.0, extract_tool=LLMExtractTool(),
        )
        events = _events(session, invoice_id)
        act = _find_step(events, "act")
        observed_code = act.rationale_code if act else None
        passed = observed_code in expected_codes
        notes = ""
    except Exception as exc:
        events = _events(session, invoice_id)
        observed_code = None
        passed = False
        notes = f"SKIPPED/ERROR -- live Gemini call unavailable: {type(exc).__name__}: {str(exc)[:300]}"

    return ScenarioResult(
        name="Ambiguous reply: extraction refuses to fabricate",
        rule_under_test="NO_COMMITMENT_DETECTED or EXTRACTION_BELOW_CONFIDENCE_FLOOR (§11 extraction, Stage 8)",
        expected=f'Reply "{reply_text}" has no genuine payment commitment (no date/amount, non-committal tone) -- '
                 f"extraction must correctly refuse to fabricate one: rationale_code must be exactly "
                 f"NO_COMMITMENT_DETECTED or EXTRACTION_BELOW_CONFIDENCE_FLOOR, decision=no_promise_captured, no Promise row created.",
        observed=f"act: rationale_code={observed_code}" + (f" | {notes}" if notes else ""),
        passed=passed,
        audit_trail=[_serialize(e) for e in events],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Scenario 5 — High-value invoice requires human approval (rule 7:
# HIGH_VALUE_REQUIRES_APPROVAL) -- regression coverage for the real bug
# Stage 9 found and fixed (rule fired but dispatch wasn't actually gated on it).
# ---------------------------------------------------------------------------

def scenario_high_value_requires_approval() -> ScenarioResult:
    session = _fresh_session()
    customer_id, invoice_id = "scn-highvalue-01", "inv-scn-highvalue-01-0"
    model = _seed_customer(session, customer_id, "cash_flow_strained_genuine")
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=75_000.0,  # >= CONFIG.high_value_threshold (50,000)
        issue_date=REFERENCE_START - dt.timedelta(days=20), due_date=REFERENCE_START - dt.timedelta(days=5),
        status="open",
    ))
    session.commit()

    ctx = _make_ctx(session, day=0, customer_model=model)
    run_agent_cycle(invoice_id, ctx)

    events = _events(session, invoice_id)
    decide = _find_step(events, "decide")
    act = _routine_act_event(events, decide)
    invoice = session.get(Invoice, invoice_id)

    decide_codes = (decide.rationale_code or "").split(",") if decide else []
    passed = bool(
        decide is not None and "HIGH_VALUE_REQUIRES_APPROVAL" in decide_codes and decide.human_approval_required
        and act is not None and act.executed_action is None  # None (withheld), never the string "none" -- see Stage 9's PENDING_APPROVAL_STATUS fix
        and invoice.status == "pending_human_approval"
    )
    return ScenarioResult(
        name="High-value invoice requires human approval",
        rule_under_test="HIGH_VALUE_REQUIRES_APPROVAL (rule 7)",
        expected="Invoice amount >= high_value_threshold (₹50,000) sets human_approval_required=True with "
                 "HIGH_VALUE_REQUIRES_APPROVAL in the rationale trail; act must NOT auto-dispatch (executed_action stays "
                 "null, not the string \"none\"), invoice.status -> pending_human_approval. Regression check for the real "
                 "gap Stage 9 found and fixed (rule fired but dispatch wasn't actually gated on it).",
        observed=f"decide: decision={decide.decision if decide else None}, rationale_code={decide.rationale_code if decide else None}, "
                 f"human_approval_required={decide.human_approval_required if decide else None} | "
                 f"act: executed_action={act.executed_action if act else None!r} | invoice.status={invoice.status}",
        passed=passed,
        audit_trail=[_serialize(e) for e in events],
    )


SCENARIOS = [
    scenario_dispute_stop,
    scenario_no_contact_honored,
    scenario_broken_promise_streak_escalation,
    scenario_ambiguous_reply_refuses_to_fabricate,
    scenario_high_value_requires_approval,
]


def run_all_scenarios() -> list[ScenarioResult]:
    return [scn() for scn in SCENARIOS]


def write_table3(results: list[ScenarioResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "table3_failure_injection.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Rule Under Test", "Expected Behavior", "Observed Behavior", "Pass/Fail"])
        for r in results:
            writer.writerow([r.name, r.rule_under_test, r.expected, r.observed, "PASS" if r.passed else "FAIL"])

    trails_path = RESULTS_DIR / "table3_failure_injection_trails.json"
    trails_path.write_text(json.dumps(
        {r.name: {"passed": r.passed, "rule_under_test": r.rule_under_test, "notes": r.notes, "audit_trail": r.audit_trail}
         for r in results}, indent=2,
    ))
    print(f"wrote {csv_path}")
    print(f"wrote {trails_path}")


def print_table3(results: list[ScenarioResult]) -> None:
    print("\n=== Table 3: failure-injection scenario pass/fail ===\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}  ({r.rule_under_test})")
        print(f"  expected: {r.expected}")
        print(f"  observed: {r.observed}")
        if r.notes:
            print(f"  notes:    {r.notes}")
        print()
    n_pass = sum(r.passed for r in results)
    print(f"{n_pass}/{len(results)} scenarios passed.")


if __name__ == "__main__":
    results = run_all_scenarios()
    print_table3(results)
    write_table3(results)
