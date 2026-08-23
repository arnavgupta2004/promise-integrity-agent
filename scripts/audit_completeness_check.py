"""
scripts/audit_completeness_check.py — Stage 9: verification, not new
implementation. Runs a larger batch (~100 invoices, 60 simulated days)
through the full Stage 7 agent loop with Stage 8's REAL LLM tools wired in
(agent.llm.draft.LLMDraftTool for outreach, agent.llm.extract.LLMExtractTool
for at least one explicit customer-reply/amount-mismatch example) -- not
the Stage 7 placeholder stub -- then walks every invoice's full audit-log
trail and checks four STRUCTURAL properties, generically, with no hardcoded
list of "expected" rationale codes:

  (a) every phase transition has a logged event
  (b) every executed action has a non-null rationale_code
  (c) no action was executed without a corresponding prior decide-phase
      authorization event (matching decision == executed_action, same day)
  (d) any action flagged human_approval_required=True was not auto-dispatched

Property checks operate purely on AuditLog's own structural fields
(step, executed_action, rationale_code, human_approval_required, and the
log_id sequence number already established in Stage 7/8 as the reliable
write-order key) -- never on an allowlist of specific rationale_code
strings. This is what lets the same sweep correctly validate Stage 7's
promise-capture events (PROMISE_CAPTURED, SPONTANEOUS_PROMISE_CAPTURED) and
Stage 8's extraction-outcome events (NO_COMMITMENT_DETECTED,
EXTRACTION_BELOW_CONFIDENCE_FLOOR) without any special-cased logic for
either -- and would keep working for whatever rationale codes get added in
future stages, unmodified.

Before trusting a clean run, self_test() deliberately corrupts a copy of a
small batch's audit trail (removes one event, nulls one rationale_code) and
confirms the checker actually catches both -- a checker that can't fail is
not a checker.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

from agent.llm.draft import LLMDraftTool as RealLLMDraftTool
from agent.llm.extract import LLMExtractTool
from agent.state_machine import AgentContext, process_customer_reply, run_agent_cycle
from agent.tools import PropensityModelTool
from backend.db import AuditLog, Base, Customer, Invoice
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import ARCHETYPE_NAMES, sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)
INVOICE_AMOUNTS = [8_000.0, 15_000.0, 30_000.0, 60_000.0, 90_000.0]  # deliberately spans the high-value threshold


def _seq(log_id: str) -> int:
    return int(log_id.rsplit("-", 1)[-1])


# ---------------------------------------------------------------------------
# Batch construction / execution
# ---------------------------------------------------------------------------

def build_batch(seed: int, n_invoices: int, credit_terms_days: int = 10, db_path: str | None = None,
                 invoice_amounts: list[float] | None = None):
    """db_path=None -> in-memory (fine for the cheap self-test mini-batch,
    which never survives past this process anyway). The real, real-LLM
    batch passes a file path instead: if a quota wall is hit partway
    through ~215 real API calls, an in-memory DB would take every already-
    completed invoice down with it. A file survives the crash and can be
    re-opened to see exactly how far the run actually got.
    """
    engine = create_engine(f"sqlite:///{db_path}" if db_path else "sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    rng = np.random.default_rng(seed)
    customers: dict[str, CustomerBehaviorModel] = {}
    invoice_ids: list[str] = []
    amounts_pool = invoice_amounts if invoice_amounts is not None else INVOICE_AMOUNTS

    for i in range(n_invoices):
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
        customer_id = f"batch-{i:03d}-{archetype}"
        latent = sample_customer_latent(customer_id, archetype, rng)
        customers[customer_id] = CustomerBehaviorModel(latent)

        session.add(Customer(
            customer_id=customer_id, name=customer_id, archetype=archetype, segment="SMB",
            credit_terms_days=credit_terms_days, onboarding_date=REFERENCE_START - dt.timedelta(days=200),
        ))
        amount = float(rng.choice(amounts_pool))
        invoice_id = f"inv-{customer_id}-0"
        session.add(Invoice(
            invoice_id=invoice_id, customer_id=customer_id, amount=amount,
            issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=credit_terms_days),
            status="open",
        ))
        invoice_ids.append(invoice_id)
    session.commit()

    return session, customers, invoice_ids


class ResilientDraftTool:
    """Wraps the real LLMDraftTool; if a call ever fails after Stage 8's own
    retry/backoff is exhausted (quota wall, not a transient blip), permanently
    switches to the free Stage 7 stub for the rest of the run instead of
    crashing the whole batch and losing every invoice already processed.
    Tracks exactly how many drafts were real vs. fallback so the report is
    honest about which is which, not silently degraded.
    """
    def __init__(self, real_tool):
        from agent.tools import LLMDraftTool as StubLLMDraftTool
        self.real_tool = real_tool
        self.stub_tool = StubLLMDraftTool()
        self.degraded = False
        self.degraded_at: tuple[int, str] | None = None
        self.n_real = 0
        self.n_fallback = 0

    def draft_message(self, action_type: str, context: dict) -> str:
        if self.degraded:
            self.n_fallback += 1
            return self.stub_tool.draft_message(action_type, context)
        try:
            result = self.real_tool.draft_message(action_type, context)
            self.n_real += 1
            return result
        except Exception as exc:
            print(f"  [degrading to stub draft tool: {type(exc).__name__}: {str(exc)[:200]}]")
            self.degraded = True
            self.n_fallback += 1
            return self.stub_tool.draft_message(action_type, context)


def run_batch(session, customers, invoice_ids, n_days: int, draft_tool, extract_tool,
              inject_amount_mismatch_for: str | None = None) -> ResilientDraftTool:
    model = PropensityModel()
    propensity_tool = PropensityModelTool(model)
    pending_promise_truth: dict[str, bool] = {}
    resilient_draft = ResilientDraftTool(draft_tool) if not isinstance(draft_tool, ResilientDraftTool) else draft_tool

    def make_ctx(day: int, customer_id: str) -> AgentContext:
        return AgentContext(
            day=day, reference_start=REFERENCE_START, session=session,
            customer_model=customers[customer_id], propensity_model_tool=propensity_tool,
            policy_config=CONFIG, pending_promise_truth=pending_promise_truth, draft_tool=resilient_draft,
        )

    # Explicit, guaranteed amount-mismatch example (Stage 8's extraction
    # path isn't reachable from the simulator alone -- there's no free-text
    # customer reply source in the simulated loop, see agent/state_machine.py's
    # module docstring). Injected on day 0, before the main sweep, so the
    # invoice is unambiguously still open.
    if inject_amount_mismatch_for:
        invoice = session.get(Invoice, inject_amount_mismatch_for)
        stated = round(invoice.amount * 0.2, -2)  # deliberately well under the real amount
        ctx0 = make_ctx(0, invoice.customer_id)
        try:
            process_customer_reply(
                inject_amount_mismatch_for, invoice.customer_id,
                f"I'll pay ₹{stated:,.0f} by Friday.", ctx0, REFERENCE_START,
                owed_amount=invoice.amount, extract_tool=extract_tool,
            )
        except Exception as exc:
            print(f"  [amount-mismatch example call failed: {type(exc).__name__}: {str(exc)[:200]}]")

    for day in range(n_days):
        for invoice_id in invoice_ids:
            invoice = session.get(Invoice, invoice_id)
            ctx = make_ctx(day, invoice.customer_id)
            run_agent_cycle(invoice_id, ctx)
        session.commit()  # persist progress after every simulated day, not just at the end
        if day % 10 == 0:
            print(f"  ...day {day}/{n_days} done "
                  f"(real drafts so far: {resilient_draft.n_real}, fallback: {resilient_draft.n_fallback})")

    return resilient_draft


# ---------------------------------------------------------------------------
# Generic structural checks (no hardcoded rationale-code list)
# ---------------------------------------------------------------------------

REQUIRED_FULL_CYCLE_STEPS = {"detect", "diagnose", "decide", "act", "reassess"}  # §12's own frozen phase names


@dataclass
class InvoiceCheckResult:
    invoice_id: str
    passed: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class CompletenessReport:
    results: list[InvoiceCheckResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pct_complete(self) -> float:
        return 100.0 * self.n_passed / self.total if self.total else float("nan")

    @property
    def failures(self) -> list[InvoiceCheckResult]:
        return [r for r in self.results if not r.passed]


def check_invoice(session, invoice_id: str, reference_start: dt.datetime) -> InvoiceCheckResult:
    events = session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()
    events.sort(key=lambda e: _seq(e.log_id))
    problems: list[str] = []

    events_by_day: dict[int, list] = defaultdict(list)
    for e in events:
        day = (e.timestamp - reference_start).days
        events_by_day[day].append(e)

    # (a) every phase transition has a logged event
    for day, day_events in sorted(events_by_day.items()):
        steps = {e.step for e in day_events}
        if len(day_events) == 1:
            if steps != {"detect"}:
                problems.append(f"day {day}: single-event day but step(s)={steps}, expected just a lightweight 'detect'")
        else:
            missing = REQUIRED_FULL_CYCLE_STEPS - steps
            if missing:
                problems.append(f"day {day}: missing phase(s) {sorted(missing)} (present: {sorted(steps)})")

    # (b) every executed action has a non-null rationale_code
    for e in events:
        if e.executed_action is not None and not e.rationale_code:
            problems.append(f"log_id={e.log_id} (day {(e.timestamp - reference_start).days}): "
                             f"executed_action={e.executed_action!r} has null/empty rationale_code")

    # (c) no action executed without a corresponding prior decide-phase
    # authorization event (same day, matching decision == executed_action).
    # Events with executed_action=None (promise-capture, extraction-outcome
    # events) are excluded from this requirement by construction -- they
    # were never authorized by decide/EIV in the first place, they're
    # outcome/side-effect records, so nothing here special-cases them.
    for day, day_events in events_by_day.items():
        decide_events = [e for e in day_events if e.step == "decide"]
        for e in day_events:
            if e.step != "act" or e.executed_action is None:
                continue
            matching = [d for d in decide_events if _seq(d.log_id) < _seq(e.log_id) and d.decision == e.executed_action]
            if not matching:
                problems.append(f"log_id={e.log_id} (day {day}): executed_action={e.executed_action!r} has no "
                                 f"prior same-day decide event authorizing that exact action")

    # (d) any action flagged human_approval_required=True was not auto-dispatched
    for e in events:
        if e.human_approval_required and e.executed_action is not None:
            problems.append(f"log_id={e.log_id} (day {(e.timestamp - reference_start).days}): "
                             f"human_approval_required=True but executed_action={e.executed_action!r} was dispatched anyway")

    return InvoiceCheckResult(invoice_id=invoice_id, passed=(len(problems) == 0), problems=problems)


def check_audit_completeness(session, invoice_ids: list[str], reference_start: dt.datetime = REFERENCE_START) -> CompletenessReport:
    return CompletenessReport(results=[check_invoice(session, iid, reference_start) for iid in invoice_ids])


# ---------------------------------------------------------------------------
# Self-test: prove the checker can actually fail before trusting a clean run
# ---------------------------------------------------------------------------

def self_test() -> bool:
    """Small (cheap, stub-driven -- this validates the CHECKER's logic, not
    Stage 8's LLM behavior, which the main batch already exercises for
    real) batch: confirm a clean pass first, then deliberately corrupt a
    copy's audit trail two different ways and confirm the checker catches
    each. Returns True iff every expected outcome (clean-pass, then both
    corruption-catches) held.

    Invoice amounts are deliberately kept below CONFIG.high_value_threshold
    here. This isn't hiding anything -- it's isolating what this specific
    self-test is verifying (that the checker correctly detects deliberately
    INJECTED corruption of properties b/c) from a separate, real,
    pre-existing gap this project's first real run of this checker actually
    found: rule 7 (HIGH_VALUE_REQUIRES_APPROVAL) sets human_approval_required
    but run_agent_cycle never gates dispatch on it, so property (d)
    legitimately fails on ANY uncorrupted batch containing a high-value
    invoice. That's a genuine implementation bug to report, not a self-test
    concern -- it's exactly what the real batch's amount-mismatch example
    and the aggregate completeness report below are for, and it is reported
    honestly there rather than getting laundered through a "clean baseline"
    assumption that would have been false.
    """
    from agent.tools import LLMDraftTool as StubLLMDraftTool

    print("=== Self-test: does the checker actually catch known-bad data? ===\n")
    below_threshold_amounts = [8_000.0, 15_000.0, 25_000.0, 40_000.0]
    session, customers, invoice_ids = build_batch(
        seed=777, n_invoices=3, credit_terms_days=5, invoice_amounts=below_threshold_amounts,
    )
    run_batch(session, customers, invoice_ids, n_days=10, draft_tool=StubLLMDraftTool(), extract_tool=None)

    baseline = check_audit_completeness(session, invoice_ids)
    print(f"Baseline (uncorrupted) mini-batch: {baseline.n_passed}/{baseline.total} invoices pass.")
    if baseline.n_passed != baseline.total:
        print("  UNEXPECTED: baseline mini-batch is not clean -- self-test can't proceed meaningfully.")
        for r in baseline.failures:
            print(f"    {r.invoice_id}: {r.problems}")
        return False

    all_ok = True

    # Corruption 1: null out a rationale_code on an event with an executed_action
    victim = (
        session.query(AuditLog)
        .filter(AuditLog.invoice_id == invoice_ids[0], AuditLog.executed_action.isnot(None))
        .first()
    )
    original_rationale = victim.rationale_code
    victim.rationale_code = None
    session.commit()

    corrupted_report = check_audit_completeness(session, [invoice_ids[0]])
    caught = not corrupted_report.results[0].passed and any(
        "null/empty rationale_code" in p for p in corrupted_report.results[0].problems
    )
    print(f"\nCorruption 1 (nulled rationale_code on {victim.log_id}): "
          f"{'CAUGHT correctly' if caught else 'NOT CAUGHT -- checker is too lenient!'}")
    if caught:
        print(f"  -> {[p for p in corrupted_report.results[0].problems if 'rationale_code' in p][0]}")
    all_ok = all_ok and caught

    victim.rationale_code = original_rationale  # restore
    session.commit()

    # Corruption 2: delete a 'decide' event out from under its matching 'act' event
    act_event = (
        session.query(AuditLog)
        .filter(AuditLog.invoice_id == invoice_ids[1], AuditLog.step == "act", AuditLog.executed_action.isnot(None))
        .first()
    )
    matching_decide = (
        session.query(AuditLog)
        .filter(AuditLog.invoice_id == invoice_ids[1], AuditLog.step == "decide", AuditLog.decision == act_event.executed_action)
        .first()
    )
    deleted_log_id = matching_decide.log_id
    session.delete(matching_decide)
    session.commit()

    corrupted_report2 = check_audit_completeness(session, [invoice_ids[1]])
    caught2 = not corrupted_report2.results[0].passed and any(
        "no prior same-day decide event" in p for p in corrupted_report2.results[0].problems
    )
    print(f"\nCorruption 2 (deleted decide event {deleted_log_id}): "
          f"{'CAUGHT correctly' if caught2 else 'NOT CAUGHT -- checker is too lenient!'}")
    if caught2:
        print(f"  -> {[p for p in corrupted_report2.results[0].problems if 'no prior same-day decide event' in p][0]}")
    all_ok = all_ok and caught2

    print(f"\nSelf-test {'PASSED' if all_ok else 'FAILED'}: checker "
          f"{'correctly rejects' if all_ok else 'FAILED TO REJECT'} known-corrupted data.\n")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_report(report: CompletenessReport, label: str) -> None:
    print(f"=== {label} ===")
    print(f"Total invoices: {report.total}")
    print(f"Complete trails: {report.n_passed}/{report.total} ({report.pct_complete:.1f}%)")
    if report.failures:
        print(f"\nFailing invoices ({len(report.failures)}):")
        for r in report.failures:
            print(f"  {r.invoice_id}: {len(r.problems)} problem(s)")
            for p in r.problems[:5]:
                print(f"    - {p}")
            if len(r.problems) > 5:
                print(f"    ... and {len(r.problems) - 5} more")
    print()


def main() -> None:
    ok = self_test()
    if not ok:
        print("ABORTING: self-test failed, the checker cannot be trusted to validate the real batch.")
        return

    N_INVOICES = 100
    N_DAYS = 60
    DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "audit_completeness_batch.db")
    print(f"=== Running real batch: {N_INVOICES} invoices, {N_DAYS} days, Stage 8 real LLM tools wired in ===")
    print(f"    (persistent DB at {DB_PATH} -- survives a quota-wall crash, not in-memory)\n")
    session, customers, invoice_ids = build_batch(seed=123, n_invoices=N_INVOICES, db_path=DB_PATH)
    mismatch_invoice_id = invoice_ids[0]

    resilient_draft = run_batch(
        session, customers, invoice_ids, n_days=N_DAYS,
        draft_tool=RealLLMDraftTool(), extract_tool=LLMExtractTool(),
        inject_amount_mismatch_for=mismatch_invoice_id,
    )
    print(f"\nDrafting summary: {resilient_draft.n_real} real LLM calls, {resilient_draft.n_fallback} stub fallbacks"
          f"{' (degraded partway through due to a quota/API wall)' if resilient_draft.degraded else ''}.\n")

    report = check_audit_completeness(session, invoice_ids)
    print_report(report, "Real batch: audit completeness report")

    print(f"=== Amount-mismatch invoice trail: {mismatch_invoice_id} ===")
    events = session.query(AuditLog).filter(AuditLog.invoice_id == mismatch_invoice_id).all()
    events.sort(key=lambda e: _seq(e.log_id))
    for e in events[:10]:
        day = (e.timestamp - REFERENCE_START).days
        print(f"  day {day:3d} [{e.step:8s}] decision={e.decision} executed={e.executed_action} "
              f"rationale={e.rationale_code} approval_required={e.human_approval_required}")


if __name__ == "__main__":
    main()
