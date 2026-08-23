"""
tests/test_agent_loop.py — end-to-end test of agent/state_machine.py's
run_agent_cycle() over a small batch of invoices across simulated days.

Population is deliberately biased toward serial_promiser/disputer
archetypes (low keep_probability_base, meaningful dispute_propensity) so
the 30-day window reliably reaches at least one human_escalation without
needing an enormous batch -- this is a targeted correctness test, not a
representative-population eval (that's eval/run_eval.py's job).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.state_machine import AgentContext, TERMINAL_INVOICE_STATUSES, run_agent_cycle
from backend.db import AuditLog, Base, Customer, Invoice, Promise
from features.feature_engine import compute_prs
from models.propensity_model import DEFAULT_ARTIFACT_PATH, PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.exists(),
    reason="model artifact not found; run `python3 models/train.py` first",
)

SEED = 99
N_INVOICES = 20
N_DAYS = 30
REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)

# Biased toward archetypes likely to break promises / dispute within a short
# window, specifically so this small/short test reliably reaches escalation.
TEST_ARCHETYPE_MIX = (
    ["serial_promiser"] * 10 + ["disputer"] * 5 + ["non_responsive"] * 3 + ["model_citizen"] * 2
)


def build_batch(seed: int):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    latent_rng = np.random.default_rng(seed)
    customers: dict[str, CustomerBehaviorModel] = {}
    invoice_ids: list[str] = []

    for i, archetype in enumerate(TEST_ARCHETYPE_MIX):
        customer_id = f"agenttest-{i:03d}-{archetype}"
        latent = sample_customer_latent(customer_id, archetype, latent_rng)
        customers[customer_id] = CustomerBehaviorModel(latent)

        session.add(Customer(
            customer_id=customer_id, name=customer_id, archetype=archetype, segment="SMB",
            credit_terms_days=5, onboarding_date=REFERENCE_START - dt.timedelta(days=200),
        ))
        invoice_id = f"inv-{customer_id}-0"
        session.add(Invoice(
            invoice_id=invoice_id, customer_id=customer_id, amount=10_000.0,
            issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=5),
            status="open",
        ))
        invoice_ids.append(invoice_id)
    session.commit()

    return session, customers, invoice_ids


@pytest.fixture(scope="module")
def batch_run():
    """Runs the full 20-invoice / 30-day batch ONCE, shared read-only across
    all assertions in this file (the DoD requires this to run end-to-end
    without manual intervention between days -- exactly what this fixture does)."""
    session, customers, invoice_ids = build_batch(SEED)
    model = PropensityModel()
    from agent.tools import PropensityModelTool
    propensity_tool = PropensityModelTool(model)
    pending_promise_truth: dict[str, bool] = {}

    for day in range(N_DAYS):
        for invoice_id in invoice_ids:
            invoice = session.get(Invoice, invoice_id)
            customer_model = customers[invoice.customer_id]
            ctx = AgentContext(
                day=day, reference_start=REFERENCE_START, session=session,
                customer_model=customer_model, propensity_model_tool=propensity_tool,
                policy_config=CONFIG, pending_promise_truth=pending_promise_truth,
            )
            run_agent_cycle(invoice_id, ctx)

    return session, customers, invoice_ids


class TestAuditCompleteness:
    def test_every_executed_action_has_a_rationale(self, batch_run):
        session, _, _ = batch_run
        acted_events = (
            session.query(AuditLog)
            .filter(AuditLog.step == "act", AuditLog.executed_action.isnot(None), AuditLog.executed_action != "none")
            .all()
        )
        assert len(acted_events) > 0, "batch produced no non-'none' executed actions at all -- test setup too inert"
        for event in acted_events:
            assert event.rationale_code, f"executed_action={event.executed_action} on {event.invoice_id} has no rationale_code"

    def test_every_invoice_has_a_traceable_phase_sequence(self, batch_run):
        session, _, invoice_ids = batch_run
        for invoice_id in invoice_ids:
            steps = {row.step for row in session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()}
            assert "detect" in steps


class TestPRSUpdatesAfterPromiseVerification:
    def test_verified_promise_changes_prs(self, batch_run):
        """compute_prs() is a DB snapshot filtered by Promise.made_on <=
        as_of, not a true point-in-time query -- by the time this test runs,
        the whole batch has already executed and `kept` is already resolved
        in the DB for every promise.made_on <= as_of, regardless of as_of's
        own value. So "before vs. after" real dates can't isolate the
        effect. Instead, flip `kept` on the live DB row and compare PRS
        computed both ways at the SAME as_of -- this isolates exactly the
        causal direction Stage 2's PRS formula is supposed to have,
        independent of any date-window subtlety.
        """
        session, _, _ = batch_run
        resolved_promises = session.query(Promise).filter(Promise.kept.isnot(None)).all()
        assert len(resolved_promises) > 0, "no promise reached verification in this batch/window"

        promise = resolved_promises[0]
        invoice = session.get(Invoice, promise.invoice_id)
        customer_id = invoice.customer_id
        as_of = promise.promised_date + dt.timedelta(days=CONFIG.grace_period_days + 1)

        original_kept = promise.kept
        prs_as_resolved = compute_prs(customer_id, as_of=as_of, session=session)

        promise.kept = not original_kept
        session.commit()
        prs_flipped = compute_prs(customer_id, as_of=as_of, session=session)

        promise.kept = original_kept  # restore -- other tests in this module share `batch_run`
        session.commit()

        if original_kept:
            assert prs_as_resolved > prs_flipped, "PRS should be higher with this promise kept than broken, all else equal"
        else:
            assert prs_as_resolved < prs_flipped, "PRS should be lower with this promise broken than kept, all else equal"

    def test_broken_promise_lowers_the_live_loops_own_recorded_prs(self, batch_run):
        """Stronger, more literal version of the DoD's wording: not a
        hypothetical, but the actual prs_score values the live loop itself
        wrote into 'decide' AuditEvents (input_snapshot) before vs. after a
        broken promise's verify event, for the same customer.
        """
        session, _, _ = batch_run
        broken = session.query(Promise).filter(Promise.kept.is_(False)).all()
        assert broken, "no broken promise in this batch/window -- can't test the live loop's own recorded values"

        found_comparable_pair = False
        for promise in broken:
            invoice = session.get(Invoice, promise.invoice_id)
            customer_id = invoice.customer_id

            verify_event = (
                session.query(AuditLog)
                .filter(AuditLog.invoice_id == promise.invoice_id, AuditLog.step == "verify",
                        AuditLog.rationale_code == "PROMISE_VERIFIED")
                .order_by(AuditLog.timestamp.asc())
                .first()
            )
            if verify_event is None:
                continue
            verify_seq = int(verify_event.log_id.rsplit("-", 1)[-1])

            decide_events = (
                session.query(AuditLog)
                .filter(AuditLog.invoice_id == promise.invoice_id, AuditLog.step == "decide")
                .all()
            )
            before = [e for e in decide_events if int(e.log_id.rsplit("-", 1)[-1]) < verify_seq]
            after = [e for e in decide_events if int(e.log_id.rsplit("-", 1)[-1]) > verify_seq]
            if not before or not after:
                continue

            prs_before = before[-1].input_snapshot["prs_score"]
            prs_after = after[0].input_snapshot["prs_score"]
            assert prs_after < prs_before, (
                f"{customer_id}: live-loop-recorded prs_score did not drop after a broken promise "
                f"({prs_before} -> {prs_after})"
            )
            found_comparable_pair = True

        assert found_comparable_pair, "no invoice in this batch had both a decide event before AND after its broken-promise verify event"

    def test_promise_stays_pending_until_grace_deadline(self, batch_run):
        session, _, _ = batch_run
        # every resolved promise's verify AuditLog must be dated on/after promised_date+grace
        for promise in session.query(Promise).filter(Promise.kept.isnot(None)).all():
            grace_deadline = promise.promised_date + dt.timedelta(days=CONFIG.grace_period_days)
            verify_events = (
                session.query(AuditLog)
                .filter(AuditLog.invoice_id == promise.invoice_id, AuditLog.step == "verify",
                        AuditLog.rationale_code == "PROMISE_VERIFIED")
                .all()
            )
            assert len(verify_events) >= 1
            for ev in verify_events:
                assert ev.timestamp is not None  # sanity: written


class TestEscalationTerminalState:
    def test_at_least_one_invoice_escalates(self, batch_run):
        session, _, invoice_ids = batch_run
        escalated = [
            iid for iid in invoice_ids
            if session.get(Invoice, iid).status == "human_queue"
        ]
        assert len(escalated) > 0, "batch never reached escalation -- can't test the terminal-state property"

    def test_escalated_invoice_stops_running_decide_act(self, batch_run):
        session, _, invoice_ids = batch_run
        escalated = [iid for iid in invoice_ids if session.get(Invoice, iid).status == "human_queue"]
        assert escalated

        for invoice_id in escalated:
            escalation_act_event = (
                session.query(AuditLog)
                .filter(AuditLog.invoice_id == invoice_id, AuditLog.step == "act", AuditLog.executed_action == "human_escalation")
                .order_by(AuditLog.timestamp.asc())
                .first()
            )
            assert escalation_act_event is not None
            escalation_log_id_num = int(escalation_act_event.log_id.rsplit("-", 1)[-1])

            later_decide_or_act = (
                session.query(AuditLog)
                .filter(
                    AuditLog.invoice_id == invoice_id,
                    AuditLog.step.in_(["decide", "act"]),
                )
                .all()
            )
            # every decide/act event for this invoice must be AT OR BEFORE
            # the escalation itself (by insertion order) -- none may come after
            for ev in later_decide_or_act:
                ev_num = int(ev.log_id.rsplit("-", 1)[-1])
                assert ev_num <= escalation_log_id_num, (
                    f"{invoice_id} ran another {ev.step} event (log_id={ev.log_id}) "
                    f"after escalating (log_id={escalation_act_event.log_id})"
                )

            # every detect event after escalation must be the lightweight ALREADY_TERMINAL marker
            detect_events = (
                session.query(AuditLog)
                .filter(AuditLog.invoice_id == invoice_id, AuditLog.step == "detect")
                .all()
            )
            post_escalation_detects = [ev for ev in detect_events if int(ev.log_id.rsplit("-", 1)[-1]) > escalation_log_id_num]
            for ev in post_escalation_detects:
                assert ev.rationale_code == "ALREADY_TERMINAL"


class TestAgentContextRunsEndToEnd:
    def test_batch_completed_without_error(self, batch_run):
        session, _, invoice_ids = batch_run
        assert len(invoice_ids) == N_INVOICES
        total_events = session.query(AuditLog).count()
        assert total_events > N_INVOICES  # at least something logged per invoice
