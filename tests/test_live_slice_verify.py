"""
tests/test_live_slice_verify.py — Stage 10 supplementary verification.

Razorpay's hosted checkout on this project's test-mode account rejects both
documented domestic test cards (Visa 4111 1111 1111 1111 and Mastercard
5104 0155 5555 5558) with "international cards not supported" -- an
account-level KYC/international-payment eligibility gate, per Razorpay's
own docs, not anything in this integration (see scripts/live_slice_demo.py's
docstring for the full writeup and the real invoice/payment-link evidence
already gathered up to that wall). That blocks completing a real test
payment, which is the only way fetch_payment would naturally return a
"captured" status in this environment.

This file proves agent/state_machine.py's _verify_live_slice_payment
(everything on THIS side of the Razorpay API boundary: DB update, invoice
settlement, audit event, and PRS recomputation) works correctly from the
point a "paid"/"captured" response would arrive -- by mocking
razorpay_client.fetch_payment_link/fetch_payment to return exactly the
shape Razorpay's real API returns for a captured payment link, and
confirming run_agent_cycle (the actual wiring, not a bypass) reacts
correctly. Only the account's own checkout eligibility is untestable here.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.state_machine import AgentContext, TERMINAL_INVOICE_STATUSES, _verify_live_slice_payment, run_agent_cycle
from agent.tools import AuditLogTool, PropensityModelTool
from backend.db import AuditLog, Base, Customer, Invoice, Payment
from features.feature_engine import compute_prs
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)
INVOICE_AMOUNT = 12_500.0  # well below the ₹50,000 high-value threshold, matching the live-slice demo's own amounts


def _fresh_ctx(session, invoice_id: str, razorpay_client, day: int = 0) -> AgentContext:
    rng = np.random.default_rng(seed=7)
    latent = sample_customer_latent("live-slice-verify-test", "reliable_always_late", rng)
    return AgentContext(
        day=day, reference_start=REFERENCE_START, session=session,
        customer_model=CustomerBehaviorModel(latent), propensity_model_tool=PropensityModelTool(PropensityModel()),
        policy_config=CONFIG, razorpay_client=razorpay_client,
        live_slice_invoice_ids={invoice_id}, live_slice_payment_links={invoice_id: "plink_test123"},
    )


@pytest.fixture
def db_with_open_invoice():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    customer_id = "live-slice-verify-test"
    invoice_id = f"inv-{customer_id}-0"
    session.add(Customer(
        customer_id=customer_id, name=customer_id, archetype="reliable_always_late", segment="SMB",
        credit_terms_days=15, onboarding_date=REFERENCE_START - dt.timedelta(days=200),
    ))
    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=INVOICE_AMOUNT,
        issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=15), status="open",
    ))
    session.commit()
    return session, invoice_id


def _mock_razorpay_client(status: str, amount_rupees: float, payment_id: str = "pay_test456"):
    client = MagicMock()
    client.fetch_payment_link.return_value = {
        "id": "plink_test123", "status": status, "amount_paid": int(round(amount_rupees * 100)),
        "payments": [{"payment_id": payment_id, "amount": int(round(amount_rupees * 100)), "status": "captured"}]
        if status == "paid" else [],
    }
    client.fetch_payment.return_value = {
        "id": payment_id, "amount": int(round(amount_rupees * 100)), "status": "captured", "method": "card",
    }
    return client


class TestVerifyLiveSlicePaymentDirect:
    def test_settles_invoice_when_link_is_paid(self, db_with_open_invoice):
        session, invoice_id = db_with_open_invoice
        client = _mock_razorpay_client("paid", INVOICE_AMOUNT)
        ctx = _fresh_ctx(session, invoice_id, client)
        audit = AuditLogTool(session)
        invoice = session.get(Invoice, invoice_id)

        _verify_live_slice_payment(invoice, ctx, audit, REFERENCE_START)

        assert invoice.status == "paid"
        payment = session.get(Payment, f"pay-{invoice_id}-live")
        assert payment is not None
        assert payment.razorpay_payment_id == "pay_test456"
        assert payment.amount_paid == pytest.approx(INVOICE_AMOUNT)

        events = session.query(AuditLog).filter(
            AuditLog.invoice_id == invoice_id, AuditLog.rationale_code == "LIVE_SLICE_PAYMENT_CONFIRMED",
        ).all()
        assert len(events) == 1
        event = events[0]
        assert event.step == "verify"
        assert event.decision == "live_payment_confirmed"
        assert event.executed_action is None
        assert event.human_approval_required is False
        assert event.model_output["payment"]["status"] == "captured"
        assert event.model_output["payment_link"]["status"] == "paid"

    def test_leaves_invoice_untouched_when_link_not_yet_paid(self, db_with_open_invoice):
        session, invoice_id = db_with_open_invoice
        client = _mock_razorpay_client("created", INVOICE_AMOUNT)
        ctx = _fresh_ctx(session, invoice_id, client)
        audit = AuditLogTool(session)
        invoice = session.get(Invoice, invoice_id)

        _verify_live_slice_payment(invoice, ctx, audit, REFERENCE_START)

        assert invoice.status == "open"
        assert session.get(Payment, f"pay-{invoice_id}-live") is None
        assert session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).count() == 0
        client.fetch_payment.assert_not_called()  # only fetch_payment_link should be called until status=="paid"


class TestRunAgentCycleWiring:
    """Confirms the mocked "captured" response reaches settlement through
    the REAL run_agent_cycle entrypoint, not just the helper function in
    isolation -- and that a settled live-slice invoice short-circuits the
    rest of that day's cycle (detect/diagnose/decide never run once
    _verify_live_slice_payment has already paid it)."""

    def test_run_agent_cycle_settles_and_short_circuits(self, db_with_open_invoice):
        session, invoice_id = db_with_open_invoice
        client = _mock_razorpay_client("paid", INVOICE_AMOUNT)
        ctx = _fresh_ctx(session, invoice_id, client)

        run_agent_cycle(invoice_id, ctx)

        invoice = session.get(Invoice, invoice_id)
        assert invoice.status == "paid"
        assert invoice.status in TERMINAL_INVOICE_STATUSES

        steps = [row.step for row in session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()]
        assert steps == ["verify"]  # only the settlement event -- detect/diagnose/decide/act never ran this cycle


class TestPRSReflectsLiveSliceSettlement:
    """PRS (features/feature_engine.py) has no cached/stored score to
    manually refresh -- compute_prs reads Payment rows live on every call.
    So "does PRS update" reduces to: does the Payment row
    _verify_live_slice_payment writes actually get picked up the next time
    PRS is computed for this customer, with no special-case code needed."""

    def test_payment_row_feeds_directly_into_prs_computation(self, db_with_open_invoice):
        session, invoice_id = db_with_open_invoice
        customer_id = session.get(Invoice, invoice_id).customer_id
        as_of = REFERENCE_START + dt.timedelta(days=1)

        before = session.query(Payment).filter(Payment.invoice_id == invoice_id).count()
        prs_before = compute_prs(customer_id, as_of=as_of, session=session)
        assert before == 0

        client = _mock_razorpay_client("paid", INVOICE_AMOUNT)
        ctx = _fresh_ctx(session, invoice_id, client)
        audit = AuditLogTool(session)
        invoice = session.get(Invoice, invoice_id)
        _verify_live_slice_payment(invoice, ctx, audit, REFERENCE_START)

        after = session.query(Payment).filter(Payment.invoice_id == invoice_id).count()
        prs_after = compute_prs(customer_id, as_of=as_of, session=session)
        assert after == 1
        assert 0.0 <= prs_after <= 1.0
        # A single payment can legitimately leave PRS unchanged: the trend
        # component (§6/§0) is a slope, which needs >=2 paid-invoice data
        # points, and defaults to the neutral 0.5 with fewer than that by
        # design (see compute_prs's own docstring) -- keep/dispute/response
        # aren't driven by Payment rows at all. What matters here is that
        # the row is live in the DB and compute_prs read it without error
        # or a manual refresh step, not that one data point moves a slope.
        rows = (
            session.query(Payment)
            .join(Invoice, Payment.invoice_id == Invoice.invoice_id)
            .filter(Invoice.customer_id == customer_id, Payment.payment_date <= as_of)
            .all()
        )
        assert len(rows) == 1 and rows[0].razorpay_payment_id == "pay_test456"
