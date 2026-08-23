"""
tests/test_llm_tools.py — real Gemini-backed tests for agent/llm/extract.py
and agent/llm/draft.py. Makes actual API calls (GEMINI_API_KEY required);
skipped entirely if the key isn't set.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

pytestmark = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")

from agent.llm.draft import BANNED_PHRASES, LLMDraftTool, _check_guardrail
from agent.llm.extract import CONFIDENCE_FLOOR, LLMExtractTool
from agent.state_machine import AgentContext, process_customer_reply
from backend.db import AuditLog, Base, Customer, Invoice
from policy.constraints import PolicyConfig

REFERENCE_START = dt.datetime(2025, 1, 1)


@pytest.fixture(scope="module")
def extract_tool():
    return LLMExtractTool()


@pytest.fixture(scope="module")
def draft_tool():
    return LLMDraftTool()


# ---------------------------------------------------------------------------
# The 5 required extraction cases
# ---------------------------------------------------------------------------

class TestExtractionCases:
    def test_clear_commitment(self, extract_tool):
        result = extract_tool.extract_promise("I'll pay ₹50,000 by the 15th.")
        print("\n[clear commitment] raw notes:", result.notes, "| confidence:", result.confidence)
        assert result.commitment_detected is True
        assert result.promised_amount == pytest.approx(50000, rel=0.01)
        assert result.promised_date is not None
        assert result.confidence >= CONFIDENCE_FLOOR

    def test_ambiguous_non_commitment(self, extract_tool):
        """Failure case #8: vague reassurance must resolve to
        commitment_detected=False, whichever mechanism gets it there."""
        result = extract_tool.extract_promise("we'll sort it out soon")
        print("\n[ambiguous] raw_commitment_detected:", result.raw_commitment_detected,
              "| confidence:", result.confidence, "| floor_applied:", result.confidence_floor_applied,
              "| notes:", result.notes)
        assert result.commitment_detected is False
        assert result.promised_date is None
        assert result.promised_amount is None

    def test_partial_conditional_commitment(self, extract_tool):
        result = extract_tool.extract_promise(
            "I can pay ₹20,000 now and the rest once my client pays me, hopefully by next Friday."
        )
        print("\n[conditional] commitment_detected:", result.commitment_detected,
              "| amount:", result.promised_amount, "| date:", result.promised_date,
              "| confidence:", result.confidence, "| notes:", result.notes)
        # a concrete partial amount + a condition/date is still a real,
        # if partial, commitment -- should be detected, not dismissed as vague
        assert result.commitment_detected is True

    def test_hostile_refusal(self, extract_tool):
        result = extract_tool.extract_promise(
            "This is ridiculous, I'm not paying anything, stop harassing me."
        )
        print("\n[hostile] commitment_detected:", result.commitment_detected,
              "| confidence:", result.confidence, "| notes:", result.notes)
        assert result.commitment_detected is False
        assert result.promised_amount is None

    def test_amount_mismatch_flagged_not_auto_accepted(self, extract_tool):
        """A promise for a different amount than owed must still be
        captured (a real commitment was made) but flagged, not silently
        trusted as satisfying the invoice."""
        result = extract_tool.extract_promise("I'll pay ₹10,000 by Friday.", owed_amount=50_000.0)
        print("\n[amount mismatch] commitment_detected:", result.commitment_detected,
              "| promised_amount:", result.promised_amount, "| amount_mismatch:", result.amount_mismatch,
              "| notes:", result.notes)
        assert result.commitment_detected is True
        assert result.promised_amount == pytest.approx(10000, rel=0.01)
        assert result.amount_mismatch is True


# ---------------------------------------------------------------------------
# Confidence-floor rule, tested directly (not relying on the LLM happening
# to produce a low-confidence-but-true case)
# ---------------------------------------------------------------------------

class TestConfidenceFloorRule:
    def test_confidence_below_floor_forces_false(self, extract_tool):
        class StubResponse:
            def __init__(self, parsed):
                self.parsed = parsed
                self.text = "{}"

        class StubSchema:
            commitment_detected = True
            promised_date = "2025-02-01"
            promised_amount = 1000.0
            confidence = 0.4  # below CONFIDENCE_FLOOR = 0.6
            notes = "low confidence stub"

        class StubClient:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    return StubResponse(StubSchema())

        tool = LLMExtractTool(client=StubClient())
        result = tool.extract_promise("some ambiguous reply")
        assert result.raw_commitment_detected is True
        assert result.confidence_floor_applied is True
        assert result.commitment_detected is False
        assert result.promised_date is None
        assert result.promised_amount is None


# ---------------------------------------------------------------------------
# Audit-event wiring for process_customer_reply
# ---------------------------------------------------------------------------

class TestProcessCustomerReplyAuditEvents:
    @pytest.fixture
    def ctx_and_invoice(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        session.add(Customer(customer_id="c1", name="c1", archetype="serial_promiser", segment="SMB",
                              credit_terms_days=30, onboarding_date=REFERENCE_START - dt.timedelta(days=100)))
        session.add(Invoice(invoice_id="inv1", customer_id="c1", amount=50_000.0,
                             issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=30),
                             status="open"))
        session.commit()
        ctx = AgentContext(
            day=5, reference_start=REFERENCE_START, session=session,
            customer_model=None, propensity_model_tool=None, policy_config=PolicyConfig(),
        )
        return ctx, session

    def test_ambiguous_reply_writes_no_commitment_detected_event(self, ctx_and_invoice, extract_tool):
        ctx, session = ctx_and_invoice
        as_of = REFERENCE_START + dt.timedelta(days=5)
        extraction = process_customer_reply(
            "inv1", "c1", "we'll sort it out soon", ctx, as_of, owed_amount=50_000.0, extract_tool=extract_tool,
        )
        assert extraction.commitment_detected is False

        events = session.query(AuditLog).filter(AuditLog.invoice_id == "inv1", AuditLog.step == "act").all()
        assert len(events) == 1
        event = events[0]
        assert event.decision == "no_promise_captured"
        assert event.rationale_code in ("NO_COMMITMENT_DETECTED", "EXTRACTION_BELOW_CONFIDENCE_FLOOR")
        assert event.input_snapshot["reply_text"] == "we'll sort it out soon"
        print("\n[audit event, ambiguous case]:")
        print(f"  step={event.step} decision={event.decision} rationale_code={event.rationale_code}")
        print(f"  input_snapshot={event.input_snapshot}")

    def test_clear_commitment_writes_promise_captured_event(self, ctx_and_invoice, extract_tool):
        ctx, session = ctx_and_invoice
        as_of = REFERENCE_START + dt.timedelta(days=5)
        extraction = process_customer_reply(
            "inv1", "c1", "I'll pay ₹50,000 by the 15th.", ctx, as_of, owed_amount=50_000.0, extract_tool=extract_tool,
        )
        assert extraction.commitment_detected is True

        events = session.query(AuditLog).filter(AuditLog.invoice_id == "inv1", AuditLog.step == "act").all()
        assert len(events) == 1
        assert events[0].decision == "promise_captured"
        assert events[0].rationale_code == "PROMISE_CAPTURED"

    def test_amount_mismatch_writes_flagged_event_requiring_approval(self, ctx_and_invoice, extract_tool):
        ctx, session = ctx_and_invoice
        as_of = REFERENCE_START + dt.timedelta(days=5)
        extraction = process_customer_reply(
            "inv1", "c1", "I'll pay ₹10,000 by Friday.", ctx, as_of, owed_amount=50_000.0, extract_tool=extract_tool,
        )
        assert extraction.commitment_detected is True
        assert extraction.amount_mismatch is True

        events = session.query(AuditLog).filter(AuditLog.invoice_id == "inv1", AuditLog.step == "act").all()
        assert len(events) == 1
        assert events[0].rationale_code == "PROMISE_CAPTURED_AMOUNT_MISMATCH"
        assert events[0].human_approval_required is True


# ---------------------------------------------------------------------------
# Drafting: tone + guardrail
# ---------------------------------------------------------------------------

class TestDraftingToneAndGuardrail:
    @pytest.mark.parametrize("action_type", ["soft_reminder", "firm_reminder", "channel_escalation", "plan_proposal"])
    def test_draft_passes_guardrail_and_is_nonempty(self, draft_tool, action_type):
        context = {"invoice_id": "inv1", "customer_id": "Acme Co", "relative_lateness": 1.2, "days_since_last_contact": 6}
        result = draft_tool.draft(action_type, context)
        print(f"\n[{action_type}] used_fallback={result.used_fallback}")
        print(f"  {result.message}")
        assert result.message.strip()
        assert result.passed_guardrail is True
        passed, hits = _check_guardrail(result.message)
        assert passed, f"drafted message for {action_type} contains banned phrase(s): {hits}"

    def test_guardrail_catches_banned_phrases_directly(self):
        bad_text = "If you do not pay immediately we will pursue legal action and report you to the credit bureau."
        passed, hits = _check_guardrail(bad_text)
        assert passed is False
        assert "legal action" in hits
        assert "credit bureau" in hits

    def test_guardrail_rejection_falls_back_to_safe_template(self):
        class BadResponse:
            text = "Pay now or we will pursue legal action and involve a debt collector."

        class BadClient:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    return BadResponse()

        tool = LLMDraftTool(client=BadClient())
        result = tool.draft("firm_reminder", {"invoice_id": "inv1", "customer_id": "c1"})
        assert result.used_fallback is True
        assert result.passed_guardrail is False
        assert "legal action" in result.guardrail_hits
        passed, _ = _check_guardrail(result.message)
        assert passed  # the returned (fallback) message itself must be clean
