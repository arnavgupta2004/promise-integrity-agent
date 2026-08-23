"""
tests/test_action_selection.py — end-to-end wiring test for §8's
select_action(): Stage 4's policy/eiv.py select_action() (hard constraints
via policy/constraints.py, then EIV ranking) driven by Stage 3's REAL
trained models.propensity_model.PropensityModel, not a stub.

This file does not reimplement any EIV math or constraint-rule logic --
it only constructs realistic (PolicyState, feature-vector, invoice_amount)
triples per scenario and calls the existing select_action(). Each scenario
prints a human-readable trace: which rule(s) fired (or that EIV ranking was
used), the full ordered rationale-code trail, and the resulting action.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from features.feature_engine import FEATURE_COLUMNS
from models.propensity_model import DEFAULT_ARTIFACT_PATH, PropensityModel
from policy.constraints import PolicyConfig, PolicyState
from policy.eiv import select_action

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.exists(),
    reason="model artifact not found; run `python3 models/train.py` first",
)

CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)


def make_features(**overrides) -> dict:
    base = {
        "relative_lateness": 0.6,
        "prs_score": 0.65,
        "prs_trend": 0.0,
        "dispute_rate": 0.0,
        "response_rate": 0.3,
        "partial_payment_rate": 0.0,
        "amount_tier": 2,
        "days_since_last_contact": 5,
        "active_promise_flag": False,
        "days_until_promised_date": -1,
        "broken_promise_streak": 0,
        "segment": "SMB",
        "intervention_type": "none",  # overwritten per-candidate inside select_action/rank_by_eiv
    }
    base.update(overrides)
    assert set(base.keys()) == set(FEATURE_COLUMNS)
    return base


def make_state(**overrides) -> PolicyState:
    base = dict(
        invoice_amount=10_000.0,
        dispute_flag=False,
        dispute_resolved=True,
        no_contact_requested=False,
        active_promise_flag=False,
        days_until_promised_date=-1,
        broken_promise_streak=0,
        contacts_in_last_3_days=0,
        total_automated_contacts_this_invoice=0,
        prs_score=0.65,
    )
    base.update(overrides)
    return PolicyState(**base)


def trace(label: str, state: PolicyState, features: dict, model: PropensityModel) -> tuple:
    action, rationale_codes, human_approval_required = select_action(
        state.invoice_amount, features, model, state, CONFIG
    )
    used_eiv = rationale_codes[-1] == "EIV_MAX"
    print(f"\n--- {label} ---")
    print(f"  invoice_amount={state.invoice_amount}  prs_score={state.prs_score}")
    print(f"  dispute_flag={state.dispute_flag}(resolved={state.dispute_resolved})  "
          f"no_contact_requested={state.no_contact_requested}")
    print(f"  active_promise_flag={state.active_promise_flag} days_until_promised_date={state.days_until_promised_date}  "
          f"broken_promise_streak={state.broken_promise_streak}")
    print(f"  contacts_in_last_3_days={state.contacts_in_last_3_days}  "
          f"total_automated_contacts_this_invoice={state.total_automated_contacts_this_invoice}")
    rules_fired = rationale_codes[:-1] if used_eiv else rationale_codes
    print(f"  rule(s) fired        : {rules_fired if rules_fired else '(none)'}")
    print(f"  full rationale trail : {rationale_codes}")
    print(f"  decision path        : {'EIV ranking used' if used_eiv else 'hard constraint forced the action'}")
    print(f"  chosen action        : {action}")
    print(f"  human_approval_required: {human_approval_required}")
    return action, rationale_codes, human_approval_required


@pytest.fixture(scope="module")
def model() -> PropensityModel:
    return PropensityModel()


class TestFiveConstructedExamples:
    def test_1_clean_case_falls_through_to_eiv(self, model):
        """No rule fires -- EIV ranks over the full 6-action eligible set."""
        state = make_state(invoice_amount=10_000.0, prs_score=0.7)
        features = make_features(relative_lateness=0.8, prs_score=0.7, response_rate=0.4)

        action, rationale_codes, human_approval_required = trace(
            "1. Clean case -> EIV ranking", state, features, model
        )

        assert rationale_codes[-1] == "EIV_MAX"
        assert len(rationale_codes) == 1  # no non-terminal rule narrowed anything first
        assert action in {"none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend", "plan_proposal"}
        assert human_approval_required is False

    def test_2_dispute_case_is_forced(self, model):
        """Rule 1: unresolved dispute -> human_escalation, model never consulted."""
        state = make_state(dispute_flag=True, dispute_resolved=False)
        features = make_features()

        action, rationale_codes, human_approval_required = trace(
            "2. Unresolved dispute -> forced human_escalation", state, features, model
        )

        assert action == "human_escalation"
        assert rationale_codes == ("DISPUTE_UNRESOLVED",)
        assert human_approval_required is False

    def test_3_high_value_requires_approval(self, model):
        """Rule 7: high-value invoice -> plan_proposal removed, approval flag
        set, EIV still runs over the remaining 5 actions."""
        state = make_state(invoice_amount=75_000.0, prs_score=0.8)
        features = make_features(relative_lateness=1.2, prs_score=0.8)

        action, rationale_codes, human_approval_required = trace(
            "3. High-value invoice -> approval required, then EIV", state, features, model
        )

        assert rationale_codes == ("HIGH_VALUE_REQUIRES_APPROVAL", "EIV_MAX")
        assert action != "plan_proposal"
        assert human_approval_required is True

    def test_4_compound_multi_code_rationale_trail(self, model):
        """Rules 5 + 7 both fire (frequency cap AND high value) before
        falling through to EIV -- the trail must show both, in order, not
        just the last one. This is the case Stage 4's fix specifically
        targeted; verifying it survives this integration layer too."""
        state = make_state(invoice_amount=80_000.0, contacts_in_last_3_days=2, prs_score=0.75)
        features = make_features(relative_lateness=1.0, prs_score=0.75, days_since_last_contact=1)

        action, rationale_codes, human_approval_required = trace(
            "4. Compound: frequency cap + high value -> EIV over {none, link_resend}", state, features, model
        )

        assert rationale_codes == ("FREQUENCY_CAP", "HIGH_VALUE_REQUIRES_APPROVAL", "EIV_MAX")
        assert action in {"none", "link_resend"}  # the only two actions frequency cap leaves eligible
        assert human_approval_required is True

    def test_5_cooling_period_forced_none(self, model):
        """Rule 3: an active promise still inside its grace period ->
        forced 'none', a different forced-none rule than no-contact-honored,
        exercised to show rule 3 specifically (not just rule 2)."""
        state = make_state(active_promise_flag=True, days_until_promised_date=2)
        features = make_features(active_promise_flag=True, days_until_promised_date=2)

        action, rationale_codes, human_approval_required = trace(
            "5. Active promise in cooling period -> forced none", state, features, model
        )

        assert action == "none"
        assert rationale_codes == ("COOLING_PERIOD_ACTIVE",)
        assert human_approval_required is False
