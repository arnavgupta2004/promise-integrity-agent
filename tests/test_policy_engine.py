"""
tests/test_policy_engine.py — unit tests for policy/constraints.py (§9's
9 hard rules) against hand-constructed PolicyState instances. No DB,
simulator, or model involved anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.constraints import (
    EIV_ELIGIBLE_ACTIONS,
    ConstraintResult,
    PolicyConfig,
    PolicyState,
    check_hard_constraints,
    evaluate_constraints,
    get_eligible_actions,
)
from policy.eiv import select_action


def make_state(**overrides) -> PolicyState:
    """A state that trips none of the 9 rules -- falls through to rule 9
    (EIV) untouched. Individual tests override exactly the field(s) needed
    to isolate the rule under test.
    """
    defaults = dict(
        invoice_amount=1000.0,
        dispute_flag=False,
        dispute_resolved=True,
        no_contact_requested=False,
        active_promise_flag=False,
        days_until_promised_date=-1,
        broken_promise_streak=0,
        contacts_in_last_3_days=0,
        total_automated_contacts_this_invoice=0,
        prs_score=0.8,
    )
    defaults.update(overrides)
    return PolicyState(**defaults)


# ---------------------------------------------------------------------------
# Rule 9 sanity: nothing fires -> full eligible set, no forced action.
# ---------------------------------------------------------------------------

class TestRule9NoConstraintFires:
    def test_clean_state_falls_through_to_eiv(self):
        result = evaluate_constraints(make_state())
        assert result.forced_action is None
        assert result.rationale_code is None
        assert result.eligible_actions == EIV_ELIGIBLE_ACTIONS
        assert result.human_approval_required is False
        assert result.triggered_codes == ()


# ---------------------------------------------------------------------------
# Rule 1: unresolved dispute -> human_escalation
# ---------------------------------------------------------------------------

class TestRule1DisputeUnresolved:
    def test_fires_on_unresolved_dispute(self):
        state = make_state(dispute_flag=True, dispute_resolved=False)
        result = evaluate_constraints(state)
        assert result.forced_action == "human_escalation"
        assert result.rationale_code == "DISPUTE_UNRESOLVED"
        assert result.eligible_actions == frozenset()

    def test_does_not_fire_when_dispute_resolved(self):
        state = make_state(dispute_flag=True, dispute_resolved=True)
        result = evaluate_constraints(state)
        assert result.forced_action is None

    def test_does_not_fire_when_no_dispute(self):
        state = make_state(dispute_flag=False, dispute_resolved=False)
        result = evaluate_constraints(state)
        assert result.forced_action is None


# ---------------------------------------------------------------------------
# Rule 2: no_contact_requested -> none, permanent
# ---------------------------------------------------------------------------

class TestRule2NoContactHonored:
    def test_fires_and_forces_none(self):
        state = make_state(no_contact_requested=True)
        result = evaluate_constraints(state)
        assert result.forced_action == "none"
        assert result.rationale_code == "NO_CONTACT_HONORED"

    def test_does_not_fire_when_false(self):
        result = evaluate_constraints(make_state(no_contact_requested=False))
        assert result.forced_action is None


# ---------------------------------------------------------------------------
# Rule 3: active promise still inside its cooling-off grace period -> none
# ---------------------------------------------------------------------------

class TestRule3CoolingPeriodActive:
    def test_fires_within_grace_period(self):
        # grace_period_days=3 default -> condition is days_until_promised_date > -3
        state = make_state(active_promise_flag=True, days_until_promised_date=1)
        result = evaluate_constraints(state)
        assert result.forced_action == "none"
        assert result.rationale_code == "COOLING_PERIOD_ACTIVE"

    def test_fires_just_inside_grace_boundary(self):
        state = make_state(active_promise_flag=True, days_until_promised_date=-2)
        result = evaluate_constraints(state)
        assert result.forced_action == "none"
        assert result.rationale_code == "COOLING_PERIOD_ACTIVE"

    def test_does_not_fire_beyond_grace_period(self):
        # -3 is the boundary itself (not > -3); well beyond it is unambiguous
        state = make_state(active_promise_flag=True, days_until_promised_date=-10)
        result = evaluate_constraints(state)
        assert result.forced_action is None

    def test_does_not_fire_without_active_promise(self):
        state = make_state(active_promise_flag=False, days_until_promised_date=1)
        result = evaluate_constraints(state)
        assert result.forced_action is None


# ---------------------------------------------------------------------------
# Rule 4: broken_promise_streak >= 2 -> human_escalation
# ---------------------------------------------------------------------------

class TestRule4PromiseStreakExceeded:
    def test_fires_at_threshold(self):
        result = evaluate_constraints(make_state(broken_promise_streak=2))
        assert result.forced_action == "human_escalation"
        assert result.rationale_code == "PROMISE_STREAK_EXCEEDED"

    def test_fires_above_threshold(self):
        result = evaluate_constraints(make_state(broken_promise_streak=5))
        assert result.forced_action == "human_escalation"

    def test_does_not_fire_below_threshold(self):
        result = evaluate_constraints(make_state(broken_promise_streak=1))
        assert result.forced_action is None


# ---------------------------------------------------------------------------
# Rule 5: contact frequency cap -- narrows eligible set, does not terminate
# ---------------------------------------------------------------------------

class TestRule5FrequencyCap:
    def test_narrows_eligible_set_but_does_not_force(self):
        result = evaluate_constraints(make_state(contacts_in_last_3_days=1))
        assert result.forced_action is None
        assert result.eligible_actions == frozenset({"none", "link_resend"})
        assert "FREQUENCY_CAP" in result.triggered_codes

    def test_link_resend_exempted_from_cap(self):
        result = evaluate_constraints(make_state(contacts_in_last_3_days=3))
        assert "link_resend" in result.eligible_actions

    def test_does_not_apply_with_zero_recent_contacts(self):
        result = evaluate_constraints(make_state(contacts_in_last_3_days=0))
        assert "FREQUENCY_CAP" not in result.triggered_codes
        assert result.eligible_actions == EIV_ELIGIBLE_ACTIONS


# ---------------------------------------------------------------------------
# Rule 6: too many automated attempts -> human_escalation
# ---------------------------------------------------------------------------

class TestRule6MaxAttemptsReached:
    def test_fires_at_threshold(self):
        result = evaluate_constraints(make_state(total_automated_contacts_this_invoice=4))
        assert result.forced_action == "human_escalation"
        assert result.rationale_code == "MAX_ATTEMPTS_REACHED"

    def test_does_not_fire_below_threshold(self):
        result = evaluate_constraints(make_state(total_automated_contacts_this_invoice=3))
        assert result.forced_action is None


# ---------------------------------------------------------------------------
# Rule 7: high-value invoice -> remove plan_proposal, require approval
# ---------------------------------------------------------------------------

class TestRule7HighValueRequiresApproval:
    def test_fires_at_threshold(self):
        config = PolicyConfig(high_value_threshold=50_000.0)
        result = evaluate_constraints(make_state(invoice_amount=50_000.0), config)
        assert result.forced_action is None
        assert "plan_proposal" not in result.eligible_actions
        assert result.human_approval_required is True
        assert "HIGH_VALUE_REQUIRES_APPROVAL" in result.triggered_codes

    def test_other_actions_remain_eligible(self):
        config = PolicyConfig(high_value_threshold=50_000.0)
        result = evaluate_constraints(make_state(invoice_amount=75_000.0), config)
        assert result.eligible_actions == frozenset({"none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend"})

    def test_does_not_fire_below_threshold(self):
        config = PolicyConfig(high_value_threshold=50_000.0)
        result = evaluate_constraints(make_state(invoice_amount=1_000.0), config)
        assert result.human_approval_required is False
        assert "plan_proposal" in result.eligible_actions


# ---------------------------------------------------------------------------
# Rule 8: plan_proposal requires PRS >= floor
# ---------------------------------------------------------------------------

class TestRule8PrsBelowPlanFloor:
    def test_fires_below_floor(self):
        config = PolicyConfig(plan_eligibility_floor=0.5)
        result = evaluate_constraints(make_state(prs_score=0.2), config)
        assert result.forced_action is None
        assert "plan_proposal" not in result.eligible_actions
        assert "PRS_BELOW_PLAN_FLOOR" in result.triggered_codes

    def test_does_not_fire_at_or_above_floor(self):
        config = PolicyConfig(plan_eligibility_floor=0.5)
        result = evaluate_constraints(make_state(prs_score=0.5), config)
        assert "plan_proposal" in result.eligible_actions
        assert "PRS_BELOW_PLAN_FLOOR" not in result.triggered_codes


# ---------------------------------------------------------------------------
# Priority-order interaction tests
# ---------------------------------------------------------------------------

class TestPriorityOrderInteractions:
    def test_rule1_wins_over_rule4_when_both_would_fire(self):
        """Both an unresolved dispute (rule 1) and a broken-promise streak
        of 2 (rule 4) would independently force human_escalation -- the
        forced *action* is identical either way, so only the rationale_code
        proves which rule actually fired first."""
        state = make_state(dispute_flag=True, dispute_resolved=False, broken_promise_streak=5)
        result = evaluate_constraints(state)
        assert result.forced_action == "human_escalation"
        assert result.rationale_code == "DISPUTE_UNRESOLVED"

    def test_rule2_wins_over_rule3_when_both_would_fire(self):
        """Both no_contact_requested (rule 2) and an active cooling-period
        promise (rule 3) would independently force "none" -- again same
        forced action, rationale_code is the only proof of precedence."""
        state = make_state(no_contact_requested=True, active_promise_flag=True, days_until_promised_date=1)
        result = evaluate_constraints(state)
        assert result.forced_action == "none"
        assert result.rationale_code == "NO_CONTACT_HONORED"

    def test_rule5_non_terminal_still_allows_rule6_to_terminate(self):
        """Rule 5 (frequency cap) only narrows the eligible set and falls
        through -- it must not swallow a later terminal rule. A state that
        trips both rule 5 and rule 6 must resolve to rule 6's forced
        human_escalation, not rule 5's non-terminal filtering."""
        state = make_state(contacts_in_last_3_days=2, total_automated_contacts_this_invoice=4)
        result = evaluate_constraints(state)
        assert result.forced_action == "human_escalation"
        assert result.rationale_code == "MAX_ATTEMPTS_REACHED"

    def test_rule7_shadows_rule8_when_plan_proposal_already_removed(self):
        """If rule 7 (high value) already removed plan_proposal, rule 8's
        guard ("plan_proposal" in eligible) is false by the time it runs --
        it must not also fire and report a misleading PRS-based rationale
        for a removal that was actually caused by the value threshold."""
        config = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)
        state = make_state(invoice_amount=100_000.0, prs_score=0.1)
        result = evaluate_constraints(state, config)
        assert result.forced_action is None
        assert "plan_proposal" not in result.eligible_actions
        assert "HIGH_VALUE_REQUIRES_APPROVAL" in result.triggered_codes
        assert "PRS_BELOW_PLAN_FLOOR" not in result.triggered_codes


# ---------------------------------------------------------------------------
# Compound non-terminal trail: multiple rule-5/7/8 firings must not
# overwrite each other in the audit trail that reaches select_action().
# ---------------------------------------------------------------------------

class TestCompoundRationaleTrailNotOverwritten:
    def test_evaluate_constraints_triggered_codes_keeps_every_non_terminal_rule(self):
        """Both rule 5 (frequency cap) and rule 7 (high value) fire on the
        same invoice, neither is terminal -- triggered_codes must contain
        both, in order, not just the last one."""
        config = PolicyConfig(high_value_threshold=50_000.0)
        state = make_state(contacts_in_last_3_days=1, invoice_amount=75_000.0)
        result = evaluate_constraints(state, config)
        assert result.forced_action is None
        assert result.triggered_codes == ("FREQUENCY_CAP", "HIGH_VALUE_REQUIRES_APPROVAL")

    def test_constraint_result_rationale_code_is_only_the_last_one(self):
        """ConstraintResult.rationale_code (singular) is documented as the
        *last* filtering rule's code -- this test pins that down explicitly
        so it's not mistaken for the full trail (that's triggered_codes)."""
        config = PolicyConfig(high_value_threshold=50_000.0)
        state = make_state(contacts_in_last_3_days=1, invoice_amount=75_000.0)
        result = evaluate_constraints(state, config)
        assert result.rationale_code == "HIGH_VALUE_REQUIRES_APPROVAL"
        assert result.rationale_code != "FREQUENCY_CAP"  # would be silently lost if this were the only code kept

    def test_select_action_returns_full_compound_trail_not_a_single_code(self):
        """The bug this test guards against: select_action() used to return
        a single collapsed rationale string (result.rationale_code or
        "EIV_MAX"), which meant FREQUENCY_CAP would vanish whenever a later
        non-terminal rule also fired before EIV ran. It now returns the
        full ordered tuple, ending in "EIV_MAX" once EIV actually decides."""
        config = PolicyConfig(high_value_threshold=50_000.0)
        state = make_state(contacts_in_last_3_days=1, invoice_amount=75_000.0, prs_score=0.8)

        class StubModel:
            def predict_proba(self, fv):
                return {"none": 0.3, "link_resend": 0.5}[fv["intervention_type"]]

        action, rationale_codes, human_approval_required = select_action(
            75_000.0, {"placeholder": True}, StubModel(), state, config
        )
        assert action == "link_resend"
        assert rationale_codes == ("FREQUENCY_CAP", "HIGH_VALUE_REQUIRES_APPROVAL", "EIV_MAX")
        assert human_approval_required is True

    def test_select_action_forced_rule_returns_single_element_trail(self):
        state = make_state(dispute_flag=True, dispute_resolved=False)

        class StubModel:
            def predict_proba(self, fv):
                raise AssertionError("model must not be scored when a hard constraint forces the action")

        action, rationale_codes, _ = select_action(1000.0, {}, StubModel(), state)
        assert action == "human_escalation"
        assert rationale_codes == ("DISPUTE_UNRESOLVED",)


# ---------------------------------------------------------------------------
# check_hard_constraints() / get_eligible_actions() called independently
# (not via evaluate_constraints() directly) must stay consistent with it,
# since §8's pseudocode and §10's tool interface reference them separately.
# ---------------------------------------------------------------------------

class TestIndependentWrapperConsistency:
    def test_independent_calls_agree_with_evaluate_constraints_when_clean(self):
        state = make_state()
        forced = check_hard_constraints(state)          # fresh evaluate_constraints() call #1
        eligible = get_eligible_actions(state)           # fresh evaluate_constraints() call #2, independent of the first
        combined = evaluate_constraints(state)

        assert forced is None
        assert forced == combined.forced_action
        assert eligible == combined.eligible_actions == EIV_ELIGIBLE_ACTIONS

    def test_independent_calls_agree_when_a_terminal_rule_fires(self):
        state = make_state(broken_promise_streak=2)
        forced = check_hard_constraints(state)
        eligible = get_eligible_actions(state)

        assert forced == "human_escalation"
        # get_eligible_actions(), called independently, must not report a
        # stale/non-empty eligible set for a state a terminal rule already resolved
        assert eligible == frozenset()

    def test_independent_calls_agree_with_non_terminal_filtering(self):
        config = PolicyConfig(high_value_threshold=50_000.0)
        state = make_state(invoice_amount=60_000.0)
        forced = check_hard_constraints(state, config)
        eligible = get_eligible_actions(state, config)

        assert forced is None
        assert "plan_proposal" not in eligible
        assert eligible == frozenset({"none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend"})

    def test_independent_calls_use_the_same_config_consistently(self):
        """A caller must pass the same config to both wrapper calls to get
        consistent results -- verifies the config parameter actually
        threads through both independently, not just one of them."""
        loose_config = PolicyConfig(plan_eligibility_floor=0.0)   # plan_proposal always eligible on PRS grounds
        strict_config = PolicyConfig(plan_eligibility_floor=0.9)  # almost never eligible

        state = make_state(prs_score=0.5)
        eligible_loose = get_eligible_actions(state, loose_config)
        eligible_strict = get_eligible_actions(state, strict_config)

        assert "plan_proposal" in eligible_loose
        assert "plan_proposal" not in eligible_strict
