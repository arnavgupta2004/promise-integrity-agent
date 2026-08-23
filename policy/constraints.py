"""
policy/constraints.py — §9 hard safety rules, in priority order.

Fully deterministic, zero ML/LLM. Zero imports from simulator, models, or
agent -- this module operates entirely on a plain PolicyState snapshot
(a dataclass, not a DB row or simulator object), so it can be built and
tested in complete isolation before the simulator or the trained
propensity model exist.

Rules are checked in the exact order given in §9; the first one that fires
either short-circuits with a forced terminal action (rules 1, 2, 3, 4, 6),
or narrows the eligible-action set without terminating (rules 5, 7, 8) and
falls through to the next rule. If nothing fires, rule 9 hands the
(possibly narrowed) eligible set to EIV ranking (policy/eiv.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Action = str  # kept as a plain str here (not simulator.behavior_model.Action)
              # specifically to avoid importing simulator -- see module docstring

# The 6 actions EIV ranks over (§7: human_escalation is excluded -- "not a
# 'pay probability' decision, always policy-forced"). Re-declared here
# rather than imported from models/simulator, for the same reason.
EIV_ELIGIBLE_ACTIONS: frozenset[Action] = frozenset({
    "none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend", "plan_proposal",
})

# "Contact actions" for rule 5's frequency cap -- link_resend is explicitly
# exempted per §9 ("low-friction").
CONTACT_ACTIONS: frozenset[Action] = frozenset({
    "soft_reminder", "firm_reminder", "channel_escalation", "plan_proposal",
})
FREQUENCY_CAPPED_ACTIONS = CONTACT_ACTIONS  # link_resend deliberately excluded


@dataclass
class PolicyState:
    """A plain snapshot of everything the 9 rules need. Hand-constructable
    in tests without any DB/simulator/model involved; in real use this is
    populated from a features/feature_engine.py feature vector plus a
    couple of raw invoice/customer fields it doesn't carry.
    """
    invoice_amount: float
    dispute_flag: bool = False
    dispute_resolved: bool = True          # irrelevant when dispute_flag is False
    no_contact_requested: bool = False
    active_promise_flag: bool = False
    days_until_promised_date: int = -1     # -1 if no active promise (matches feature_engine's convention)
    broken_promise_streak: int = 0
    contacts_in_last_3_days: int = 0
    total_automated_contacts_this_invoice: int = 0
    prs_score: float = 0.5


@dataclass
class PolicyConfig:
    """Tunable thresholds referenced by §9. Values are the contract's own
    "e.g." examples where given (grace_period, PLAN_ELIGIBILITY_FLOOR);
    HIGH_VALUE_THRESHOLD isn't given a concrete number in §9, so ₹50,000 is
    used as a reasonable illustrative default, overridable per call.
    """
    grace_period_days: int = 3
    high_value_threshold: float = 50_000.0
    plan_eligibility_floor: float = 0.5


@dataclass
class ConstraintResult:
    forced_action: Optional[Action]           # None means "no hard constraint forced a terminal action -- proceed to EIV"
    rationale_code: Optional[str]              # the code for forced_action, or the last filtering rule's code if forced_action is None
    eligible_actions: frozenset[Action]        # the (possibly narrowed) set EIV should rank over; empty when forced_action is set
    human_approval_required: bool = False
    triggered_codes: tuple[str, ...] = field(default_factory=tuple)   # every filtering rule that fired, in order (for audit logging)


def evaluate_constraints(state: PolicyState, config: Optional[PolicyConfig] = None) -> ConstraintResult:
    """The single entry point implementing all 9 rules in order. See
    check_hard_constraints()/get_eligible_actions() below for thin wrappers
    matching §8's pseudocode names exactly.
    """
    config = config or PolicyConfig()

    def forced(action: Action, code: str) -> ConstraintResult:
        return ConstraintResult(forced_action=action, rationale_code=code, eligible_actions=frozenset())

    # Rule 1: unresolved dispute -> human, always, first.
    if state.dispute_flag and not state.dispute_resolved:
        return forced("human_escalation", "DISPUTE_UNRESOLVED")

    # Rule 2: permanent, never overridden by anything below.
    if state.no_contact_requested:
        return forced("none", "NO_CONTACT_HONORED")

    # Rule 3: an active promise still inside its cooling-off grace period.
    if state.active_promise_flag and state.days_until_promised_date > -config.grace_period_days:
        return forced("none", "COOLING_PERIOD_ACTIVE")

    # Rule 4: repeated broken promises -> stop automating, hand to a human.
    if state.broken_promise_streak >= 2:
        return forced("human_escalation", "PROMISE_STREAK_EXCEEDED")

    eligible = set(EIV_ELIGIBLE_ACTIONS)
    human_approval_required = False
    triggered: list[str] = []

    # Rule 5: frequency cap -- narrows eligible set, does not terminate.
    if state.contacts_in_last_3_days >= 1:
        eligible -= FREQUENCY_CAPPED_ACTIONS
        triggered.append("FREQUENCY_CAP")

    # Rule 6: too many automated attempts on this invoice -> human, terminal.
    if state.total_automated_contacts_this_invoice >= 4:
        return forced("human_escalation", "MAX_ATTEMPTS_REACHED")

    # Rule 7: high-value invoices need a human in the loop; plan_proposal
    # specifically is too high-stakes to auto-execute on a large invoice.
    if state.invoice_amount >= config.high_value_threshold:
        eligible.discard("plan_proposal")
        human_approval_required = True
        triggered.append("HIGH_VALUE_REQUIRES_APPROVAL")

    # Rule 8: plan_proposal only offered to customers with adequate PRS.
    if "plan_proposal" in eligible and state.prs_score < config.plan_eligibility_floor:
        eligible.discard("plan_proposal")
        triggered.append("PRS_BELOW_PLAN_FLOOR")

    # Rule 9: nothing forced a terminal action -- hand off to EIV.
    rationale_code = triggered[-1] if triggered else None
    return ConstraintResult(
        forced_action=None,
        rationale_code=rationale_code,
        eligible_actions=frozenset(eligible),
        human_approval_required=human_approval_required,
        triggered_codes=tuple(triggered),
    )


def check_hard_constraints(state: PolicyState, config: Optional[PolicyConfig] = None) -> Optional[Action]:
    """Thin wrapper matching §8's pseudocode name/shape exactly:
    returns the forced action, or None if no hard constraint terminated
    the decision (i.e. rule 9 -- proceed to EIV)."""
    return evaluate_constraints(state, config).forced_action


def get_eligible_actions(state: PolicyState, config: Optional[PolicyConfig] = None) -> frozenset[Action]:
    """Thin wrapper matching §8's pseudocode name/shape exactly: the
    eligible-action set EIV should rank over. Only meaningful when
    check_hard_constraints() returned None; §8's own pseudocode only calls
    this after that check, so callers should do the same."""
    return evaluate_constraints(state, config).eligible_actions
