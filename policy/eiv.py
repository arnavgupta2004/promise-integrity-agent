"""
policy/eiv.py — §8 EIV (expected incremental value) action selection.

Depends on policy/constraints.py only (plus stdlib typing) -- no import of
simulator or models. §8's pseudocode calls build_feature_vector(invoice,
customer, intervention_type=...) and model.predict_proba(...) directly, but
wiring this module to a live DB session or an actual trained model artifact
would violate "must not depend on the trained model being ready": both are
taken as parameters instead. `model` is duck-typed (anything exposing
.predict_proba(feature_dict) -> float, e.g. models.propensity_model.
PropensityModel, or a trivial stub in a test); `base_features` is a
pre-computed feature dict (typically the output of
features.build_feature_vector(invoice_id, "none", ...) from the caller) --
this module only ever overrides its "intervention_type" key per candidate
action, exactly mirroring §7's "score the same feature vector once per
candidate action (varying only intervention_type)".
"""
from __future__ import annotations

from typing import Optional, Protocol

from policy.constraints import Action, PolicyConfig, PolicyState, evaluate_constraints

# §8: "ACTION_COST is a fixed cost table (e.g., soft_reminder=5, firm_reminder=15,
# channel_escalation=20, link_resend=5, plan_proposal=100 [human review time],
# human_escalation=300)." human_escalation is never EIV-scored (§7), included
# here only for completeness/documentation.
ACTION_COST: dict[Action, float] = {
    "none": 0.0,
    "soft_reminder": 5.0,
    "firm_reminder": 15.0,
    "channel_escalation": 20.0,
    "link_resend": 5.0,
    "plan_proposal": 100.0,
    "human_escalation": 300.0,
}


class PredictsProba(Protocol):
    def predict_proba(self, feature_vector: dict) -> float: ...


def rank_by_eiv(
    invoice_amount: float,
    base_features: dict,
    model: PredictsProba,
    eligible_actions: frozenset[Action],
) -> tuple[Action, float]:
    """§8's ranking loop exactly: best_action, best_eiv = "none", 0.0; for
    each eligible action, eiv = amount * (p - p_none) - cost[action]; keep
    the max. "none" is never beaten by an action with eiv <= 0, matching
    the initial baseline.
    """
    features_none = {**base_features, "intervention_type": "none"}
    p_none = model.predict_proba(features_none)

    best_action, best_eiv = "none", 0.0
    for action in eligible_actions:
        if action == "none":
            continue
        features = {**base_features, "intervention_type": action}
        p = model.predict_proba(features)
        eiv = invoice_amount * (p - p_none) - ACTION_COST[action]
        if eiv > best_eiv:
            best_action, best_eiv = action, eiv

    return best_action, best_eiv


def select_action(
    invoice_amount: float,
    base_features: dict,
    model: PredictsProba,
    state: PolicyState,
    config: Optional[PolicyConfig] = None,
) -> tuple[Action, tuple[str, ...], bool]:
    """§8's select_action(invoice, customer, model, policy_constraints) ->
    Action, adapted to this module's actual inputs (see module docstring).
    Returns (action, rationale_codes, human_approval_required).

    Step 1: hard constraints (§9) -- short-circuits with a forced action if any fires.
    Step 2: EIV ranking (this file) over whatever eligible set constraints left.

    rationale_codes is the FULL ordered trail, not a single collapsed code:
    for a terminal rule it's a one-element tuple (e.g. ("DISPUTE_UNRESOLVED",));
    when multiple non-terminal rules (5/7/8) narrow the eligible set before
    falling through to EIV, every one of them appears, in the order they
    fired, with "EIV_MAX" appended last. A single "rationale_code" string
    would silently drop all but the last non-terminal rule whenever more
    than one fires on the same invoice (e.g. both the frequency cap and the
    high-value-approval rule) -- exactly the audit-completeness gap §12
    and Stage 9 need to not have. Callers writing to AuditEvent.rationale_code
    (a single Optional[str] per §12) should join this tuple (e.g. ",".join(...))
    rather than take just one element.
    """
    result = evaluate_constraints(state, config)
    if result.forced_action is not None:
        return result.forced_action, (result.rationale_code,), result.human_approval_required

    if not result.eligible_actions:
        return "none", result.triggered_codes, result.human_approval_required

    best_action, _ = rank_by_eiv(invoice_amount, base_features, model, result.eligible_actions)
    rationale_codes = result.triggered_codes + ("EIV_MAX",)
    return best_action, rationale_codes, result.human_approval_required
