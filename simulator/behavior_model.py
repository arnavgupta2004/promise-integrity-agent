"""
CustomerBehaviorModel and potential-outcome generation — contract §3, §5.

Section 5 gives the outcome mechanism as pseudocode with several
deliberately-open functions (`f(...)`). The concrete formulas chosen to
fill those in are documented inline below, next to each function, rather
than left implicit.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Action = Literal[
    "none", "soft_reminder", "firm_reminder", "channel_escalation",
    "link_resend", "plan_proposal", "human_escalation"
]

ACTIONS: tuple[Action, ...] = (
    "none", "soft_reminder", "firm_reminder", "channel_escalation",
    "link_resend", "plan_proposal", "human_escalation",
)


@dataclass
class CustomerLatentState:
    customer_id: str
    archetype: str
    keep_probability_base: float       # promise-keep prob
    avg_days_to_pay: float
    dispute_propensity: float
    response_propensity: float
    fatigue_sensitivity: float         # how much repeated contact reduces response
    trend_slope: float                 # drift in reliability over simulated time


@dataclass
class PotentialOutcome:
    invoice_id: str
    action: Action
    will_pay_within_N: bool
    days_to_pay: Optional[int]
    will_respond: bool
    will_promise: bool
    promise_kept: Optional[bool]


# ---------------------------------------------------------------------------
# §5 mechanism constants. The contract's pseudocode only specifies direction
# and relative magnitude ("small positive shift", "larger positive shift",
# "+ largest positive", ...) — the numeric values below are the concrete
# implementation choice, picked so the qualitative per-archetype behavior in
# §4 (and the smoke-test DoD) actually shows up in simulated output.
#
# Payment timing is modeled around a single, fixed-per-invoice "would-pay"
# day (see _baseline_payment_day) rather than by re-testing a rolling
# N-day-forward window with a fresh independent Bernoulli every simulated
# day. The latter was tried first and rejected: re-drawing an independent
# "will you pay in the next N days" coin flip on every calendar day, with a
# fresh random draw each time, means the invoice gets many highly-correlated
# "shots on goal" at a moderate per-day probability, which pushes the
# expected closing day far earlier than the archetype's real
# avg_days_to_pay (empirically: reliable-always-late customers, target ~40
# days, were closing around day 3). Anchoring on one fixed target day per
# invoice (itself drawn from Normal(avg_days_to_pay, ...), so it still
# reproduces the archetype's distribution) and asking "does that day fall
# inside today's [day, day+horizon) window" avoids the compounding effect
# and keeps `days_to_pay` statistics faithful to §4's table.
# ---------------------------------------------------------------------------

# Horizon (days) used for `will_pay_within_N` when the caller's context
# doesn't specify one. §6 notes N is "7 or 21 days" depending on risk tier;
# 21 (the more permissive tier) is used as the default.
DEFAULT_HORIZON_DAYS = 21

# Soft edge (in days) applied around the fixed target payment day: models
# residual day-to-day uncertainty in exactly when payment lands, without
# reintroducing the daily-resampling problem described above.
EDGE_FUZZ_DAYS = 2.5

# Recency window (days) beyond which a prior contact no longer triggers the
# firm_reminder fatigue penalty or the response-fatigue damping. Matches the
# FREQUENCY_CAP window later used by the policy engine (§9), for consistency.
RECENT_CONTACT_WINDOW_DAYS = 3

# `action_effect` from §5, reframed as a day-count "pull" on the target
# payment day (positive = earlier) rather than a probability delta -- this
# is what lets each action's effect compose cleanly with a fixed-anchor-day
# model. Relative ordering (small < larger < largest) mirrors §5's
# pseudocode directly.
ACTION_DAY_PULL = {
    "none": 0.0,
    "soft_reminder": 3.0,
    "firm_reminder": 8.0,
    "channel_escalation": 6.0,
    "link_resend": 2.0,
    "plan_proposal": 15.0,          # positive branch; see _action_day_pull
    "human_escalation": 20.0,       # largest pull; eval/backstop only
}
PLAN_PROPOSAL_NEGATIVE_DAY_PULL = -10.0   # serial-promisers exploit plans, don't keep them -> net delay
PLAN_PROPOSAL_KEEP_THRESHOLD = 0.5        # mirrors PLAN_ELIGIBILITY_FLOOR in §9
FIRM_REMINDER_FATIGUE_PENALTY_DAYS = 4.0

# Daily baseline chance (scaled by response_propensity) that a customer
# communicates *unprompted* on a given day (e.g. calls to say they'll pay
# soon), plus the extra response probability an actual contact action adds
# that day. §5 doesn't specify this, but without some non-zero baseline
# under action="none", will_respond/will_promise would be identically zero
# whenever the policy-under-test never contacts anyone — making the
# archetype table's response_propensity/keep_probability_base columns
# unobservable in exactly the "always none" smoke-test scenario they're
# meant to be checked with.
DAILY_SPONTANEOUS_RESPONSE_RATE = 0.13
ACTION_RESPONSE_BOOST = {
    "none": 0.0,
    "soft_reminder": 0.15,
    "firm_reminder": 0.25,
    "channel_escalation": 0.30,
    "link_resend": 0.10,
    "plan_proposal": 0.35,
    "human_escalation": 0.40,
}
# Given a response happens, probability that it contains a promise/commitment.
ACTION_PROMISE_WEIGHT = {
    "none": 0.40,             # spontaneous "I'll pay by Friday" style contact
    "soft_reminder": 0.30,
    "firm_reminder": 0.40,
    "channel_escalation": 0.35,
    "link_resend": 0.15,
    "plan_proposal": 0.75,    # a plan proposal explicitly asks for a commitment
    "human_escalation": 0.10,
}

# Fraction of fatigue_sensitivity applied as a damping multiplier on
# response probability when the customer was contacted within the recent
# window (models "how much repeated contact reduces response").
FATIGUE_RESPONSE_DAMPING = 0.5


def _stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed derived from arbitrary hashable parts.

    Uses sha256 instead of Python's salted `hash()` (randomized per
    process) so generate_potential_outcomes()/realize() are pure,
    reproducible functions of their inputs. That's what makes them
    "usable later by the eval harness without re-running the simulation":
    calling generate_potential_outcomes() again for the same
    (customer, invoice_id, day) reproduces bit-identical counterfactuals.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _logistic_cdf(x: float, mu: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-(x - mu) / scale))


def _within_window_prob(day: float, horizon: float, effective_payment_day: float) -> float:
    """`base_pay_prob` (as reframed here): smooth probability that the
    customer's fixed-but-fuzzy target payment day falls inside
    [day, day+horizon). Implemented as the probability mass a logistic
    "soft point" at effective_payment_day places inside that window, so it
    is ~0 when the window is far from the target day and rises smoothly as
    the window comes to contain it.
    """
    lo = _logistic_cdf(day, effective_payment_day, EDGE_FUZZ_DAYS)
    hi = _logistic_cdf(day + horizon, effective_payment_day, EDGE_FUZZ_DAYS)
    return max(hi - lo, 0.0)


def _action_day_pull(action: Action, latent: CustomerLatentState, days_since_last_contact: float) -> float:
    """`action_effect` branch table from §5, expressed as a day-count pull
    on the target payment day (positive = pulls payment earlier)."""
    if action == "none":
        return 0.0
    if action == "plan_proposal":
        return (
            ACTION_DAY_PULL["plan_proposal"]
            if latent.keep_probability_base > PLAN_PROPOSAL_KEEP_THRESHOLD
            else PLAN_PROPOSAL_NEGATIVE_DAY_PULL  # serial-promisers exploit plans, don't keep them
        )
    if action == "soft_reminder":
        return ACTION_DAY_PULL["soft_reminder"] * latent.response_propensity
    if action == "firm_reminder":
        pull = ACTION_DAY_PULL["firm_reminder"] * latent.response_propensity
        if days_since_last_contact <= RECENT_CONTACT_WINDOW_DAYS:
            pull -= FIRM_REMINDER_FATIGUE_PENALTY_DAYS * latent.fatigue_sensitivity
        return pull
    if action == "channel_escalation":
        # reaches non-responders better: benefit scales with (1 - response_propensity)
        return ACTION_DAY_PULL["channel_escalation"] * (1.0 - latent.response_propensity) + 1.0
    if action == "link_resend":
        return ACTION_DAY_PULL["link_resend"]
    if action == "human_escalation":
        return ACTION_DAY_PULL["human_escalation"]
    raise ValueError(f"unknown action: {action}")


def _response_prob(action: Action, latent: CustomerLatentState, days_since_last_contact: float) -> float:
    """`promise_prob = f(response_propensity, action)` in §5, split into a
    response-probability stage (this function) and a promise-given-response
    stage (ACTION_PROMISE_WEIGHT), since a promise can't happen without a
    response.
    """
    base = DAILY_SPONTANEOUS_RESPONSE_RATE * latent.response_propensity
    boost = ACTION_RESPONSE_BOOST[action] * latent.response_propensity
    prob = base + boost
    if days_since_last_contact <= RECENT_CONTACT_WINDOW_DAYS and action != "none":
        prob *= (1.0 - FATIGUE_RESPONSE_DAMPING * latent.fatigue_sensitivity)
    return min(max(prob, 0.0), 1.0)


class CustomerBehaviorModel:
    """One instance per customer. Owns the latent state and generates
    potential outcomes. Never exposed to the agent/policy — only to
    the simulator engine and the eval harness."""

    def __init__(self, latent: CustomerLatentState):
        self.latent = latent
        self._cache: dict[tuple[str, int], dict[Action, PotentialOutcome]] = {}
        self._last_call: Optional[tuple[str, int]] = None
        self._baseline_payment_day: dict[str, float] = {}

    def _get_baseline_payment_day(self, invoice_id: str, issue_day: int) -> float:
        """The fixed, per-invoice "would pay under action=none the whole
        time" target day, drawn once and memoized -- see the module-level
        comment above ACTION_DAY_PULL for why this replaces a repeatedly
        re-sampled rolling-window test. Seeded deterministically by
        (customer_id, invoice_id) only (not day/action), so it's stable
        across every call for this invoice, including calls made later by
        the eval harness without re-running the simulation.
        """
        if invoice_id not in self._baseline_payment_day:
            rng = np.random.default_rng(_stable_seed(self.latent.customer_id, invoice_id, "baseline_payment_day"))
            sampled = rng.normal(self.latent.avg_days_to_pay, max(self.latent.avg_days_to_pay * 0.18, 1.0))
            self._baseline_payment_day[invoice_id] = issue_day + max(sampled, 1.0)
        return self._baseline_payment_day[invoice_id]

    def generate_potential_outcomes(
        self, invoice_id: str, day: int, context: dict
    ) -> dict[Action, PotentialOutcome]:
        cache_key = (invoice_id, day)
        self._last_call = cache_key
        if cache_key in self._cache:
            return self._cache[cache_key]

        issue_day = context.get("issue_day", 0)
        days_since_last_contact = context.get("days_since_last_contact", float("inf"))
        horizon = context.get("horizon_days", DEFAULT_HORIZON_DAYS)
        baseline_payment_day = self._get_baseline_payment_day(invoice_id, issue_day)

        outcomes: dict[Action, PotentialOutcome] = {}
        for action in ACTIONS:
            rng = np.random.default_rng(_stable_seed(self.latent.customer_id, invoice_id, day, action))

            day_pull = _action_day_pull(action, self.latent, days_since_last_contact)
            effective_payment_day = max(baseline_payment_day - day_pull, issue_day + 1.0)

            base_pay_prob = _within_window_prob(day, horizon, effective_payment_day)
            final_pay_prob = min(max(base_pay_prob + self.latent.trend_slope * day, 0.0), 1.0)
            will_pay = bool(rng.random() < final_pay_prob)

            days_to_pay: Optional[int] = None
            if will_pay:
                jitter = rng.normal(0.0, EDGE_FUZZ_DAYS)
                days_to_pay = int(min(max(round(effective_payment_day - day + jitter), 1), horizon))

            resp_prob = _response_prob(action, self.latent, days_since_last_contact)
            will_respond = bool(rng.random() < resp_prob)

            will_promise = False
            promise_kept: Optional[bool] = None
            if will_respond:
                promise_prob = ACTION_PROMISE_WEIGHT[action]
                will_promise = bool(rng.random() < promise_prob)
                if will_promise:
                    # `promise_kept = Bernoulli(keep_probability_base), independent
                    # draw, adjusted by trend` per §5.
                    keep_prob = min(
                        max(self.latent.keep_probability_base + self.latent.trend_slope * day, 0.0), 1.0
                    )
                    promise_kept = bool(rng.random() < keep_prob)

            outcomes[action] = PotentialOutcome(
                invoice_id=invoice_id,
                action=action,
                will_pay_within_N=will_pay,
                days_to_pay=days_to_pay,
                will_respond=will_respond,
                will_promise=will_promise,
                promise_kept=promise_kept,
            )

        self._cache[cache_key] = outcomes
        return outcomes

    def realize(self, action: Action, day: int) -> PotentialOutcome:
        """Returns the single realized outcome for the action actually taken.
        Used during rollout; generate_potential_outcomes() is used by the
        eval harness to reveal counterfactuals after the fact.

        §3 gives `realize(action, day)` with no invoice_id parameter, so it
        looks up the invoice from the most recent generate_potential_outcomes()
        call for this day (the engine always calls that first, per step()).
        """
        if self._last_call is None or self._last_call[1] != day:
            raise RuntimeError(
                "realize() must be called for the same day as a prior "
                "generate_potential_outcomes() call on this model"
            )
        invoice_id, _ = self._last_call
        return self._cache[(invoice_id, day)][action]
