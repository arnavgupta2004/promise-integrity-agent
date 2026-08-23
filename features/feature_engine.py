"""
features/feature_engine.py — §6 feature vector construction + PRS
(architecture doc §0, §3).

Pure, dependency-light leaf module: reads only from backend.db (read-only,
per the Module interfaces section of the contract) plus stdlib/numpy. Never
imports models/, policy/, or agent/ — this module has no idea a propensity
model or a policy engine exist.

`build_feature_vector`/`compute_prs` take an optional `session` (defaults to
a fresh backend.db.SessionLocal()) and an optional `as_of` (defaults to
datetime.utcnow()). Neither is part of the contract's frozen call signature
(`build_feature_vector(invoice_id, intervention_type) -> dict`,
`compute_prs(customer_id) -> float`) -- both remain callable with exactly
those two/one positional args. The additions exist because a pure function
that reads mutable DB state needs *some* way to (a) accept an injected
session for testing without touching the real SQLite file, and (b) be
evaluated at a specific point in time, which `prs_trend` requires internally
(it samples PRS at several past timestamps).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db import Communication, Customer, Dispute, Invoice, Payment, Promise, SessionLocal

FEATURE_COLUMNS = [
    "relative_lateness",           # float, days_overdue / avg_days_to_pay, capped at 3.0
    "prs_score",                   # float [0,1]
    "prs_trend",                   # float, slope over last 90 sim-days
    "dispute_rate",                # float, lifetime
    "response_rate",               # float, last 5 contacts
    "partial_payment_rate",        # float, lifetime
    "amount_tier",                 # int 0-4, quantile bucket relative to customer's own history
    "days_since_last_contact",     # int
    "active_promise_flag",         # bool
    "days_until_promised_date",    # int, -1 if no active promise
    "broken_promise_streak",       # int, consecutive most-recent
    "segment",                     # categorical, one-hot or LightGBM native categorical
    "intervention_type",           # categorical: the treatment variable itself
]
TARGET = "paid_within_N"           # bool, N fixed per risk tier (7 or 21 days) -- not computed
                                    # by this module; it's derived from ground truth elsewhere (§7)

# ---------------------------------------------------------------------------
# Tunable constants. §6 doesn't pin exact defaults/windows for most of these
# -- documented here rather than left as magic numbers inline.
# ---------------------------------------------------------------------------
DEFAULT_CREDIT_TERMS_DAYS = 30      # fallback for avg_days_to_pay when a customer has no payment history yet
RELATIVE_LATENESS_CAP = 3.0         # per §6
AMOUNT_TIER_BUCKETS = 5             # "0-4" per §6 -> 5 quantile buckets
NO_CONTACT_SENTINEL_DAYS = 9999     # large sentinel for "never contacted" (days_since_last_contact is a typed int, no Optional)
RESPONSE_RATE_WINDOW = 5            # "last 5 contacts" per §6
PRS_TREND_LOOKBACK_DAYS = 90        # per §6's prs_trend comment
PRS_TREND_SAMPLE_POINTS = 10        # number of points sampled across the lookback window to fit the slope

# --- PRS formula (architecture §0): weighted sum of four components, each
# in [0,1]. promise-keep rate is called out in §0 as "the single most
# important derived feature" -> largest weight.
PRS_KEEP_WEIGHT = 0.45
PRS_TREND_WEIGHT = 0.25
PRS_DISPUTE_WEIGHT = 0.15
PRS_RESPONSE_WEIGHT = 0.15
assert abs(PRS_KEEP_WEIGHT + PRS_TREND_WEIGHT + PRS_DISPUTE_WEIGHT + PRS_RESPONSE_WEIGHT - 1.0) < 1e-9

# Neutral value used for any PRS sub-component when its underlying evidence
# is absent (a brand-new customer with no promises/disputes/contacts yet).
# Deliberately 0.5, not 0.0: PRS feeds decisions (e.g. §9's PLAN_ELIGIBILITY_FLOOR),
# and starting every new customer at 0 (maximal distrust) would bias the
# policy against everyone who is simply new, before they've done anything
# wrong. This is different from the raw §6 frequency columns (dispute_rate,
# response_rate, partial_payment_rate), which default their own "no data"
# case to 0.0 -- those are plain observed-event-rate columns (0 events / 0
# denominator = 0 is the least-assuming reading), not reliability
# judgments, so they don't get the same neutral-prior treatment.
PRS_NEUTRAL_DEFAULT = 0.5

# Exponential-decay half-life for recency-weighting the promise-keep rate:
# a promise resolved this many days ago counts for half the weight of one
# resolved today.
PRS_RECENCY_HALF_LIFE_DAYS = 30.0
PRS_RECENCY_MAX_PROMISES = 20  # cap the query; decay already makes older ones negligible

# Converts a fitted "lateness-ratio per day" slope into a shift around the
# neutral 0.5 midpoint. Chosen so a slope on the order of the simulator's
# own trend_slope magnitude (~0.01/day, see simulator/archetypes.py's
# "degrading" archetype) moves the component across roughly its full range
# -- not derived from ground truth, just a matching order of magnitude.
PRS_TREND_SCALE = 50.0


def _resolve_session(session: Optional[Session]) -> tuple[Session, bool]:
    if session is not None:
        return session, False
    return SessionLocal(), True


def _resolve_as_of(as_of: Optional[dt.datetime]) -> dt.datetime:
    return as_of if as_of is not None else dt.datetime.utcnow()


def _days_between(later: dt.datetime, earlier: dt.datetime) -> float:
    return (later - earlier).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# PRS (architecture §0): rolling promise-keep rate (recency-weighted),
# relative-lateness trend, dispute frequency, response rate to prior outreach.
# ---------------------------------------------------------------------------

def _recency_weighted_keep_rate(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    promises = (
        session.query(Promise)
        .join(Invoice, Promise.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Promise.kept.isnot(None), Promise.made_on <= as_of)
        .order_by(Promise.made_on.desc())
        .limit(PRS_RECENCY_MAX_PROMISES)
        .all()
    )
    if not promises:
        return PRS_NEUTRAL_DEFAULT

    total_weight = 0.0
    total_weighted_kept = 0.0
    for p in promises:
        days_ago = max(_days_between(as_of, p.made_on), 0.0)
        weight = 0.5 ** (days_ago / PRS_RECENCY_HALF_LIFE_DAYS)
        total_weight += weight
        if p.kept:
            total_weighted_kept += weight

    return total_weighted_kept / total_weight if total_weight > 0 else PRS_NEUTRAL_DEFAULT


def _lateness_trend_component(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    """Is the customer's payment lateness (relative to their own credit
    terms) improving or worsening over time? Fits a linear slope of
    (payment_date - due_date) / credit_terms_days against calendar time
    across the customer's paid invoices; a worsening (positive) slope pulls
    the component below 0.5, an improving (negative) slope pushes it above.
    """
    customer = session.get(Customer, customer_id)
    terms = float(customer.credit_terms_days) if customer and customer.credit_terms_days else float(DEFAULT_CREDIT_TERMS_DAYS)

    rows = (
        session.query(Payment, Invoice)
        .join(Invoice, Payment.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Payment.payment_date <= as_of)
        .order_by(Payment.payment_date.asc())
        .all()
    )
    if len(rows) < 2:
        return PRS_NEUTRAL_DEFAULT

    xs = np.array([_days_between(payment.payment_date, as_of) for payment, _ in rows])  # <= 0, oldest most negative
    ys = np.array([_days_between(payment.payment_date, invoice.due_date) / max(terms, 1.0) for payment, invoice in rows])
    slope = float(np.polyfit(xs, ys, 1)[0])
    return float(min(max(0.5 - slope * PRS_TREND_SCALE, 0.0), 1.0))


def _dispute_component(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    total_invoices = (
        session.query(func.count(Invoice.invoice_id))
        .filter(Invoice.customer_id == customer_id, Invoice.issue_date <= as_of)
        .scalar()
        or 0
    )
    if total_invoices == 0:
        return PRS_NEUTRAL_DEFAULT

    total_disputes = (
        session.query(func.count(Dispute.dispute_id))
        .join(Invoice, Dispute.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Dispute.raised_date <= as_of)
        .scalar()
        or 0
    )
    rate = total_disputes / total_invoices
    return float(min(max(1.0 - rate, 0.0), 1.0))


def _response_component(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    comms = (
        session.query(Communication)
        .join(Invoice, Communication.invoice_id == Invoice.invoice_id)
        .filter(
            Invoice.customer_id == customer_id,
            Communication.timestamp <= as_of,
            Communication.dispatched_by == "agent",
        )
        .order_by(Communication.timestamp.desc())
        .limit(RESPONSE_RATE_WINDOW)
        .all()
    )
    if not comms:
        return PRS_NEUTRAL_DEFAULT
    return float(sum(1 for c in comms if c.response_received) / len(comms))


def compute_prs(customer_id: str, as_of: Optional[dt.datetime] = None, session: Optional[Session] = None) -> float:
    """Payment/Promise Reliability Score (architecture §0): a [0,1] scalar
    combining four recency/history-aware signals:

        PRS = 0.45 * keep_component        (rolling promise-keep rate, exponentially recency-weighted, half-life 30d)
            + 0.25 * trend_component        (slope of relative-lateness over the customer's paid invoices)
            + 0.15 * dispute_component      (1 - lifetime dispute rate)
            + 0.15 * response_component     (reply rate over the last 5 agent-dispatched contacts)

    Each component defaults to PRS_NEUTRAL_DEFAULT (0.5) when its
    underlying evidence is absent, so a customer with no history at all
    returns exactly 0.5 -- a defined, non-error, non-extreme value -- rather
    than raising or silently collapsing to 0.
    """
    session, owns_session = _resolve_session(session)
    as_of = _resolve_as_of(as_of)
    try:
        keep_component = _recency_weighted_keep_rate(session, customer_id, as_of)
        trend_component = _lateness_trend_component(session, customer_id, as_of)
        dispute_component = _dispute_component(session, customer_id, as_of)
        response_component = _response_component(session, customer_id, as_of)

        prs = (
            PRS_KEEP_WEIGHT * keep_component
            + PRS_TREND_WEIGHT * trend_component
            + PRS_DISPUTE_WEIGHT * dispute_component
            + PRS_RESPONSE_WEIGHT * response_component
        )
        return float(min(max(prs, 0.0), 1.0))
    finally:
        if owns_session:
            session.close()


def _prs_trend(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    """§6's `prs_trend`: slope of the PRS *score itself* over the last 90
    simulated days -- distinct from compute_prs's own internal lateness
    trend component. Sampled by evaluating compute_prs at several past
    points and fitting a linear slope.
    """
    sample_offsets = np.linspace(-PRS_TREND_LOOKBACK_DAYS, 0, PRS_TREND_SAMPLE_POINTS)
    ys = [compute_prs(customer_id, as_of=as_of + dt.timedelta(days=float(d)), session=session) for d in sample_offsets]
    slope = float(np.polyfit(sample_offsets, ys, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# §6 feature vector
# ---------------------------------------------------------------------------

def _customer_avg_days_to_pay(session: Session, customer_id: str, as_of: dt.datetime, exclude_invoice_id: Optional[str] = None) -> float:
    q = (
        session.query(Payment, Invoice)
        .join(Invoice, Payment.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Payment.payment_date <= as_of)
    )
    if exclude_invoice_id is not None:
        q = q.filter(Invoice.invoice_id != exclude_invoice_id)
    rows = q.all()
    if not rows:
        customer = session.get(Customer, customer_id)
        terms = customer.credit_terms_days if customer and customer.credit_terms_days else DEFAULT_CREDIT_TERMS_DAYS
        return float(terms)
    days = [_days_between(payment.payment_date, invoice.issue_date) for payment, invoice in rows]
    return float(max(np.mean(days), 1.0))


def _lifetime_dispute_rate(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    total_invoices = (
        session.query(func.count(Invoice.invoice_id))
        .filter(Invoice.customer_id == customer_id, Invoice.issue_date <= as_of)
        .scalar()
        or 0
    )
    if total_invoices == 0:
        return 0.0
    total_disputes = (
        session.query(func.count(Dispute.dispute_id))
        .join(Invoice, Dispute.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Dispute.raised_date <= as_of)
        .scalar()
        or 0
    )
    return float(total_disputes / total_invoices)


def _recent_response_rate(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    comms = (
        session.query(Communication)
        .join(Invoice, Communication.invoice_id == Invoice.invoice_id)
        .filter(
            Invoice.customer_id == customer_id,
            Communication.timestamp <= as_of,
            Communication.dispatched_by == "agent",
        )
        .order_by(Communication.timestamp.desc())
        .limit(RESPONSE_RATE_WINDOW)
        .all()
    )
    if not comms:
        return 0.0
    return float(sum(1 for c in comms if c.response_received) / len(comms))


def _lifetime_partial_payment_rate(session: Session, customer_id: str, as_of: dt.datetime) -> float:
    rows = (
        session.query(Payment)
        .join(Invoice, Payment.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Payment.payment_date <= as_of)
        .all()
    )
    if not rows:
        return 0.0
    return float(sum(1 for p in rows if p.partial_flag) / len(rows))


def _amount_tier(session: Session, customer_id: str, invoice: Invoice, as_of: dt.datetime) -> int:
    rows = (
        session.query(Invoice.amount)
        .filter(Invoice.customer_id == customer_id, Invoice.issue_date <= as_of)
        .all()
    )
    amounts = sorted(a for (a,) in rows)
    if len(amounts) < 2:
        return AMOUNT_TIER_BUCKETS // 2  # not enough history to form quantiles -> neutral middle tier
    edges = np.quantile(amounts, [0.2, 0.4, 0.6, 0.8])
    tier = int(np.searchsorted(edges, invoice.amount, side="right"))
    return min(tier, AMOUNT_TIER_BUCKETS - 1)


def _days_since_last_contact(session: Session, invoice_id: str, as_of: dt.datetime) -> int:
    last = (
        session.query(Communication)
        .filter(Communication.invoice_id == invoice_id, Communication.timestamp <= as_of)
        .order_by(Communication.timestamp.desc())
        .first()
    )
    if last is None:
        return NO_CONTACT_SENTINEL_DAYS
    return int(_days_between(as_of, last.timestamp))


def _active_promise(session: Session, invoice_id: str, as_of: dt.datetime) -> tuple[bool, int]:
    promise = (
        session.query(Promise)
        .filter(Promise.invoice_id == invoice_id, Promise.kept.is_(None), Promise.made_on <= as_of)
        .order_by(Promise.made_on.desc())
        .first()
    )
    if promise is None:
        return False, -1
    if promise.promised_date is None:
        return True, -1
    return True, int(_days_between(promise.promised_date, as_of))


def _broken_promise_streak(session: Session, customer_id: str, as_of: dt.datetime) -> int:
    promises = (
        session.query(Promise)
        .join(Invoice, Promise.invoice_id == Invoice.invoice_id)
        .filter(Invoice.customer_id == customer_id, Promise.kept.isnot(None), Promise.made_on <= as_of)
        .order_by(Promise.made_on.desc())
        .all()
    )
    streak = 0
    for p in promises:
        if p.kept is False:
            streak += 1
        else:
            break
    return streak


def build_feature_vector(
    invoice_id: str,
    intervention_type: str,
    as_of: Optional[dt.datetime] = None,
    session: Optional[Session] = None,
) -> dict:
    """§6: builds the S-learner input row for one (invoice, candidate
    intervention) pair. Returns exactly FEATURE_COLUMNS. Read-only against
    backend.db; never touches models/ or policy/.
    """
    session, owns_session = _resolve_session(session)
    as_of = _resolve_as_of(as_of)
    try:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None:
            raise ValueError(f"unknown invoice_id: {invoice_id}")
        customer = session.get(Customer, invoice.customer_id)
        if customer is None:
            raise ValueError(f"invoice {invoice_id} references unknown customer {invoice.customer_id}")

        avg_days_to_pay = _customer_avg_days_to_pay(session, customer.customer_id, as_of, exclude_invoice_id=invoice_id)
        days_overdue = _days_between(as_of, invoice.due_date)
        relative_lateness = min(days_overdue / avg_days_to_pay, RELATIVE_LATENESS_CAP)

        active_promise_flag, days_until_promised_date = _active_promise(session, invoice_id, as_of)

        return {
            "relative_lateness": float(relative_lateness),
            "prs_score": compute_prs(customer.customer_id, as_of=as_of, session=session),
            "prs_trend": _prs_trend(session, customer.customer_id, as_of),
            "dispute_rate": _lifetime_dispute_rate(session, customer.customer_id, as_of),
            "response_rate": _recent_response_rate(session, customer.customer_id, as_of),
            "partial_payment_rate": _lifetime_partial_payment_rate(session, customer.customer_id, as_of),
            "amount_tier": _amount_tier(session, customer.customer_id, invoice, as_of),
            "days_since_last_contact": _days_since_last_contact(session, invoice_id, as_of),
            "active_promise_flag": active_promise_flag,
            "days_until_promised_date": days_until_promised_date,
            "broken_promise_streak": _broken_promise_streak(session, customer.customer_id, as_of),
            "segment": customer.segment or "unknown",
            "intervention_type": intervention_type,
        }
    finally:
        if owns_session:
            session.close()
