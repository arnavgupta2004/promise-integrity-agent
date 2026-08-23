"""
agent/state_machine.py — §10 orchestration loop; Layer 1 of architecture
§9's four-layer split (the only genuinely autonomous layer). Detect →
Diagnose → Decide → Act → Verify → Reassess, run per invoice, writing an
AuditEvent per phase per §12.

State lives in the DB (Invoice.status + Promise rows), per §10's own note
("no separate in-memory agent state store needed at this scale"). The one
exception, documented below, is `pending_promise_truth`: a deliberate
delayed-reveal of the simulator's already-known ground truth, needed to
exercise genuine autonomous reassessment (see "Deferred promise
verification").

--- Deferred promise verification ---
Stage 1's simulator resolves `promise_kept` in the same call that produces
`will_promise=True` -- there's no simulated "pending, not yet known"
window. Reading that value straight into Promise.kept at ACT time would
short-circuit the very state-machine property this stage is supposed to
demonstrate: reassessing *after time passes*, not immediately. So ACT
writes `Promise.kept=None` (genuinely pending in the DB -- feature_engine's
active_promise_flag/days_until_promised_date and policy rule 3's cooling
period both key off exactly this), and the simulator's already-computed
answer is held in `AgentContext.pending_promise_truth` (a plain dict, owned
by whichever test/driver constructs the context -- analogous to the eval
harness's own privileged access to potential outcomes, kept out of the
DB/feature layer entirely) until VERIFY reveals it once
promised_date + grace_period_days has actually elapsed. This is what
architecture §10 calls "deterministic" verification -- checking on a
schedule, not reading a value that was secretly known all along.

--- Terminal state (Stage 6 fix) ---
Invoice.status moves to "human_queue" the moment human_escalation fires
(policy rules 4/6), or "paid" once realize() reports a landed payment.
Both are checked first, before any other phase runs -- an invoice already
in either state is skipped (just a lightweight "still terminal" detect
event, never decide/act), which is what stops the ~6x escalation
re-flagging Stage 6's eval harness measured from ever reaching the live
audit log.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Optional

from sqlalchemy.orm import Session

from backend.db import Communication, Dispute, Invoice, Payment, Promise
from policy.constraints import PolicyConfig, PolicyState
from simulator.behavior_model import CustomerBehaviorModel

from agent.llm.extract import LLMExtractTool, PromiseExtraction
from agent.tools import (
    AuditEvent,
    AuditLogTool,
    FeatureEngineTool,
    LLMDraftTool,
    PolicyEngineTool,
    PropensityModelTool,
)

PENDING_APPROVAL_STATUS = "pending_human_approval"
# Distinct from "human_queue" on purpose, not merged into it: human_queue
# (rules 4/6) means the customer relationship itself has broken down enough
# that a human needs to take over. pending_human_approval (rule 7) means
# nothing about trust has broken -- the policy engine just won't let the
# agent unilaterally send *this one class of action* on a high-value
# invoice without sign-off. Conflating the two would make an audit-trail
# reviewer unable to tell "this needs a human because we're worried about
# them" from "this needs a human because of the dollar amount" without
# reading rationale_code. Both share the exact same mechanical
# stop-processing pattern (TERMINAL_INVOICE_STATUSES + the lightweight
# ALREADY_TERMINAL detect-only skip), so the cost of keeping them separate
# is one extra string constant, not a second code path.
TERMINAL_INVOICE_STATUSES = {"paid", "human_queue", PENDING_APPROVAL_STATUS}
DEFAULT_PROMISE_HORIZON_DAYS = 7  # how far out a captured promise's promised_date is set, absent a specific date

Phase = Literal["detect", "diagnose", "decide", "act", "verify", "reassess", "closed"]


@dataclass
class InvoiceAgentState:
    invoice_id: str
    phase: Phase
    next_reassess_at: Optional[dt.datetime]


@dataclass
class AgentContext:
    """Everything run_agent_cycle needs beyond invoice_id. §10's own
    pseudocode signature is run_agent_cycle(invoice_id) -> None; the state
    that actually persists (Invoice.status, Promise rows, AuditLog) lives
    in the DB exactly as that section says. What's bundled here is the
    simulated environment's equivalent of "today's date" and "which
    service to call" -- not agent state, external dependencies -- made
    explicit rather than reached for as module globals, consistent with
    every other frozen-signature-vs-testability tradeoff elsewhere in this
    project (features.build_feature_vector's session/as_of,
    SimulationEngine's eval_mode, etc.).
    """
    day: int
    reference_start: dt.datetime
    session: Session
    customer_model: CustomerBehaviorModel
    propensity_model_tool: PropensityModelTool
    policy_config: PolicyConfig
    pending_promise_truth: dict[str, bool] = field(default_factory=dict)
    # Injectable outreach-drafting tool. Defaults to None, meaning "use
    # agent.tools's free Stage-7 stub" (preserves every existing test's
    # zero-cost, deterministic behavior unchanged). Callers that want
    # Stage 8's real Gemini-backed drafting (e.g. a live batch run) pass an
    # agent.llm.draft.LLMDraftTool instance here instead -- both share the
    # same draft_message(action_type, context) -> str interface, so
    # run_agent_cycle's call site doesn't need to know which one it has.
    draft_tool: Optional[object] = None
    # --- Stage 10: §13 live-slice (real Razorpay test-mode API calls) ---
    # Defaults (None / empty set/dict) preserve every existing test's fully
    # simulated behavior unchanged -- a caller must opt an invoice into the
    # live slice explicitly by both passing a constructed
    # integration.razorpay_client.RazorpayClient AND listing the invoice_id
    # here. Untyped as `object` for the same reason draft_tool is: the call
    # site only needs the 5 §13 methods, not a hard import-time dependency.
    razorpay_client: Optional[object] = None
    live_slice_invoice_ids: set[str] = field(default_factory=set)
    # invoice_id -> Razorpay payment_link_id, the live counterpart of
    # pending_promise_truth: state that isn't a simulator ground-truth
    # value, but still doesn't belong in the frozen §2 schema (no
    # payment-link-id column exists), so it's tracked here instead.
    live_slice_payment_links: dict[str, str] = field(default_factory=dict)
    # --- Stage 12: rule 2 (NO_CONTACT_HONORED) ---
    # There is no "customer opted out of contact" column anywhere in the
    # frozen §2 schema -- unlike rule 1 (dispute resolution), which the
    # schema already models via the Dispute table, this genuinely has
    # nowhere else to live. Same shape as live_slice_invoice_ids: a caller
    # opts a customer in explicitly; default (empty set) preserves every
    # existing test's behavior unchanged. Keyed by customer_id, not
    # invoice_id, because a real do-not-contact request is a customer-level
    # fact, not specific to one of their invoices.
    no_contact_customer_ids: set[str] = field(default_factory=set)


def load_state(invoice_id: str, ctx: AgentContext) -> InvoiceAgentState:
    invoice = ctx.session.get(Invoice, invoice_id)
    if invoice is None:
        raise ValueError(f"unknown invoice_id: {invoice_id}")
    if invoice.status in TERMINAL_INVOICE_STATUSES:
        return InvoiceAgentState(invoice_id=invoice_id, phase="closed", next_reassess_at=None)

    pending_promise = (
        ctx.session.query(Promise)
        .filter(Promise.invoice_id == invoice_id, Promise.kept.is_(None))
        .order_by(Promise.made_on.desc())
        .first()
    )
    if pending_promise is not None:
        next_reassess_at = pending_promise.promised_date + dt.timedelta(days=ctx.policy_config.grace_period_days)
        return InvoiceAgentState(invoice_id=invoice_id, phase="reassess", next_reassess_at=next_reassess_at)

    return InvoiceAgentState(invoice_id=invoice_id, phase="detect", next_reassess_at=None)


def _as_of(ctx: AgentContext) -> dt.datetime:
    return ctx.reference_start + dt.timedelta(days=ctx.day)


def sync_dispute_state(invoice: Invoice, customer_model: CustomerBehaviorModel, session: Session,
                        as_of: dt.datetime, reference_start: dt.datetime) -> None:
    """Stage 13: materializes CustomerBehaviorModel.dispute_outcome()'s
    pure, simulator-generated dispute timeline into real Dispute rows /
    Invoice.dispute_flag, the moment each simulated day crosses the
    relevant threshold -- mirrors _settle_payment_if_any's pure-outcome-
    to-DB-write pattern exactly (a simulator-side "will this happen, and
    when" answer, materialized into the DB only once that day actually
    arrives).

    Exported (not underscore-prefixed): eval/run_eval.py's Agent arm calls
    this SAME function (the contract's module rules explicitly permit
    eval -> agent dependencies) rather than reimplementing dispute
    materialization a second time with its own drift risk -- the same
    reason rule 1/2's PolicyState derivation logic needs to stay identical
    between the two call sites.
    """
    issue_day = (invoice.issue_date - reference_start).days
    as_of_day = (as_of - reference_start).days
    outcome = customer_model.dispute_outcome(invoice.invoice_id, issue_day)
    if not outcome.will_dispute:
        return

    existing = session.query(Dispute).filter(Dispute.invoice_id == invoice.invoice_id).first()
    if existing is None:
        if as_of_day >= outcome.raised_day:
            session.add(Dispute(
                dispute_id=f"disp-{invoice.invoice_id}", invoice_id=invoice.invoice_id,
                raised_date=reference_start + dt.timedelta(days=outcome.raised_day),
                reason="simulated_dispute", resolved=False,
            ))
            invoice.dispute_flag = True
            session.commit()
        return

    if not existing.resolved and as_of_day >= outcome.resolved_day:
        existing.resolved = True
        existing.resolution_date = reference_start + dt.timedelta(days=outcome.resolved_day)
        session.commit()


def _verify_pending_promises(invoice_id: str, customer_id: str, ctx: AgentContext, audit: AuditLogTool, as_of: dt.datetime) -> None:
    pending = (
        ctx.session.query(Promise)
        .filter(Promise.invoice_id == invoice_id, Promise.kept.is_(None))
        .all()
    )
    for p in pending:
        grace_deadline = p.promised_date + dt.timedelta(days=ctx.policy_config.grace_period_days)
        if as_of < grace_deadline:
            continue  # not due for verification yet -- rule 3 (COOLING_PERIOD_ACTIVE) keeps decide/act quiet meanwhile

        kept = ctx.pending_promise_truth.get(p.promise_id, False)
        p.kept = kept
        p.broken_reason = None if kept else "not_kept_by_grace_deadline"
        ctx.session.commit()

        audit.write(AuditEvent(
            invoice_id=invoice_id, customer_id=customer_id, step="verify",
            input_snapshot={"promise_id": p.promise_id, "promised_date": str(p.promised_date), "grace_deadline": str(grace_deadline)},
            model_output=None,
            decision="promise_kept" if kept else "promise_broken",
            rationale_code="PROMISE_VERIFIED",
            constraint_triggered=None, executed_action=None, human_approval_required=False,
            timestamp=as_of,
        ))


def _build_policy_state(invoice: Invoice, features: dict, ctx: AgentContext, as_of: dt.datetime) -> PolicyState:
    contacts_recent = (
        ctx.session.query(Communication)
        .filter(
            Communication.invoice_id == invoice.invoice_id,
            Communication.timestamp > as_of - dt.timedelta(days=3),
            Communication.timestamp <= as_of,
        ).count()
    )
    total_contacts = (
        ctx.session.query(Communication)
        .filter(Communication.invoice_id == invoice.invoice_id, Communication.dispatched_by == "agent")
        .count()
    )
    # Rule 1 (DISPUTE_UNRESOLVED): dispute_resolved is derived from the real
    # Dispute table (frozen §2 schema already has it -- raised_date/resolved
    # -- it just wasn't being read here before Stage 12). A dispute-flagged
    # invoice with no open (resolved=False) Dispute row counts as resolved;
    # dispute_flag=False makes this value irrelevant to rule 1 either way.
    unresolved_dispute = (
        ctx.session.query(Dispute)
        .filter(Dispute.invoice_id == invoice.invoice_id, Dispute.resolved.is_(False))
        .first()
    )
    return PolicyState(
        invoice_amount=invoice.amount,
        dispute_flag=bool(invoice.dispute_flag), dispute_resolved=unresolved_dispute is None,
        # Rule 2 (NO_CONTACT_HONORED): see AgentContext.no_contact_customer_ids
        # -- the frozen schema has no column for this, so it's tracked there.
        no_contact_requested=invoice.customer_id in ctx.no_contact_customer_ids,
        active_promise_flag=features["active_promise_flag"],
        days_until_promised_date=features["days_until_promised_date"],
        broken_promise_streak=features["broken_promise_streak"],
        contacts_in_last_3_days=contacts_recent,
        total_automated_contacts_this_invoice=total_contacts,
        prs_score=features["prs_score"],
    )


def _record_promise(invoice_id: str, customer_id: str, chosen_action: str, ctx: AgentContext,
                     outcome, audit: AuditLogTool, as_of: dt.datetime) -> None:
    """Symmetric with _verify_pending_promises's own AuditEvent on
    resolution: capture gets a visible event too, not just a silent DB row.
    Distinguishes, via rationale_code, whether the promise was a direct
    consequence of the agent's own chosen action that day (chosen_action !=
    "none") or a spontaneous customer-initiated promise on a day the agent
    chose "none" -- Stage 1's simulator can produce will_promise=True either
    way (see simulator/behavior_model.py's DAILY_SPONTANEOUS_RESPONSE_RATE),
    and without this distinction a reviewer reading only the day's routine
    "act" event (decision=none, executed=none) would have no way to know a
    promise was captured that day at all.
    """
    promise_id = f"prom-{invoice_id}-{ctx.day}"
    promised_date = as_of + dt.timedelta(days=DEFAULT_PROMISE_HORIZON_DAYS)
    ctx.session.add(Promise(
        promise_id=promise_id, invoice_id=invoice_id, promised_date=promised_date,
        promised_amount=None, made_on=as_of, extraction_confidence=0.9,
        kept=None,  # pending -- see module docstring
    ))
    ctx.session.commit()
    ctx.pending_promise_truth[promise_id] = bool(outcome.promise_kept)

    is_spontaneous = chosen_action == "none"
    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="act",
        input_snapshot={
            "promise_id": promise_id, "promised_date": str(promised_date),
            "promised_amount": None, "made_on": str(as_of), "triggering_action": chosen_action,
        },
        model_output=None,
        decision="promise_captured",
        rationale_code="SPONTANEOUS_PROMISE_CAPTURED" if is_spontaneous else "PROMISE_CAPTURED",
        constraint_triggered=None, executed_action=None, human_approval_required=False,
        timestamp=as_of,
    ))


def _parse_promised_date(date_str: Optional[str], as_of: dt.datetime) -> dt.datetime:
    if date_str:
        try:
            return dt.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    return as_of + dt.timedelta(days=DEFAULT_PROMISE_HORIZON_DAYS)


def process_customer_reply(
    invoice_id: str, customer_id: str, reply_text: str, ctx: AgentContext, as_of: dt.datetime,
    owed_amount: Optional[float] = None, extract_tool: Optional[LLMExtractTool] = None,
) -> PromiseExtraction:
    """Stage 8: real customer reply text -> §11 structured extraction
    (agent/llm/extract.py, Gemini structured output) -> the SAME
    audit-event pattern Stage 7 established for promise capture
    (PROMISE_CAPTURED / SPONTANEOUS_PROMISE_CAPTURED).

    Every reply gets a visible audit event, whether or not a promise
    resulted -- a reply that looked promise-like but was correctly rejected
    (vague language, or a genuine commitment whose confidence fell below
    §11's 0.6 floor) is exactly as traceable as one that succeeded. Two
    distinct rejection rationale codes so a reviewer can tell, from the
    rationale_code alone, which kind of rejection happened without
    re-reading the extraction:
      - NO_COMMITMENT_DETECTED: the model itself found no genuine commitment
      - EXTRACTION_BELOW_CONFIDENCE_FLOOR: the model DID detect one, but
        confidence < 0.6 forced commitment_detected to False (§11's rule)

    A captured promise whose stated amount doesn't match what's actually
    owed is still recorded (a real commitment was made) but flagged via
    PROMISE_CAPTURED_AMOUNT_MISMATCH and human_approval_required=True --
    "flag, don't auto-accept" -- rather than silently trusting a promise
    for the wrong amount.
    """
    audit = AuditLogTool(ctx.session)
    tool = extract_tool or LLMExtractTool()
    extraction = tool.extract_promise(reply_text, owed_amount=owed_amount)

    if extraction.commitment_detected:
        promise_id = f"prom-{invoice_id}-{ctx.day}-reply"
        promised_date = _parse_promised_date(extraction.promised_date, as_of)
        ctx.session.add(Promise(
            promise_id=promise_id, invoice_id=invoice_id, promised_date=promised_date,
            promised_amount=extraction.promised_amount, made_on=as_of,
            extraction_confidence=extraction.confidence, kept=None,
        ))
        ctx.session.commit()
        ctx.pending_promise_truth[promise_id] = True  # no simulator ground truth for a real reply; verify checks payments when the time comes

        rationale = "PROMISE_CAPTURED_AMOUNT_MISMATCH" if extraction.amount_mismatch else "PROMISE_CAPTURED"
        audit.write(AuditEvent(
            invoice_id=invoice_id, customer_id=customer_id, step="act",
            input_snapshot={
                "reply_text": reply_text, "promise_id": promise_id,
                "promised_date": str(promised_date), "promised_amount": extraction.promised_amount,
                "confidence": extraction.confidence, "owed_amount": owed_amount,
                "amount_mismatch": extraction.amount_mismatch, "notes": extraction.notes,
            },
            model_output={"raw_commitment_detected": extraction.raw_commitment_detected},
            decision="promise_captured", rationale_code=rationale,
            constraint_triggered=None, executed_action=None,
            human_approval_required=bool(extraction.amount_mismatch),
            timestamp=as_of,
        ))
    else:
        rationale = "EXTRACTION_BELOW_CONFIDENCE_FLOOR" if extraction.confidence_floor_applied else "NO_COMMITMENT_DETECTED"
        audit.write(AuditEvent(
            invoice_id=invoice_id, customer_id=customer_id, step="act",
            input_snapshot={
                "reply_text": reply_text, "confidence": extraction.confidence, "notes": extraction.notes,
                "raw_commitment_detected": extraction.raw_commitment_detected,
            },
            model_output=None,
            decision="no_promise_captured", rationale_code=rationale,
            constraint_triggered=None, executed_action=None, human_approval_required=False,
            timestamp=as_of,
        ))

    return extraction


def _dispatch(invoice_id: str, action: str, message: str, outcome, ctx: AgentContext, as_of: dt.datetime) -> None:
    ctx.session.add(Communication(
        comm_id=f"comm-{invoice_id}-{ctx.day}", invoice_id=invoice_id, channel="email",
        timestamp=as_of, message_type=action, message_text=message, dispatched_by="agent",
        response_received=bool(outcome.will_respond), response_text=None,
    ))
    ctx.session.commit()


def _dispatch_live_payment_link(invoice: Invoice, ctx: AgentContext, audit: AuditLogTool, as_of: dt.datetime) -> None:
    """link_resend on a live-slice invoice: create a REAL Razorpay Payment
    Link via ctx.razorpay_client instead of going through the simulated
    _dispatch path. Proving this call genuinely reaches Razorpay's
    test-mode API -- not standing in for it -- is the entire point of §13's
    live slice, so this is a parallel code path, not a parameterization of
    _dispatch.
    """
    invoice_id = invoice.invoice_id
    link = ctx.razorpay_client.create_payment_link({
        "amount": int(round(invoice.amount * 100)),  # Razorpay amounts are in paise
        "currency": "INR",
        "description": f"Payment for invoice {invoice_id}",
        "reference_id": invoice_id,
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
    })
    ctx.live_slice_payment_links[invoice_id] = link["id"]

    message = f"Please complete your payment for invoice {invoice_id} using this secure link: {link['short_url']}"
    ctx.session.add(Communication(
        comm_id=f"comm-{invoice_id}-{ctx.day}", invoice_id=invoice_id, channel="email",
        timestamp=as_of, message_type="link_resend", message_text=message, dispatched_by="agent",
        response_received=False, response_text=None,
    ))
    ctx.session.commit()

    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=invoice.customer_id, step="act",
        input_snapshot={"amount": invoice.amount, "reference_id": invoice_id},
        model_output={"razorpay_payment_link_id": link["id"], "short_url": link["short_url"], "status": link.get("status")},
        decision="link_resend", rationale_code="LIVE_SLICE_REAL_PAYMENT_LINK_CREATED",
        constraint_triggered=None, executed_action="link_resend", human_approval_required=False,
        timestamp=as_of,
    ))


def _verify_live_slice_payment(invoice: Invoice, ctx: AgentContext, audit: AuditLogTool, as_of: dt.datetime) -> None:
    """Live-slice invoices are settled by a real human completing a real
    Razorpay test-mode checkout, not by the simulator's potential-outcomes
    model -- this is the live counterpart to _settle_payment_if_any.
    Instead of reading a pre-computed simulated outcome, it makes a real
    fetch_payment_link call and only marks the invoice paid once Razorpay
    itself reports the link as paid.
    """
    invoice_id = invoice.invoice_id
    link_id = ctx.live_slice_payment_links.get(invoice_id)
    if link_id is None or ctx.razorpay_client is None:
        return  # no live payment link created yet for this invoice

    link = ctx.razorpay_client.fetch_payment_link(link_id)
    if link.get("status") != "paid":
        return  # not yet completed (manual test-mode checkout pending) -- nothing to verify this cycle

    payments = link.get("payments") or []
    payment_id = payments[0]["payment_id"] if payments else None
    payment = ctx.razorpay_client.fetch_payment(payment_id) if payment_id else {}

    ctx.session.add(Payment(
        payment_id=f"pay-{invoice_id}-live", invoice_id=invoice_id,
        amount_paid=(payment.get("amount") or link.get("amount_paid") or 0) / 100.0,
        payment_date=as_of, partial_flag=False,
        razorpay_payment_id=payment_id,
    ))
    invoice.status = "paid"
    ctx.session.commit()

    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=invoice.customer_id, step="verify",
        input_snapshot={"razorpay_payment_link_id": link_id, "razorpay_payment_id": payment_id},
        model_output={"payment_link": link, "payment": payment},
        decision="live_payment_confirmed", rationale_code="LIVE_SLICE_PAYMENT_CONFIRMED",
        constraint_triggered=None, executed_action=None, human_approval_required=False,
        timestamp=as_of,
    ))


def _settle_payment_if_any(invoice: Invoice, outcome, ctx: AgentContext, as_of: dt.datetime) -> None:
    if not outcome.will_pay_within_N:
        return
    payment_date = as_of + dt.timedelta(days=outcome.days_to_pay or 0)
    ctx.session.add(Payment(
        payment_id=f"pay-{invoice.invoice_id}", invoice_id=invoice.invoice_id,
        amount_paid=invoice.amount, payment_date=payment_date, partial_flag=False,
    ))
    invoice.status = "paid"
    ctx.session.commit()


def run_agent_cycle(invoice_id: str, ctx: AgentContext) -> None:
    audit = AuditLogTool(ctx.session)
    as_of = _as_of(ctx)
    invoice = ctx.session.get(Invoice, invoice_id)
    if invoice is None:
        raise ValueError(f"unknown invoice_id: {invoice_id}")
    customer_id = invoice.customer_id

    # --- Stage 13: materialize the simulator's dispute timeline (if any)
    # for this invoice. Deliberately BEFORE the terminal-state short-circuit
    # below, not after: DISPUTE_UNRESOLVED (rule 1) forces human_escalation
    # the SAME cycle a dispute is raised, which makes the invoice terminal
    # immediately -- if this call sat after the terminal check (like the
    # live-slice one below), every later cycle would return before ever
    # reaching it again, and Dispute.resolved could structurally never
    # become True for ANY invoice, no matter how much simulated time
    # passed. Running it here means a dispute can still resolve after
    # escalation (a human takes over the actual resolution; the agent
    # doesn't act on it further either way, but the DB correctly reflects
    # a real dispute lifecycle instead of "always open forever").
    sync_dispute_state(invoice, ctx.customer_model, ctx.session, as_of, ctx.reference_start)

    # --- terminal-state short-circuit (Stage 6 fix) ---
    if invoice.status in TERMINAL_INVOICE_STATUSES:
        audit.write(AuditEvent(
            invoice_id=invoice_id, customer_id=customer_id, step="detect",
            input_snapshot={"invoice_status": invoice.status},
            model_output=None, decision=None, rationale_code="ALREADY_TERMINAL",
            constraint_triggered=None, executed_action=None, human_approval_required=False,
            timestamp=as_of,
        ))
        return

    # --- live-slice: check real Razorpay payment status before anything
    # else this cycle (§13) -- if a prior link_resend's link has now been
    # paid, settle it and stop; running detect/diagnose/decide/act against
    # an invoice Razorpay already reports as paid would be stale ---
    if invoice_id in ctx.live_slice_invoice_ids:
        _verify_live_slice_payment(invoice, ctx, audit, as_of)
        if invoice.status == "paid":
            return

    # --- verify: resolve any promise whose grace period has elapsed ---
    _verify_pending_promises(invoice_id, customer_id, ctx, audit, as_of)

    # --- detect + diagnose ---
    feature_tool = FeatureEngineTool(ctx.session)
    features = feature_tool.build_feature_vector(invoice_id, as_of)
    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="detect",
        input_snapshot=features, model_output=None, decision=None,
        rationale_code=None, constraint_triggered=None, executed_action=None, human_approval_required=False,
        timestamp=as_of,
    ))
    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="diagnose",
        input_snapshot=features, model_output=None, decision=None,
        rationale_code=None, constraint_triggered=None, executed_action=None, human_approval_required=False,
        timestamp=as_of,
    ))

    # --- decide: policy engine (hard constraints, then EIV) ---
    policy_state = _build_policy_state(invoice, features, ctx, as_of)
    policy_tool = PolicyEngineTool(ctx.propensity_model_tool, ctx.policy_config)
    auth = policy_tool.authorize(invoice.amount, features, policy_state)
    rationale_joined = ",".join(auth.rationale_codes)
    # last code before "EIV_MAX" (if any) is the constraint that narrowed
    # the eligible set; a bare ("EIV_MAX",) or a single forced-rule code
    # means no *additional* constraint beyond the decision itself
    constraint_triggered = None
    if auth.rationale_codes and auth.rationale_codes[-1] == "EIV_MAX" and len(auth.rationale_codes) > 1:
        constraint_triggered = auth.rationale_codes[0]
    elif auth.rationale_codes and auth.rationale_codes[-1] != "EIV_MAX":
        constraint_triggered = auth.rationale_codes[-1]
    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="decide",
        input_snapshot=features, model_output={"chosen_action": auth.action},
        decision=auth.action, rationale_code=rationale_joined,
        constraint_triggered=constraint_triggered, executed_action=None,
        human_approval_required=auth.human_approval_required,
        timestamp=as_of,
    ))

    # --- act ---
    executed_action: Optional[str] = None
    if auth.action == "human_escalation":
        invoice.status = "human_queue"
        ctx.session.commit()
        executed_action = "human_escalation"
    elif auth.human_approval_required:
        # Mirrors process_customer_reply's PROMISE_CAPTURED_AMOUNT_MISMATCH
        # pattern exactly (that pathway is the reference implementation,
        # not redesigned here): record what was PROPOSED via `decision`,
        # never set executed_action, never dispatch. Applies regardless of
        # whether auth.action is a real action or "none" -- the policy
        # engine flagged this invoice/cycle as needing a human's eyes
        # before the agent does *anything* further here, not just before
        # it sends a message. Stops automation the same way human_escalation
        # already does (PENDING_APPROVAL_STATUS is in TERMINAL_INVOICE_STATUSES),
        # just tagged with a distinct status so a reviewer can tell why.
        invoice.status = PENDING_APPROVAL_STATUS
        ctx.session.commit()
        executed_action = None
    else:
        context = {
            "issue_day": (invoice.issue_date - ctx.reference_start).days,
            "due_day": (invoice.due_date - ctx.reference_start).days,
            "days_since_last_contact": features["days_since_last_contact"],
            "horizon_days": 30,
        }
        ctx.customer_model.generate_potential_outcomes(invoice_id, ctx.day, context)
        outcome = ctx.customer_model.realize(auth.action, ctx.day)
        executed_action = auth.action

        is_live_link_resend = (
            auth.action == "link_resend"
            and invoice_id in ctx.live_slice_invoice_ids
            and ctx.razorpay_client is not None
        )
        if is_live_link_resend:
            _dispatch_live_payment_link(invoice, ctx, audit, as_of)
        elif auth.action != "none":
            draft_tool = ctx.draft_tool or LLMDraftTool()
            message = draft_tool.draft_message(auth.action, {"invoice_id": invoice_id, "customer_id": customer_id})
            _dispatch(invoice_id, auth.action, message, outcome, ctx, as_of)

        if outcome.will_promise:
            _record_promise(invoice_id, customer_id, auth.action, ctx, outcome, audit, as_of)

        # Live-slice invoices are settled by _verify_live_slice_payment
        # (real Razorpay status) at the top of a later cycle, never by the
        # simulator's potential-outcomes model -- settling here off a
        # simulated outcome would fake the very thing §13 is meant to prove.
        if invoice_id not in ctx.live_slice_invoice_ids:
            _settle_payment_if_any(invoice, outcome, ctx, as_of)

    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="act",
        input_snapshot={"chosen_action": auth.action}, model_output=None,
        decision=auth.action, rationale_code=rationale_joined,
        constraint_triggered=constraint_triggered, executed_action=executed_action,
        human_approval_required=auth.human_approval_required,
        timestamp=as_of,
    ))

    # --- reassess: schedule (observable in the audit trail; the actual
    # re-invocation is the outer daily sweep visiting every open invoice
    # again tomorrow -- rule 3's cooling period is what makes that sweep
    # correctly do nothing until the scheduled time, autonomously, with no
    # invoice-specific re-triggering from outside) ---
    state = load_state(invoice_id, ctx)
    audit.write(AuditEvent(
        invoice_id=invoice_id, customer_id=customer_id, step="reassess",
        input_snapshot={"next_reassess_at": str(state.next_reassess_at) if state.next_reassess_at else None, "phase": state.phase},
        model_output=None, decision=None, rationale_code=None,
        constraint_triggered=None, executed_action=None, human_approval_required=False,
        timestamp=as_of,
    ))
