"""
backend/rationale_explanations.py — rationale_code -> plain-language mapping
for the dashboard's audit-trail drill-down (§15/architecture §15's "why did
the agent do this" requirement).

RATIONALE_EXPLANATIONS covers every distinct rationale_code value actually
written anywhere in the codebase as of Stage 11, confirmed by grep (not a
list transcribed from the contract's original 9 rules and then extended by
memory):
    grep -rhoE 'rationale_code\\s*=\\s*"[A-Z_]+"' --include="*.py" .
    grep -nE '"[A-Z][A-Z_]{2,}"' policy/constraints.py policy/eiv.py
  -> policy/constraints.py's 8 named hard-rule codes (§9) + eiv.py's
     "EIV_MAX" (9 total), plus agent/state_machine.py's PROMISE_VERIFIED,
     PROMISE_CAPTURED, SPONTANEOUS_PROMISE_CAPTURED,
     PROMISE_CAPTURED_AMOUNT_MISMATCH, NO_COMMITMENT_DETECTED,
     EXTRACTION_BELOW_CONFIDENCE_FLOOR, ALREADY_TERMINAL,
     LIVE_SLICE_REAL_PAYMENT_LINK_CREATED, LIVE_SLICE_PAYMENT_CONFIRMED,
     and scripts/live_slice_demo.py's demo-only DEMO_FORCED_LINK_RESEND.

A code that shows up in the DB but isn't in this dict (shouldn't happen
given the grep above, but code changes over time) falls back to a visibly
honest "no explanation written yet" placeholder rather than crashing or
silently showing nothing -- see explain_code().
"""
from __future__ import annotations

RATIONALE_EXPLANATIONS: dict[str, str] = {
    # --- policy/constraints.py, §9 hard rules, in priority order ---
    "DISPUTE_UNRESOLVED": "An unresolved dispute is open on this invoice — escalated to a human immediately, ahead of every other rule.",
    "NO_CONTACT_HONORED": "The customer asked not to be contacted — the agent takes no action, permanently, regardless of any other signal.",
    "COOLING_PERIOD_ACTIVE": "A promise was made recently and its grace period hasn't elapsed yet — the agent waits rather than contacting the customer again.",
    "PROMISE_STREAK_EXCEEDED": "This customer has broken 2 or more promises in a row — the agent stops automating and hands the case to a human.",
    "FREQUENCY_CAP": "The customer was already contacted within the last 3 days — reminder-type actions are temporarily unavailable (a payment-link resend is still allowed).",
    "MAX_ATTEMPTS_REACHED": "The agent has already made 4 or more automated contact attempts on this invoice without resolution — escalated to a human.",
    "HIGH_VALUE_REQUIRES_APPROVAL": "This invoice is at or above the high-value threshold — a human must approve before any action is taken, and payment plans are disabled.",
    "PRS_BELOW_PLAN_FLOOR": "This customer's reliability score is too low to be offered a payment plan.",
    # --- policy/eiv.py, §8 ---
    "EIV_MAX": "No hard rule forced or blocked an action — the agent picked whichever eligible action has the highest expected incremental value.",
    # --- agent/state_machine.py: promise capture/verification (Stage 7) ---
    "PROMISE_VERIFIED": "A previously pending promise's grace period has now elapsed — the agent checked whether it was actually kept.",
    "PROMISE_CAPTURED": "The customer made a payment commitment as a direct result of the agent's own outreach that day.",
    "SPONTANEOUS_PROMISE_CAPTURED": "The customer made a payment commitment even though the agent took no action that day.",
    # --- agent/state_machine.py / agent/llm/extract.py: reply extraction (Stage 8) ---
    "PROMISE_CAPTURED_AMOUNT_MISMATCH": "The customer committed to a specific amount, but it doesn't match what's actually owed — recorded, but flagged for a human to review rather than auto-trusted.",
    "NO_COMMITMENT_DETECTED": "The customer's reply was reviewed but contained no genuine payment commitment.",
    "EXTRACTION_BELOW_CONFIDENCE_FLOOR": "The customer's reply looked like it might contain a commitment, but the model's confidence was too low — treated as no commitment.",
    # --- agent/state_machine.py: terminal-state short-circuit (Stage 9) ---
    "ALREADY_TERMINAL": "This invoice is already closed out (paid, escalated, or awaiting approval) — the agent does nothing further today.",
    # --- agent/state_machine.py: Razorpay live-slice (Stage 10) ---
    "LIVE_SLICE_REAL_PAYMENT_LINK_CREATED": "A real Razorpay payment link was created for this invoice — a live-slice demo invoice, not a simulated dispatch.",
    "LIVE_SLICE_PAYMENT_CONFIRMED": "Razorpay confirmed a real test-mode payment was received for this invoice — a live-slice demo invoice, not a simulated outcome.",
    # --- scripts/live_slice_demo.py: demo-only, never a real policy decision ---
    "DEMO_FORCED_LINK_RESEND": "A payment-link resend was explicitly triggered for this live-slice demo invoice (the policy engine didn't select it naturally within the demo window) — not a genuine policy decision.",
}


def explain_code(code: str) -> str:
    return RATIONALE_EXPLANATIONS.get(code, f"(no plain-language explanation written yet for rationale_code={code!r})")


def explain_rationale_codes(rationale_code: str | None) -> list[dict[str, str]]:
    """rationale_code as stored on an AuditEvent may be a single code
    ("EIV_MAX") or a comma-joined compound trail written by
    agent/state_machine.py's `",".join(auth.rationale_codes)`
    (e.g. "FREQUENCY_CAP,HIGH_VALUE_REQUIRES_APPROVAL,EIV_MAX" -- every
    hard rule that narrowed the eligible set, in the order it fired,
    followed by EIV_MAX if the decision still fell through to ranking).
    Every code in the trail gets its own explanation, in order -- not just
    the first one -- so a compound trail reads as the actual chain of
    reasoning it represents.
    """
    if not rationale_code:
        return []
    return [{"code": code, "explanation": explain_code(code)} for code in rationale_code.split(",")]
