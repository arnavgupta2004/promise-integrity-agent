"""
agent/llm/draft.py — outreach message drafting (Layer 3, architecture §9:
"narrow, tool-scoped... one tool among several"). Replaces Stage 7's
template-only LLMDraftTool placeholder with a real (Gemini) bounded call --
still narrow, still just drafts text, nothing more.

Every draft is guardrail-checked before it's returned -- "LLM output is
never sent unchecked." A draft that fails the guardrail never reaches the
caller; draft_message() falls back to the same safe static template Stage 7
used instead. human_escalation never reaches this tool at all: it's an
internal routing event, not customer-facing outreach (agent/state_machine.py
never dispatches a message for it).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from google import genai

from agent.llm._retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"

TONE_GUIDANCE = {
    "soft_reminder": "friendly and light-touch -- a gentle nudge, warm in tone, assumes good faith and that this is likely just an oversight",
    "firm_reminder": "direct and clear about urgency, but professional -- no aggression, no threats, just a clear statement that the invoice is now overdue and needs attention",
    "channel_escalation": "formal, like a proper escalation notice -- more serious in tone than a firm reminder, states clearly that prior attempts to reach the customer have been unsuccessful, but still professional and never threatening",
    "link_resend": "neutral and helpful, low-friction -- brief, just re-sharing the payment link in case the original was missed",
    "plan_proposal": "collaborative and solution-oriented -- offers to work out a payment plan together, empathetic framing",
}

# "LLM output is never sent unchecked" -- no threats, no legal language in
# any drafted customer-facing message (human_escalation is the only place
# legal/collections language would ever be appropriate, and that action
# never goes through this tool at all -- see module docstring).
BANNED_PHRASES = [
    "legal action", "lawsuit", "sue you", "litigation", "attorney", "law firm",
    "collections agency", "collection agency", "debt collector",
    "credit score", "credit report", "damage your credit", "credit bureau",
    "seize", "garnish", "repossess", "court", "penalty of law", "criminal",
    "we will report you", "consequences", "immediate action or else",
]

FALLBACK_TEMPLATES = {
    "soft_reminder": "Hi {customer_name}, just a friendly reminder that invoice {invoice_id} is now due. Let us know if you have any questions.",
    "firm_reminder": "Hi {customer_name}, invoice {invoice_id} is now overdue. Please arrange payment at your earliest convenience.",
    "channel_escalation": "Hi {customer_name}, we haven't been able to reach you about invoice {invoice_id} -- please get in touch so we can resolve this together.",
    "link_resend": "Hi {customer_name}, here's your payment link for invoice {invoice_id} again in case the original was missed.",
    "plan_proposal": "Hi {customer_name}, we'd like to offer a payment plan for invoice {invoice_id} -- reply and we'll set it up.",
}

DRAFT_PROMPT_TEMPLATE = """Write a short outreach message to a customer about an overdue invoice.

Action type: {action_type}
Required tone: {tone}

Context:
- customer: {customer_name}
- invoice: {invoice_id}
{extra_context}

Rules:
- 2-4 sentences, plain text, no subject line, no signature block
- Never use threats, legal language, or mention collections/credit bureaus/lawsuits
- Never invent specific amounts, dates, or facts not given in the context above
- Write only the message body, nothing else
"""


def _check_guardrail(text: str) -> tuple[bool, list[str]]:
    lowered = text.lower()
    hits = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    return len(hits) == 0, hits


@dataclass
class DraftResult:
    message: str
    passed_guardrail: bool
    guardrail_hits: list[str]
    used_fallback: bool


class LLMDraftTool:
    name = "llm_draft"

    def __init__(self, client: Optional[genai.Client] = None, model: str = GEMINI_MODEL):
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def draft_message(self, action_type: str, context: dict) -> str:
        return self.draft(action_type, context).message

    def draft(self, action_type: str, context: dict) -> DraftResult:
        tone = TONE_GUIDANCE.get(action_type)
        if tone is None:
            # unknown/human_escalation-style action_type -- not customer
            # outreach, no LLM call, just the safe static line
            fallback = FALLBACK_TEMPLATES.get(action_type, "Regarding invoice {invoice_id}: {action_type}.").format(
                customer_name=context.get("customer_id", "there"),
                invoice_id=context.get("invoice_id", "?"),
                action_type=action_type,
            )
            return DraftResult(message=fallback, passed_guardrail=True, guardrail_hits=[], used_fallback=True)

        extra_lines = []
        if context.get("relative_lateness") is not None:
            extra_lines.append(f"- how overdue: {context['relative_lateness']:.1f}x their usual payment timing")
        if context.get("days_since_last_contact") is not None:
            extra_lines.append(f"- days since last contact: {context['days_since_last_contact']}")
        extra_context = "\n".join(extra_lines)

        prompt = DRAFT_PROMPT_TEMPLATE.format(
            action_type=action_type, tone=tone,
            customer_name=context.get("customer_id", "there"),
            invoice_id=context.get("invoice_id", "?"),
            extra_context=extra_context,
        )

        response = call_with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt,
        ))
        draft_text = (response.text or "").strip()

        passed, hits = _check_guardrail(draft_text)
        if not draft_text or not passed:
            fallback = FALLBACK_TEMPLATES.get(action_type, "Regarding invoice {invoice_id}: {action_type}.").format(
                customer_name=context.get("customer_id", "there"),
                invoice_id=context.get("invoice_id", "?"),
                action_type=action_type,
            )
            return DraftResult(message=fallback, passed_guardrail=passed, guardrail_hits=hits, used_fallback=True)

        return DraftResult(message=draft_text, passed_guardrail=True, guardrail_hits=[], used_fallback=False)

    def call(self, **kwargs) -> dict:
        result = self.draft(kwargs["action_type"], kwargs["context"])
        return {"message": result.message, "used_fallback": result.used_fallback}
