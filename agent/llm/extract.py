"""
agent/llm/extract.py — §11's structured promise-extraction schema (Layer 3,
architecture §9: "narrow, tool-scoped... invoked by the orchestration loop
as one tool among several").

Uses Gemini's native structured/JSON-mode output (response_schema on the
generation config, expressed as a Pydantic model) -- not free-text parsing.
The model is constrained to emit exactly the §11 schema's shape; there is
no regex/substring JSON-extraction-from-prose anywhere in this file.

Hard rule (§11): confidence < 0.6 forces commitment_detected = false
regardless of the raw model output. This is the concrete implementation of
failure case #8 -- fail toward "ask a human" rather than trust a
low-confidence extraction, never the reverse.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from agent.llm._retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
CONFIDENCE_FLOOR = 0.6  # §11's exact threshold

# A stated amount is considered "mismatched" against what's actually owed
# if it differs by more than this fraction -- small differences (rounding,
# a customer citing an amount net of a discount they believe applies) don't
# need a mismatch flag; a customer promising to pay for a clearly different
# invoice does.
AMOUNT_MISMATCH_TOLERANCE = 0.05


class PromiseExtractionSchema(BaseModel):
    """Exact §11 JSON schema, expressed as a Pydantic model for Gemini's
    response_schema. google-genai converts this directly into the
    structured-output schema the model is constrained to -- this class IS
    the schema, not a post-hoc validator for free text.
    """
    commitment_detected: bool
    promised_date: Optional[str] = None
    promised_amount: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


@dataclass
class PromiseExtraction:
    commitment_detected: bool
    promised_date: Optional[str]
    promised_amount: Optional[float]
    confidence: float
    notes: str
    raw_commitment_detected: bool           # what the model said, before the confidence floor rule
    confidence_floor_applied: bool          # True iff the floor rule is what flipped commitment_detected to False
    amount_mismatch: Optional[bool] = None  # None = not evaluated (no owed_amount given); True = stated amount doesn't match what's owed -- flag, don't auto-accept


EXTRACTION_PROMPT_TEMPLATE = """You are analyzing a customer's reply to a payment reminder for an overdue invoice.
Today's date is {today}. Determine whether the customer has made a concrete commitment to pay -- a specific date and/or amount -- as opposed to a vague, non-committal, hostile, or refusing response.

Customer reply:
\"\"\"{reply_text}\"\"\"

Extract:
- commitment_detected: true only if there is a genuine, specific commitment (a date and/or amount stated or clearly implied). Vague reassurance like "soon" or "we'll sort it out" with no specific date or amount is NOT a commitment. A conditional commitment ("I'll pay once X happens") still counts as commitment_detected=true if a concrete condition, date, or amount is given -- note the condition in `notes`.
- promised_date: the committed payment date if stated, in YYYY-MM-DD format. Resolve day-only references ("the 15th", "next Friday") against today's date above -- if they name a day-of-month that has already passed this month, assume they mean that day in the next calendar month. Use null only if truly no date is stated or implied.
- promised_amount: the committed payment amount as a number if stated (null if no amount is stated)
- confidence: your confidence (0.0-1.0) that this is a genuine, specific commitment as opposed to vague language, a conditional/uncertain statement, or refusal
- notes: one brief sentence explaining your reasoning
"""


class LLMExtractTool:
    name = "llm_extract"

    def __init__(self, client: Optional[genai.Client] = None, model: str = GEMINI_MODEL):
        self.client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def extract_promise(self, reply_text: str, owed_amount: Optional[float] = None) -> PromiseExtraction:
        today = dt.date.today().isoformat()
        response = call_with_retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=EXTRACTION_PROMPT_TEMPLATE.format(reply_text=reply_text, today=today),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromiseExtractionSchema,
                temperature=0.0,
            ),
        ))
        raw: PromiseExtractionSchema = response.parsed

        confidence_floor_applied = raw.commitment_detected and raw.confidence < CONFIDENCE_FLOOR
        commitment_detected = raw.commitment_detected and raw.confidence >= CONFIDENCE_FLOOR

        amount_mismatch: Optional[bool] = None
        if commitment_detected and raw.promised_amount is not None and owed_amount is not None and owed_amount > 0:
            relative_diff = abs(raw.promised_amount - owed_amount) / owed_amount
            amount_mismatch = relative_diff > AMOUNT_MISMATCH_TOLERANCE

        return PromiseExtraction(
            commitment_detected=commitment_detected,
            promised_date=raw.promised_date if commitment_detected else None,
            promised_amount=raw.promised_amount if commitment_detected else None,
            confidence=raw.confidence,
            notes=raw.notes,
            raw_commitment_detected=raw.commitment_detected,
            confidence_floor_applied=confidence_floor_applied,
            amount_mismatch=amount_mismatch,
        )

    def call(self, **kwargs) -> dict:
        result = self.extract_promise(kwargs["reply_text"], kwargs.get("owed_amount"))
        return {"extraction": result}
