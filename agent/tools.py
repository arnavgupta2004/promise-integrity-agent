"""
agent/tools.py — §10 tool registry, implementing architecture §9's Layer
2 (policy engine) / Layer 3 (LLM, stubbed here) as tools the orchestration
loop (Layer 1, agent/state_machine.py) calls, never as controllers of their
own. Each concrete tool exposes both a generic `call(**kwargs) -> dict`
(matching §10's `Tool` Protocol exactly) and a specific, type-checked
method (matching §10's own bullet-list names) for use inside the loop.

LLMDraftTool is an explicit placeholder per this stage's scope: template
strings only, no real LLM call. Stage 8 replaces its body, not its
interface. LLMExtractTool is likewise stubbed and, at this stage, never
actually invoked by the loop -- there is no free-text customer reply to
parse here; the simulator's PotentialOutcome already says definitively
whether/when a promise was made (see agent/state_machine.py's docstring on
why verification is nonetheless deferred rather than read off immediately).
It's kept only for interface completeness against §10's tool list, ready
for Stage 8 to give it a real body once real reply text exists.
"""
from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass
from typing import Optional, Protocol

from sqlalchemy.orm import Session

from backend.db import AuditLog, Communication
from features.feature_engine import build_feature_vector
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig, PolicyState
from policy.eiv import select_action


class Tool(Protocol):
    name: str

    def call(self, **kwargs) -> dict: ...


@dataclass
class AuditEvent:
    """§12: one row per action or state change. rationale_code is a single
    Optional[str] per the schema; a compound rationale trail (see
    policy/eiv.py's select_action docstring) is joined with ',' before
    landing here -- the schema doesn't have room for a list, so joining is
    the lossless-enough compromise, not a silent drop.
    """
    invoice_id: str
    customer_id: str
    step: str  # "detect" | "diagnose" | "decide" | "act" | "verify" | "reassess"
    input_snapshot: dict
    model_output: Optional[dict]
    decision: Optional[str]
    rationale_code: Optional[str]
    constraint_triggered: Optional[str]
    executed_action: Optional[str]
    human_approval_required: bool
    timestamp: Optional[dt.datetime] = None  # simulated as_of date; falls back to wall-clock utcnow() if omitted


class FeatureEngineTool:
    name = "feature_engine"

    def __init__(self, session: Session):
        self.session = session

    def build_feature_vector(self, invoice_id: str, as_of: dt.datetime) -> dict:
        return build_feature_vector(invoice_id, "none", as_of=as_of, session=self.session)

    def call(self, **kwargs) -> dict:
        return self.build_feature_vector(kwargs["invoice_id"], kwargs["as_of"])


class PropensityModelTool:
    name = "propensity_model"

    def __init__(self, model: PropensityModel):
        self.model = model

    def score(self, features: dict, action: str) -> float:
        return self.model.predict_proba({**features, "intervention_type": action})

    def call(self, **kwargs) -> dict:
        return {"probability": self.score(kwargs["features"], kwargs["action"])}


@dataclass
class AuthDecision:
    action: str
    rationale_codes: tuple[str, ...]
    human_approval_required: bool


class PolicyEngineTool:
    """Layer 2 (architecture §9): deterministic, non-negotiable. The
    orchestration loop calls this and must accept its verdict verbatim --
    it never overrides or second-guesses what comes back.
    """
    name = "policy_engine"

    def __init__(self, propensity_model_tool: PropensityModelTool, config: Optional[PolicyConfig] = None):
        self.propensity_model_tool = propensity_model_tool
        self.config = config or PolicyConfig()

    def authorize(self, invoice_amount: float, base_features: dict, state: PolicyState) -> AuthDecision:
        action, rationale_codes, human_approval_required = select_action(
            invoice_amount, base_features, self.propensity_model_tool.model, state, self.config
        )
        return AuthDecision(action=action, rationale_codes=rationale_codes, human_approval_required=human_approval_required)

    def call(self, **kwargs) -> dict:
        decision = self.authorize(kwargs["invoice_amount"], kwargs["base_features"], kwargs["state"])
        return {"action": decision.action, "rationale_codes": decision.rationale_codes,
                "human_approval_required": decision.human_approval_required}


_DRAFT_TEMPLATES = {
    "soft_reminder": "Hi {customer_name}, just a friendly reminder that invoice {invoice_id} is now due. Let us know if you have any questions.",
    "firm_reminder": "Hi {customer_name}, invoice {invoice_id} is now overdue. Please arrange payment at your earliest convenience.",
    "channel_escalation": "Hi {customer_name}, we haven't been able to reach you about invoice {invoice_id} -- please get in touch so we can resolve this together.",
    "link_resend": "Hi {customer_name}, here's your payment link for invoice {invoice_id} again in case the original was missed.",
    "plan_proposal": "Hi {customer_name}, we'd like to offer a payment plan for invoice {invoice_id} -- reply and we'll set it up.",
    "human_escalation": "Invoice {invoice_id} has been routed to a human agent for manual follow-up.",
}


class LLMDraftTool:
    """STUB (this stage only): returns a filled-in template string, no LLM
    call of any kind. Layer 3 (architecture §9): narrow, tool-scoped, one
    tool among several -- Stage 8 gives this a real bounded LLM call
    without changing this class's interface.
    """
    name = "llm_draft"

    def draft_message(self, action_type: str, context: dict) -> str:
        template = _DRAFT_TEMPLATES.get(action_type, "Regarding invoice {invoice_id}: {action_type}.")
        return template.format(
            customer_name=context.get("customer_id", "there"),
            invoice_id=context.get("invoice_id", "?"),
            action_type=action_type,
        )

    def call(self, **kwargs) -> dict:
        return {"message": self.draft_message(kwargs["action_type"], kwargs["context"])}


@dataclass
class PromiseExtraction:
    commitment_detected: bool
    promised_date: Optional[str]
    promised_amount: Optional[float]
    confidence: float
    notes: str


class LLMExtractTool:
    """STUB, not invoked at this stage -- see module docstring."""
    name = "llm_extract"

    def extract_promise(self, reply_text: str) -> Optional[PromiseExtraction]:
        return None

    def call(self, **kwargs) -> dict:
        result = self.extract_promise(kwargs["reply_text"])
        return {"extraction": result}


class AuditLogTool:
    """log_id uses a process-wide monotonic counter (class-level, not
    per-instance) -- run_agent_cycle() constructs a fresh AuditLogTool on
    every call, so a per-instance counter would restart at 0 each time and
    collide across calls for the same (invoice_id, step). The counter also
    doubles as a reliable global write-order key (tests use it to confirm
    no decide/act event was written after an invoice's escalation).
    """
    name = "audit_log"
    _global_counter = itertools.count(1)

    def __init__(self, session: Session):
        self.session = session

    def write(self, event: AuditEvent) -> None:
        seq = next(AuditLogTool._global_counter)
        self.session.add(AuditLog(
            log_id=f"audit-{event.invoice_id}-{event.step}-{seq}",
            timestamp=event.timestamp or dt.datetime.utcnow(),
            invoice_id=event.invoice_id,
            customer_id=event.customer_id,
            step=event.step,
            input_snapshot=event.input_snapshot,
            model_output=event.model_output,
            decision=event.decision,
            rationale_code=event.rationale_code,
            constraint_triggered=event.constraint_triggered,
            executed_action=event.executed_action,
            human_approval_required=event.human_approval_required,
        ))
        self.session.commit()

    def call(self, **kwargs) -> dict:
        self.write(kwargs["event"])
        return {}
