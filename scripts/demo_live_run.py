"""
scripts/demo_live_run.py — the "live loop run" beat of the 5-minute demo
(scripts/demo_script.md). Runs the REAL Stage 7 agent loop
(agent.state_machine.run_agent_cycle) live, in the terminal, for two fresh
invoices over 12 simulated days -- one reliable customer, one unreliable
one -- printing each phase's audit event as it happens, so a reviewer
watches detect->diagnose->decide->act->reassess actually execute rather
than looking at a pre-computed table.

Deliberately independent of the Gemini API: uses agent.tools.LLMDraftTool
(the free, deterministic Stage-7 stub), not agent.llm.draft's real
Gemini-backed one. This keeps runtime to a couple of seconds and removes
any dependence on today's quota state -- see demo_script.md's explicit
requirement not to route any live demo beat through a first-try-dependent
Gemini call. Message drafting quality isn't the point of this beat; the
loop's own decision-making (policy engine + trained propensity model) is
real and unmodified either way.

Uses an in-memory DB, entirely separate from data/audit_completeness_batch.db
-- this beat is illustrative or a fresh, small run, not a re-run of the
batch the dashboard/eval numbers are drawn from.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.state_machine import AgentContext, run_agent_cycle
from agent.tools import AuditLogTool, LLMDraftTool, PropensityModelTool
from backend.db import AuditLog, Base, Customer, Invoice
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

REFERENCE_START = dt.datetime(2025, 1, 1)
N_DAYS = 12
DEMO_CUSTOMERS = [
    ("demo-live-reliable", "model_citizen", 18_000.0),
    ("demo-live-unreliable", "serial_promiser", 18_000.0),
]


def _seq(log_id: str) -> int:
    return int(log_id.rsplit("-", 1)[-1])


def build() -> tuple:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    rng = np.random.default_rng(2026)
    customers, invoice_ids = {}, []
    for i, (customer_id, archetype, amount) in enumerate(DEMO_CUSTOMERS):
        latent = sample_customer_latent(customer_id, archetype, rng)
        customers[customer_id] = CustomerBehaviorModel(latent)
        session.add(Customer(
            customer_id=customer_id, name=customer_id, archetype=archetype, segment="SMB",
            credit_terms_days=10, onboarding_date=REFERENCE_START - dt.timedelta(days=120),
        ))
        invoice_id = f"inv-{customer_id}-0"
        session.add(Invoice(
            invoice_id=invoice_id, customer_id=customer_id, amount=amount,
            issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=10), status="open",
        ))
        invoice_ids.append(invoice_id)
    session.commit()
    return session, customers, invoice_ids


def main() -> None:
    session, customers, invoice_ids = build()
    propensity_tool = PropensityModelTool(PropensityModel())
    pending_promise_truth: dict[str, bool] = {}
    draft_tool = LLMDraftTool()  # free stub -- no Gemini dependency, see module docstring

    def make_ctx(day: int, customer_id: str) -> AgentContext:
        return AgentContext(
            day=day, reference_start=REFERENCE_START, session=session,
            customer_model=customers[customer_id], propensity_model_tool=propensity_tool,
            policy_config=PolicyConfig(), pending_promise_truth=pending_promise_truth, draft_tool=draft_tool,
        )

    print(f"=== Live agent loop: {len(invoice_ids)} invoices, {N_DAYS} days ===\n")
    for day in range(N_DAYS):
        print(f"--- day {day} ({(REFERENCE_START + dt.timedelta(days=day)).date()}) ---")
        for invoice_id in invoice_ids:
            invoice = session.get(Invoice, invoice_id)
            before_count = session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).count()
            ctx = make_ctx(day, invoice.customer_id)
            run_agent_cycle(invoice_id, ctx)
            session.commit()
            events = (
                session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()
            )
            events.sort(key=lambda e: _seq(e.log_id))
            new_events = events[before_count:]
            for e in new_events:
                if e.step in ("decide", "act", "verify") and (e.decision or e.rationale_code):
                    print(f"  [{invoice_id}] {e.step:8s} decision={e.decision!s:16s} "
                          f"executed={e.executed_action!s:14s} rationale={e.rationale_code}")
        print()

    print("=== Final state ===")
    for invoice_id in invoice_ids:
        inv = session.get(Invoice, invoice_id)
        print(f"  {invoice_id}: status={inv.status}")


if __name__ == "__main__":
    main()
