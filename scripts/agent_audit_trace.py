"""
Runs the same 20-invoice/30-day batch as tests/test_agent_loop.py and
pretty-prints the full audit-log trail for a handful of representative
invoices: one that escalates (to confirm the log goes quiet afterward),
one kept promise, one broken promise, one that pays off normally, and one
quiet/clean model-citizen-like invoice.
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
from agent.tools import PropensityModelTool
from backend.db import AuditLog, Base, Customer, Invoice, Promise
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

SEED = 99
N_DAYS = 30
REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)

TEST_ARCHETYPE_MIX = (
    ["serial_promiser"] * 10 + ["disputer"] * 5 + ["non_responsive"] * 3 + ["model_citizen"] * 2
)


def build_batch(seed: int):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    latent_rng = np.random.default_rng(seed)
    customers: dict[str, CustomerBehaviorModel] = {}
    invoice_ids: list[str] = []

    for i, archetype in enumerate(TEST_ARCHETYPE_MIX):
        customer_id = f"agenttest-{i:03d}-{archetype}"
        latent = sample_customer_latent(customer_id, archetype, latent_rng)
        customers[customer_id] = CustomerBehaviorModel(latent)

        session.add(Customer(
            customer_id=customer_id, name=customer_id, archetype=archetype, segment="SMB",
            credit_terms_days=5, onboarding_date=REFERENCE_START - dt.timedelta(days=200),
        ))
        invoice_id = f"inv-{customer_id}-0"
        session.add(Invoice(
            invoice_id=invoice_id, customer_id=customer_id, amount=10_000.0,
            issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=5),
            status="open",
        ))
        invoice_ids.append(invoice_id)
    session.commit()

    return session, customers, invoice_ids


def _seq(log_id: str) -> int:
    # log_id = "audit-{invoice_id}-{step}-{seq}" -- seq is the process-wide
    # monotonic counter, the only reliable chronological ordering key
    # (log_id as a whole string-sorts by step name first, which scrambles order)
    return int(log_id.rsplit("-", 1)[-1])


def print_trace(session, invoice_id: str, label: str) -> None:
    invoice = session.get(Invoice, invoice_id)
    events = session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()
    events.sort(key=lambda ev: _seq(ev.log_id))
    print(f"\n{'=' * 90}")
    print(f"{label}  |  invoice={invoice_id}  |  final status={invoice.status}  |  {len(events)} audit events")
    print("=" * 90)
    for ev in events:
        day = (ev.timestamp - REFERENCE_START).days
        parts = [f"day {day:3d}", f"[{ev.step:8s}]"]
        if ev.rationale_code:
            parts.append(f"rationale={ev.rationale_code}")
        if ev.decision:
            parts.append(f"decision={ev.decision}")
        if ev.executed_action:
            parts.append(f"executed={ev.executed_action}")
        if ev.constraint_triggered:
            parts.append(f"constraint={ev.constraint_triggered}")
        if ev.human_approval_required:
            parts.append("approval_required=True")
        print("  " + "  ".join(parts))


def main() -> None:
    session, customers, invoice_ids = build_batch(SEED)
    model = PropensityModel()
    propensity_tool = PropensityModelTool(model)
    pending_promise_truth: dict[str, bool] = {}

    for day in range(N_DAYS):
        for invoice_id in invoice_ids:
            invoice = session.get(Invoice, invoice_id)
            customer_model = customers[invoice.customer_id]
            ctx = AgentContext(
                day=day, reference_start=REFERENCE_START, session=session,
                customer_model=customer_model, propensity_model_tool=propensity_tool,
                policy_config=CONFIG, pending_promise_truth=pending_promise_truth,
            )
            run_agent_cycle(invoice_id, ctx)

    escalated = [iid for iid in invoice_ids if session.get(Invoice, iid).status == "human_queue"]
    paid = [iid for iid in invoice_ids if session.get(Invoice, iid).status == "paid"]
    kept_promise_invoices = [p.invoice_id for p in session.query(Promise).filter(Promise.kept.is_(True)).all()]
    broken_promise_invoices = [p.invoice_id for p in session.query(Promise).filter(Promise.kept.is_(False)).all()]
    model_citizen_invoices = [iid for iid in invoice_ids if "model_citizen" in iid]

    print(f"Batch summary: {len(invoice_ids)} invoices, {N_DAYS} days")
    print(f"  escalated (human_queue): {len(escalated)}")
    print(f"  paid: {len(paid)}")
    print(f"  still open: {len(invoice_ids) - len(escalated) - len(paid)}")
    print(f"  invoices with a kept promise: {len(set(kept_promise_invoices))}")
    print(f"  invoices with a broken promise: {len(set(broken_promise_invoices))}")

    picks = []
    if escalated:
        picks.append((escalated[0], "ESCALATED (terminal)"))
    if broken_promise_invoices:
        picks.append((broken_promise_invoices[0], "BROKEN PROMISE"))
    if kept_promise_invoices:
        picks.append((kept_promise_invoices[0], "KEPT PROMISE"))
    if paid:
        picks.append((paid[0], "PAID OFF"))
    if model_citizen_invoices:
        picks.append((model_citizen_invoices[0], "MODEL-CITIZEN-LIKE (quiet)"))
    # fill up to 6 with anything not already picked
    picked_ids = {iid for iid, _ in picks}
    for iid in invoice_ids:
        if len(picks) >= 6:
            break
        if iid not in picked_ids:
            picks.append((iid, "OTHER"))
            picked_ids.add(iid)

    for invoice_id, label in picks:
        print_trace(session, invoice_id, label)


if __name__ == "__main__":
    main()
