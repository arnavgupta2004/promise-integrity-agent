"""
scripts/live_slice_demo.py — Stage 10 / contract §13 live-slice demo.

Creates a handful of REAL Razorpay test-mode invoices (via
RazorpayClient.create_invoice), tags them as live-slice in AgentContext, and
runs them through the actual agent.state_machine.run_agent_cycle loop --
the same loop the rest of this project uses -- so link_resend on one of
these invoices creates a REAL Payment Link (create_payment_link) instead of
a simulated dispatch, and the verify phase checks REAL payment status
(fetch_payment_link / fetch_payment) instead of the simulator's
potential-outcomes model.

Amounts are deliberately kept well below Stage 9's ₹50,000
high_value_threshold (policy/constraints.py's HIGH_VALUE_REQUIRES_APPROVAL
trigger): a live-slice invoice landing in pending_human_approval would never
reach a real dispatch call, defeating the point of this stage.

--- Why a forced trigger exists ---
The propensity model's EIV choice isn't guaranteed to land on link_resend
within a short demo window -- same reachability gap
scripts/audit_completeness_check.py hit with Stage 8's amount-mismatch path,
where the natural simulated loop couldn't reach a real customer-reply
example either and an explicit, clearly-labeled injection was used instead.
Here: if the natural daily sweep hasn't dispatched a real link_resend for
ANY live-slice invoice by the end of this run, one designated invoice gets
an explicit trigger -- via the exact same _dispatch_live_payment_link
function the real agent loop calls, so the Razorpay API path itself is
never faked -- preceded by a "decide" AuditEvent carrying rationale_code
DEMO_FORCED_LINK_RESEND. A reviewer reading the audit log can always tell
policy-selected link_resends (rationale_code from the real decide phase,
e.g. "EIV_MAX") apart from this one (DEMO_FORCED_LINK_RESEND) -- nothing is
disguised as a genuine policy decision.

--- Idempotent, two-run design (no flags needed) ---
Real Razorpay API calls are irreversible, so every step here is guarded:
customers/invoices are only created if not already in the local DB,
create_invoice is only called if an invoice doesn't already have a
razorpay_invoice_id, and the forced trigger only fires if no live-slice
invoice has a payment_link_id yet. Local tracking state that has nowhere to
live in the frozen §2 schema -- payment_link_id per invoice, and which
simulated "day" the sweep is up to -- is kept in a small JSON sidecar file
(not the DB), loaded at the start of every run and saved at the end.

MANUAL STEP (required, cannot be automated -- see contract §13):
  1. Run this script:  python3 scripts/live_slice_demo.py
  2. It prints a short_url for the invoice that got a real payment link.
     Open it in a browser.
  3. Complete the payment using a Razorpay test-mode card:
       card number: 4111 1111 1111 1111, any future expiry, any CVV,
       any OTP -- Razorpay's test-mode checkout accepts these unconditionally.
  4. Run this script again:  python3 scripts/live_slice_demo.py
     _verify_live_slice_payment (already wired into run_agent_cycle) will
     pick up the real "paid" status via fetch_payment_link/fetch_payment
     and settle the invoice for real in the local DB + audit log.

--- Known limitation: this account's manual-payment step is blocked ---
On the test-mode account used for this demo, step 3 above cannot actually
be completed: Razorpay's hosted checkout rejects BOTH documented domestic
test cards (Visa 4111 1111 1111 1111 and Mastercard 5104 0155 5555 5558)
with "international cards not supported." Per Razorpay's own docs this
traces to KYC/international-payment eligibility gating on the ACCOUNT, not
to the payment link, the card numbers, or anything in this integration --
there is nothing in razorpay_client.py, this script, or the state-machine
wiring to fix here.

What IS verified working, with real evidence, up to that account-level
wall:
  - create_invoice: 7/7 real invoices created (e.g. inv_TTIK59VNhhz0mC),
    visible in the test-mode dashboard.
  - create_payment_link, reached via the REAL decide->act policy path (no
    forced trigger needed): plink_TTILoDkBek4Gkt, short_url
    https://rzp.io/rzp/kEvQD3qn, logged in the audit trail with rationale
    LIVE_SLICE_REAL_PAYMENT_LINK_CREATED and the raw API response attached.
  - fetch_payment_link: confirmed working via a standalone smoke test
    (status="created" returned correctly for a freshly created link).

fetch_payment is the one §13 call this account's own limitation prevents
exercising end-to-end via a real completed checkout. See
tests/test_live_slice_verify.py for how the downstream verification logic
(_verify_live_slice_payment: DB update, invoice.status transition to
"paid", audit event) is instead confirmed by injecting a "captured"-status
fetch_payment_link/fetch_payment response via a mocked razorpay_client --
i.e. everything on this side of the Razorpay API boundary is proven
correct; only the account's own checkout eligibility is not.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

from agent.state_machine import AgentContext, _dispatch_live_payment_link, run_agent_cycle
from agent.tools import AuditEvent, AuditLogTool, LLMDraftTool, PropensityModelTool
from backend.db import Base, Communication, Customer, Invoice
from integration.razorpay_client import RazorpayClient
from models.propensity_model import PropensityModel
from policy.constraints import PolicyConfig
from simulator.archetypes import ARCHETYPE_NAMES, sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel

REFERENCE_START = dt.datetime(2025, 1, 1)
CONFIG = PolicyConfig(high_value_threshold=50_000.0, plan_eligibility_floor=0.5)
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "live_slice_demo.db"
SIDECAR_PATH = Path(__file__).resolve().parents[1] / "data" / "live_slice_demo_state.json"

# All deliberately < CONFIG.high_value_threshold (₹50,000) -- see module docstring.
LIVE_SLICE_AMOUNTS = [4_500.0, 8_200.0, 12_500.0, 18_700.0, 24_300.0, 31_500.0, 41_800.0]
DAYS_PER_RUN = 10


def _load_sidecar() -> dict:
    if SIDECAR_PATH.exists():
        return json.loads(SIDECAR_PATH.read_text())
    return {"payment_links": {}, "next_day": 0}


def _save_sidecar(state: dict) -> None:
    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR_PATH.write_text(json.dumps(state, indent=2))


def build_or_load_batch(session, rzp: RazorpayClient) -> list[str]:
    """Idempotent: an invoice already present in the DB (and already
    carrying a razorpay_invoice_id) is loaded, not re-created -- re-running
    this script never creates duplicate Razorpay invoices."""
    rng = np.random.default_rng(seed=1013)
    invoice_ids: list[str] = []

    for i, amount in enumerate(LIVE_SLICE_AMOUNTS):
        customer_id = f"live-slice-{i:02d}"
        invoice_id = f"inv-{customer_id}-0"
        invoice_ids.append(invoice_id)

        customer = session.get(Customer, customer_id)
        if customer is None:
            archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
            sample_customer_latent(customer_id, archetype, rng)  # advance rng deterministically, latent unused here
            session.add(Customer(
                customer_id=customer_id, name=f"Live Slice Demo Customer {i:02d}", archetype=archetype,
                segment="SMB", credit_terms_days=15, onboarding_date=REFERENCE_START - dt.timedelta(days=200),
            ))
            session.add(Invoice(
                invoice_id=invoice_id, customer_id=customer_id, amount=amount,
                issue_date=REFERENCE_START, due_date=REFERENCE_START + dt.timedelta(days=15), status="open",
            ))
            session.commit()

        invoice = session.get(Invoice, invoice_id)
        if not invoice.razorpay_invoice_id:
            created = rzp.create_invoice({
                "type": "invoice",
                "description": f"Promise Integrity Agent live-slice demo -- {invoice_id}",
                "customer": {
                    "name": f"Live Slice Demo Customer {i:02d}",
                    "email": f"live-slice-demo-{i:02d}@example.com",
                    "contact": "9000090000",
                },
                "line_items": [
                    {"name": f"Invoice {invoice_id}", "amount": int(round(amount * 100)), "currency": "INR"},
                ],
                "sms_notify": 0,
                "email_notify": 0,
            })
            invoice.razorpay_invoice_id = created["id"]
            session.commit()
            print(f"  created Razorpay invoice {created['id']} for {invoice_id} (₹{amount:,.0f}) "
                  f"-- short_url: {created.get('short_url')}")
            time.sleep(8)  # avoid tripping the test-mode account's rate limit across the batch
        else:
            print(f"  {invoice_id} already has razorpay_invoice_id={invoice.razorpay_invoice_id} (skipping create)")

    return invoice_ids


def _customer_model_for(customer_id: str, i: int) -> CustomerBehaviorModel:
    rng = np.random.default_rng(seed=1013)
    latent = None
    for j in range(i + 1):
        archetype = ARCHETYPE_NAMES[j % len(ARCHETYPE_NAMES)]
        latent = sample_customer_latent(f"live-slice-{j:02d}", archetype, rng)
    return CustomerBehaviorModel(latent)


def main() -> None:
    rzp = RazorpayClient()  # raises LiveModeKeyError loudly if creds are missing/live -- see integration/razorpay_client.py

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    print("=== Creating / loading live-slice invoices ===")
    invoice_ids = build_or_load_batch(session, rzp)
    live_slice_invoice_ids = set(invoice_ids)

    state = _load_sidecar()
    live_slice_payment_links: dict[str, str] = dict(state["payment_links"])
    start_day = state["next_day"]

    propensity_tool = PropensityModelTool(PropensityModel())
    pending_promise_truth: dict[str, bool] = {}
    draft_tool = LLMDraftTool()  # free Stage-7 stub -- this demo proves the Razorpay wiring, not drafting quality

    def make_ctx(day: int, customer_id: str, i: int) -> AgentContext:
        return AgentContext(
            day=day, reference_start=REFERENCE_START, session=session,
            customer_model=_customer_model_for(customer_id, i), propensity_model_tool=propensity_tool,
            policy_config=CONFIG, pending_promise_truth=pending_promise_truth, draft_tool=draft_tool,
            razorpay_client=rzp, live_slice_invoice_ids=live_slice_invoice_ids,
            live_slice_payment_links=live_slice_payment_links,
        )

    print(f"\n=== Running agent loop: days {start_day}-{start_day + DAYS_PER_RUN - 1} "
          f"over {len(invoice_ids)} live-slice invoices ===")
    for day in range(start_day, start_day + DAYS_PER_RUN):
        for i, invoice_id in enumerate(invoice_ids):
            invoice = session.get(Invoice, invoice_id)
            ctx = make_ctx(day, invoice.customer_id, i)
            run_agent_cycle(invoice_id, ctx)
        session.commit()

    natural_link = any(
        session.query(Communication)
        .filter(Communication.invoice_id == iid, Communication.message_type == "link_resend")
        .count() > 0
        for iid in invoice_ids
    )
    if not live_slice_payment_links and not natural_link:
        target_id = next(iid for iid in invoice_ids if session.get(Invoice, iid).status == "open")
        invoice = session.get(Invoice, target_id)
        as_of = REFERENCE_START + dt.timedelta(days=start_day + DAYS_PER_RUN - 1)
        audit = AuditLogTool(session)
        i = invoice_ids.index(target_id)
        ctx = make_ctx(start_day + DAYS_PER_RUN - 1, invoice.customer_id, i)
        print(f"\n  [no invoice naturally reached link_resend within this window -- "
              f"explicitly triggering it for {target_id}, see module docstring]")
        audit.write(AuditEvent(
            invoice_id=target_id, customer_id=invoice.customer_id, step="decide",
            input_snapshot={"reason": "natural policy sweep did not select link_resend within demo window"},
            model_output=None, decision="link_resend", rationale_code="DEMO_FORCED_LINK_RESEND",
            constraint_triggered=None, executed_action=None, human_approval_required=False,
            timestamp=as_of,
        ))
        _dispatch_live_payment_link(invoice, ctx, audit, as_of)
        session.commit()

    state["payment_links"] = live_slice_payment_links
    state["next_day"] = start_day + DAYS_PER_RUN
    _save_sidecar(state)

    print("\n=== Live-slice status ===")
    for invoice_id in invoice_ids:
        invoice = session.get(Invoice, invoice_id)
        link_id = live_slice_payment_links.get(invoice_id)
        print(f"  {invoice_id}: status={invoice.status}, razorpay_invoice_id={invoice.razorpay_invoice_id}, "
              f"payment_link_id={link_id}")

    pending_links = {
        iid: link_id for iid, link_id in live_slice_payment_links.items()
        if session.get(Invoice, iid).status != "paid"
    }
    if pending_links:
        print("\n=== MANUAL STEP REQUIRED ===")
        print("Open one of these payment links and complete a Razorpay test-mode payment,")
        print("then re-run this script to verify it settles for real:")
        for iid, link_id in pending_links.items():
            link = rzp.fetch_payment_link(link_id)
            print(f"  {iid}: {link.get('short_url')}  (status={link.get('status')})")
    else:
        print("\nAll live-slice invoices with a payment link are already settled.")


if __name__ == "__main__":
    main()
