"""
Manual PRS trace (Stage 2 DoD): constructs three synthetic customer
histories -- reliable, serial-promiser-like, and model-citizen-like -- in
an in-memory DB and prints compute_prs()'s score plus its four component
breakdown for each, to sanity-check the scores come out intuitively
separated rather than just numerically different.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base, Communication, Customer, Dispute, Invoice, Payment, Promise
from features.feature_engine import (
    _dispute_component,
    _lateness_trend_component,
    _recency_weighted_keep_rate,
    _response_component,
    compute_prs,
)

AS_OF = dt.datetime(2026, 1, 1)


def build_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def add_customer(session, customer_id, segment="enterprise", credit_terms_days=30):
    session.add(Customer(
        customer_id=customer_id, name=customer_id, archetype="n/a", segment=segment,
        credit_terms_days=credit_terms_days, onboarding_date=AS_OF - dt.timedelta(days=365),
    ))


def add_invoice_cycle(session, customer_id, cycle_idx, issue_days_ago, days_to_pay, credit_terms_days=30,
                       promise_made=True, promise_kept=None, disputed=False,
                       n_contacts=1, n_responses=1, partial=False):
    invoice_id = f"inv-{customer_id}-{cycle_idx}"
    issue_date = AS_OF - dt.timedelta(days=issue_days_ago)
    due_date = issue_date + dt.timedelta(days=credit_terms_days)
    payment_date = issue_date + dt.timedelta(days=days_to_pay)

    session.add(Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=1000.0,
        issue_date=issue_date, due_date=due_date, status="paid", dispute_flag=disputed,
    ))
    session.add(Payment(
        payment_id=f"pay-{invoice_id}", invoice_id=invoice_id,
        amount_paid=1000.0, payment_date=payment_date, partial_flag=partial,
    ))
    if disputed:
        session.add(Dispute(
            dispute_id=f"disp-{invoice_id}", invoice_id=invoice_id,
            raised_date=issue_date + dt.timedelta(days=5), reason="billing_disagreement", resolved=True,
            resolution_date=payment_date,
        ))
    for i in range(n_contacts):
        session.add(Communication(
            comm_id=f"comm-{invoice_id}-{i}", invoice_id=invoice_id, channel="email",
            timestamp=issue_date + dt.timedelta(days=due_date.day % 5 + i * 3),
            message_type="soft_reminder", message_text="reminder", dispatched_by="agent",
            response_received=(i < n_responses),
        ))
    if promise_made:
        session.add(Promise(
            promise_id=f"prom-{invoice_id}", invoice_id=invoice_id,
            promised_date=due_date + dt.timedelta(days=3), promised_amount=1000.0,
            made_on=due_date - dt.timedelta(days=2), extraction_confidence=0.9,
            kept=promise_kept,
        ))


def build_reliable_customer(session):
    cid = "cust_reliable"
    add_customer(session, cid)
    # 6 invoice cycles over the past ~6 months, always pays a bit late but
    # consistently, keeps 5/6 promises (1 broken far in the past), responds well
    kept_pattern = [True, True, True, False, True, True]  # oldest -> newest
    for i, kept in enumerate(kept_pattern):
        cycles_ago = len(kept_pattern) - i
        add_invoice_cycle(
            session, cid, i, issue_days_ago=cycles_ago * 40, days_to_pay=38,
            promise_kept=kept, disputed=False, n_contacts=2, n_responses=2,
        )
    session.commit()
    return cid


def build_serial_promiser_customer(session):
    cid = "cust_serial_promiser"
    add_customer(session, cid)
    # 6 invoice cycles, responds to every contact (readily) but breaks most
    # promises, pays slower and slower, one dispute
    kept_pattern = [True, False, False, False, False, False]  # oldest -> newest
    for i, kept in enumerate(kept_pattern):
        cycles_ago = len(kept_pattern) - i
        add_invoice_cycle(
            session, cid, i, issue_days_ago=cycles_ago * 45, days_to_pay=30 + i * 6,
            promise_kept=kept, disputed=(i == 3), n_contacts=3, n_responses=3,
        )
    session.commit()
    return cid


def build_model_citizen_customer(session):
    cid = "cust_model_citizen"
    add_customer(session, cid)
    # 6 invoice cycles, always pays early, always keeps promises, no disputes
    for i in range(6):
        cycles_ago = 6 - i
        add_invoice_cycle(
            session, cid, i, issue_days_ago=cycles_ago * 35, days_to_pay=24,
            promise_kept=True, disputed=False, n_contacts=1, n_responses=1,
        )
    session.commit()
    return cid


def trace(session, customer_id, label):
    keep = _recency_weighted_keep_rate(session, customer_id, AS_OF)
    trend = _lateness_trend_component(session, customer_id, AS_OF)
    dispute = _dispute_component(session, customer_id, AS_OF)
    response = _response_component(session, customer_id, AS_OF)
    prs = compute_prs(customer_id, as_of=AS_OF, session=session)

    print(f"{label} ({customer_id})")
    print(f"  keep_component     = {keep:.3f}  (weight 0.45)")
    print(f"  trend_component    = {trend:.3f}  (weight 0.25)")
    print(f"  dispute_component  = {dispute:.3f}  (weight 0.15)")
    print(f"  response_component = {response:.3f}  (weight 0.15)")
    print(f"  PRS                = {prs:.3f}")
    print()
    return prs


def main():
    session = build_session()
    cid_reliable = build_reliable_customer(session)
    cid_serial = build_serial_promiser_customer(session)
    cid_model = build_model_citizen_customer(session)

    print(f"Manual PRS trace, as_of={AS_OF.date()}\n")
    prs_reliable = trace(session, cid_reliable, "Reliable (5/6 promises kept, 0 disputes, responsive)")
    prs_serial = trace(session, cid_serial, "Serial-promiser-like (1/6 promises kept, responds readily, worsening lateness)")
    prs_model = trace(session, cid_model, "Model-citizen-like (6/6 promises kept, 0 disputes, always early)")

    print("Summary:")
    print(f"  model_citizen ({prs_model:.3f}) > reliable ({prs_reliable:.3f}) > serial_promiser ({prs_serial:.3f})")
    assert prs_model > prs_reliable > prs_serial, "PRS values did not separate as expected"
    assert prs_model - prs_serial > 0.3, "top vs bottom PRS spread too small to be a meaningful signal"
    print("\nPRS values are correctly ordered and clearly separated.")


if __name__ == "__main__":
    main()
