"""
tests/test_feature_engine.py — unit tests for features/feature_engine.py
against a fresh in-memory SQLite DB per test (never touches the real
promise_integrity.db file).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base, Communication, Customer, Dispute, Invoice, Payment, Promise
from features.feature_engine import (
    FEATURE_COLUMNS,
    NO_CONTACT_SENTINEL_DAYS,
    RELATIVE_LATENESS_CAP,
    build_feature_vector,
    compute_prs,
)

AS_OF = dt.datetime(2026, 1, 1)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def make_customer(session, customer_id, segment="enterprise", credit_terms_days=30):
    c = Customer(
        customer_id=customer_id,
        name=f"Test Co ({customer_id})",
        archetype="reliable_always_late",
        segment=segment,
        credit_terms_days=credit_terms_days,
        onboarding_date=AS_OF - dt.timedelta(days=365),
    )
    session.add(c)
    session.commit()
    return c


def make_invoice(session, invoice_id, customer_id, amount=1000.0, issue_date=None, due_date=None):
    issue_date = issue_date or (AS_OF - dt.timedelta(days=40))
    due_date = due_date or (AS_OF - dt.timedelta(days=10))
    inv = Invoice(
        invoice_id=invoice_id, customer_id=customer_id, amount=amount,
        issue_date=issue_date, due_date=due_date, status="open",
    )
    session.add(inv)
    session.commit()
    return inv


def make_promise(session, promise_id, invoice_id, made_on, kept, promised_date=None):
    p = Promise(
        promise_id=promise_id, invoice_id=invoice_id,
        promised_date=promised_date or (made_on + dt.timedelta(days=5)),
        promised_amount=1000.0, made_on=made_on,
        extraction_confidence=0.9, kept=kept,
    )
    session.add(p)
    session.commit()
    return p


class TestNoHistoryDefaults:
    def test_build_feature_vector_returns_all_columns_without_error(self, session):
        make_customer(session, "cust_new")
        make_invoice(session, "inv_new", "cust_new")

        features = build_feature_vector("inv_new", "soft_reminder", as_of=AS_OF, session=session)

        assert set(features.keys()) == set(FEATURE_COLUMNS)

    def test_no_history_produces_sensible_neutral_defaults(self, session):
        make_customer(session, "cust_new")
        make_invoice(session, "inv_new", "cust_new")

        features = build_feature_vector("inv_new", "soft_reminder", as_of=AS_OF, session=session)

        # PRS: no promise/payment/contact history -> those three components
        # default neutral (0.5), but the dispute component legitimately sees
        # 1 invoice with 0 disputes (real, if thin, evidence of a clean
        # record) and reports 1.0 there, not 0.5 -- so the blended score
        # lands a bit above neutral, not exactly at it. Not extreme either way.
        assert 0.5 <= features["prs_score"] <= 0.65
        assert features["prs_trend"] == pytest.approx(0.0, abs=0.01)
        # observed-frequency columns: 0.0 when there's no denominator
        assert features["dispute_rate"] == 0.0
        assert features["response_rate"] == 0.0
        assert features["partial_payment_rate"] == 0.0
        # no promise ever made on this invoice
        assert features["active_promise_flag"] is False
        assert features["days_until_promised_date"] == -1
        assert features["broken_promise_streak"] == 0
        # never contacted
        assert features["days_since_last_contact"] == NO_CONTACT_SENTINEL_DAYS
        # not enough invoices to form quantiles -> neutral middle tier
        assert features["amount_tier"] == 2
        assert features["segment"] == "enterprise"
        assert features["intervention_type"] == "soft_reminder"

    def test_no_history_customer_compute_prs_directly(self, session):
        make_customer(session, "cust_blank")
        assert compute_prs("cust_blank", as_of=AS_OF, session=session) == pytest.approx(0.5, abs=0.01)


class TestPRSBrokenVsKeptPromise:
    def test_recent_broken_promise_shows_visibly_lower_prs(self, session):
        make_customer(session, "cust_kept")
        make_customer(session, "cust_broken")
        make_invoice(session, "inv_kept", "cust_kept")
        make_invoice(session, "inv_broken", "cust_broken")

        # otherwise identical: one resolved promise each, same recency, same amount
        make_promise(session, "p_kept", "inv_kept", made_on=AS_OF - dt.timedelta(days=10), kept=True)
        make_promise(session, "p_broken", "inv_broken", made_on=AS_OF - dt.timedelta(days=10), kept=False)

        prs_kept = compute_prs("cust_kept", as_of=AS_OF, session=session)
        prs_broken = compute_prs("cust_broken", as_of=AS_OF, session=session)

        assert prs_kept > prs_broken
        # visibly lower, not just numerically different
        assert prs_kept - prs_broken > 0.1


class TestRelativeLatenessCap:
    def test_relative_lateness_capped_at_3(self, session):
        make_customer(session, "cust_verylate", credit_terms_days=30)
        # no payment history -> avg_days_to_pay falls back to credit_terms_days=30
        # due 200 days before as_of -> raw ratio ~6.7, must clip to 3.0
        make_invoice(session, "inv_verylate", "cust_verylate", due_date=AS_OF - dt.timedelta(days=200))

        features = build_feature_vector("inv_verylate", "none", as_of=AS_OF, session=session)

        assert features["relative_lateness"] == RELATIVE_LATENESS_CAP

    def test_relative_lateness_uncapped_case_stays_below_cap(self, session):
        make_customer(session, "cust_mildlate", credit_terms_days=30)
        # due 10 days before as_of -> raw ratio ~0.33, well under the cap
        make_invoice(session, "inv_mildlate", "cust_mildlate", due_date=AS_OF - dt.timedelta(days=10))

        features = build_feature_vector("inv_mildlate", "none", as_of=AS_OF, session=session)

        assert 0.0 < features["relative_lateness"] < RELATIVE_LATENESS_CAP
