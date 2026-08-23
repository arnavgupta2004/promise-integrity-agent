from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import datetime as dt


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    archetype = Column(String, nullable=False)          # ground-truth label, hidden from agent
    segment = Column(String)
    credit_terms_days = Column(Integer, default=30)
    onboarding_date = Column(DateTime)
    razorpay_customer_id = Column(String, nullable=True)  # set only for live-slice customers


class Invoice(Base):
    __tablename__ = "invoices"
    invoice_id = Column(String, primary_key=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"))
    amount = Column(Float, nullable=False)
    issue_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    status = Column(String, default="open")             # open | paid | partially_paid | disputed | written_off
    dispute_flag = Column(Boolean, default=False)
    dispute_reason = Column(String, nullable=True)
    razorpay_invoice_id = Column(String, nullable=True)


class Payment(Base):
    __tablename__ = "payments"
    payment_id = Column(String, primary_key=True)
    invoice_id = Column(String, ForeignKey("invoices.invoice_id"))
    amount_paid = Column(Float, nullable=False)
    payment_date = Column(DateTime, nullable=False)
    partial_flag = Column(Boolean, default=False)
    razorpay_payment_id = Column(String, nullable=True)


class Communication(Base):
    __tablename__ = "communications"
    comm_id = Column(String, primary_key=True)
    invoice_id = Column(String, ForeignKey("invoices.invoice_id"))
    channel = Column(String)                             # email | sms | whatsapp
    timestamp = Column(DateTime)
    message_type = Column(String)                        # soft_reminder | firm_reminder | plan_proposal | escalation_notice
    message_text = Column(Text)
    dispatched_by = Column(String)                        # "agent" | "human"
    response_received = Column(Boolean, default=False)
    response_text = Column(Text, nullable=True)


class Promise(Base):
    __tablename__ = "promises"
    promise_id = Column(String, primary_key=True)
    invoice_id = Column(String, ForeignKey("invoices.invoice_id"))
    promised_date = Column(DateTime)
    promised_amount = Column(Float)
    made_on = Column(DateTime)
    extraction_confidence = Column(Float)
    kept = Column(Boolean, nullable=True)                # null until resolved
    broken_reason = Column(String, nullable=True)


class Dispute(Base):
    __tablename__ = "disputes"
    dispute_id = Column(String, primary_key=True)
    invoice_id = Column(String, ForeignKey("invoices.invoice_id"))
    raised_date = Column(DateTime)
    reason = Column(String)
    resolved = Column(Boolean, default=False)
    resolution_date = Column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    log_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    invoice_id = Column(String)
    customer_id = Column(String)
    step = Column(String)                # detect|diagnose|decide|act|verify|reassess
    input_snapshot = Column(JSON)
    model_output = Column(JSON, nullable=True)
    decision = Column(String, nullable=True)
    rationale_code = Column(String, nullable=True)
    constraint_triggered = Column(String, nullable=True)
    executed_action = Column(String, nullable=True)
    human_approval_required = Column(Boolean, default=False)


class ModelRun(Base):
    __tablename__ = "model_runs"
    run_id = Column(String, primary_key=True)
    model_version = Column(String)
    trained_on = Column(DateTime)
    policy_config_version = Column(String)


DATABASE_URL = "sqlite:///./promise_integrity.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_all_tables() -> None:
    Base.metadata.create_all(bind=engine)
