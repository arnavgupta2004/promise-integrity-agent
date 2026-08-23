"""
backend/routes/dashboard.py — Stage 11 dashboard API (architecture §15):
portfolio risk overview, per-invoice audit-trail drill-down, and the
Stage 6 3-arm comparison. Every endpoint reads real data -- no mocked
responses anywhere.

DASHBOARD_DB_PATH (default: data/audit_completeness_batch.db) is a
separate, independently-configurable engine/session from backend.db's own
default SessionLocal (which points at an empty promise_integrity.db --
nothing has ever run the live agent loop against it). The Stage 9 batch DB
is the fullest real dataset in the repo (100 invoices, ~7,800 audit-log
rows, a full 60-simulated-day run through the actual agent loop), and is
what this dashboard is meant to demo against; pointing at a different DB
for a different demo run is a one-line env var, not a code change.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from backend.db import AuditLog, Customer, Invoice
from backend.rationale_explanations import explain_rationale_codes
from features.feature_engine import compute_prs

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DB_PATH = os.environ.get("DASHBOARD_DB_PATH", str(REPO_ROOT / "data" / "audit_completeness_batch.db"))
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Same low/mid/high PRS bands eval/run_eval.py's own table1b_by_prs_band.csv
# uses, so a risk tier here means the same thing it means in Stage 6's
# results -- just re-labeled: a LOW payment-reliability score is HIGH risk.
RISK_TIERS = [
    ("high_risk", 0.0, 0.4, "High risk (PRS 0.0–0.4)"),
    ("medium_risk", 0.4, 0.6, "Medium risk (PRS 0.4–0.6)"),
    ("low_risk", 0.6, 1.01, "Low risk (PRS 0.6–1.0)"),
]

_engine = create_engine(f"sqlite:///{DASHBOARD_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)

router = APIRouter(prefix="/api")


def _seq(log_id: str) -> int:
    """The reliable chronological ordering key for AuditLog rows,
    established back in Stage 7/8/9: log_id is "audit-{invoice_id}-{step}-{seq}"
    where seq is a process-wide monotonic counter -- log_id string-sorts by
    step name, not by when the event actually happened."""
    return int(log_id.rsplit("-", 1)[-1])


def _risk_tier(prs: float) -> tuple[str, str]:
    for key, lo, hi, label in RISK_TIERS:
        if lo <= prs < hi:
            return key, label
    return RISK_TIERS[-1][0], RISK_TIERS[-1][3]


@router.get("/portfolio")
def portfolio_overview() -> dict:
    """View (1): aggregate ₹ at risk + count by risk tier, computed live
    from the real DB -- invoice amounts and compute_prs's own
    Payment/Promise-history read, not a cached or precomputed figure."""
    session = _Session()
    try:
        as_of_row = session.query(func.max(AuditLog.timestamp)).scalar()
        as_of = as_of_row or dt.datetime.utcnow()

        all_invoices = session.query(Invoice).all()
        if not all_invoices:
            raise HTTPException(status_code=404, detail=f"no invoices found in {DASHBOARD_DB_PATH}")

        unpaid = [inv for inv in all_invoices if inv.status != "paid"]
        paid = [inv for inv in all_invoices if inv.status == "paid"]

        tiers: dict[str, dict] = {key: {"label": label, "count": 0, "amount": 0.0} for key, _, _, label in RISK_TIERS}
        by_status: dict[str, dict] = {}
        for inv in unpaid:
            prs = compute_prs(inv.customer_id, as_of=as_of, session=session)
            tier_key, _ = _risk_tier(prs)
            tiers[tier_key]["count"] += 1
            tiers[tier_key]["amount"] += inv.amount
            by_status.setdefault(inv.status, {"count": 0, "amount": 0.0})
            by_status[inv.status]["count"] += 1
            by_status[inv.status]["amount"] += inv.amount

        return {
            "as_of": as_of.isoformat(),
            "total_invoices": len(all_invoices),
            "unpaid_invoice_count": len(unpaid),
            "total_amount_at_risk": sum(inv.amount for inv in unpaid),
            "total_amount_recovered": sum(inv.amount for inv in paid),
            "paid_invoice_count": len(paid),
            "risk_tiers": [{"key": key, **tiers[key]} for key, _, _, _ in RISK_TIERS],
            "by_status": [{"status": k, **v} for k, v in sorted(by_status.items())],
        }
    finally:
        session.close()


@router.get("/invoices")
def list_invoices() -> list[dict]:
    """Backing list for view (2)'s invoice selector."""
    session = _Session()
    try:
        rows = (
            session.query(Invoice, Customer.archetype)
            .join(Customer, Customer.customer_id == Invoice.customer_id)
            .order_by(Invoice.invoice_id)
            .all()
        )
        return [
            {
                "invoice_id": inv.invoice_id, "customer_id": inv.customer_id, "archetype": archetype,
                "amount": inv.amount, "status": inv.status,
            }
            for inv, archetype in rows
        ]
    finally:
        session.close()


@router.get("/invoices/{invoice_id}/audit-trail")
def invoice_audit_trail(invoice_id: str) -> dict:
    """View (2): the real AuditLog rows for one invoice, in true
    chronological order (via the seq-number key, not log_id string sort),
    each rationale_code translated into plain language -- every code in a
    compound comma-joined trail, not just the first."""
    session = _Session()
    try:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None:
            raise HTTPException(status_code=404, detail=f"unknown invoice_id: {invoice_id}")
        customer = session.get(Customer, invoice.customer_id)

        rows = session.query(AuditLog).filter(AuditLog.invoice_id == invoice_id).all()
        rows.sort(key=lambda r: _seq(r.log_id))

        events = [
            {
                "log_id": r.log_id, "seq": _seq(r.log_id), "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "step": r.step, "decision": r.decision, "executed_action": r.executed_action,
                "rationale_code": r.rationale_code, "rationale_explanations": explain_rationale_codes(r.rationale_code),
                "constraint_triggered": r.constraint_triggered, "human_approval_required": bool(r.human_approval_required),
                "input_snapshot": r.input_snapshot, "model_output": r.model_output,
            }
            for r in rows
        ]
        return {
            "invoice_id": invoice_id, "customer_id": invoice.customer_id,
            "archetype": customer.archetype if customer else None,
            "amount": invoice.amount, "status": invoice.status, "event_count": len(events), "events": events,
        }
    finally:
        session.close()


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"eval result not found: {path.name} (run eval/run_eval.py first)")
    with path.open() as f:
        return list(csv.DictReader(f))


@router.get("/eval/three-arm")
def eval_three_arm_comparison() -> dict:
    """View (3): Stage 6's real 3-arm eval output, read directly from
    eval/results/ -- never regenerated or mocked here."""
    rows = _read_csv(EVAL_RESULTS_DIR / "table1_three_arm_comparison.csv")
    arms = []
    for row in rows:
        arms.append({
            "arm": row["Arm"],
            "total_recovered": float(row["Total ₹ Recovered"]),
            "cost": float(row["Cost"]),
            "net_recovery": float(row["Net Recovery"]),
            "incremental_vs_no_intervention": float(row["Incremental vs. No-Intervention"]),
            "incremental_vs_naive": float(row["Incremental vs. Naive"]),
        })
    return {"arms": arms}


@router.get("/eval/by-prs-band")
def eval_by_prs_band() -> dict:
    rows = _read_csv(EVAL_RESULTS_DIR / "table1b_by_prs_band.csv")
    bands = []
    for row in rows:
        bands.append({
            "prs_band": row["PRS Band"], "n_customers": int(row["n_customers"]), "arm": row["Arm"],
            "net_recovery": float(row["Net Recovery"]),
        })
    return {"bands": bands}
