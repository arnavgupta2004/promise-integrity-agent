"""
SimulationEngine — day-step driver (contract §3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from simulator.behavior_model import Action, CustomerBehaviorModel, PotentialOutcome

# CustomerLatentState carries no per-customer credit-terms field, so a
# single shared NET-30 term is used for every invoice the simulator issues.
DEFAULT_CREDIT_TERMS_DAYS = 30


@dataclass
class SimInvoice:
    """Lightweight invoice representation local to the simulator, so
    `simulator` stays a leaf module with no dependency on `backend.db`
    (per the Module interfaces section of the contract)."""
    invoice_id: str
    customer_id: str
    issue_day: int
    due_day: int
    status: str = "open"          # open | paid
    paid_day: Optional[int] = None
    last_contact_day: Optional[int] = None
    total_contacts: int = 0


@dataclass
class RealizedRecord:
    day: int
    customer_id: str
    archetype: str
    invoice_id: str
    action: Action
    outcome: PotentialOutcome


@dataclass
class PotentialRecord:
    day: int
    invoice_id: str
    customer_id: str
    outcomes: dict[Action, PotentialOutcome]


@dataclass
class SimulationResult:
    """§3 only forward-references this type's name without specifying its
    fields; this is the concrete shape chosen to carry both the realized
    rollout and, when eval_mode is on, the full potential-outcome
    counterfactuals the eval harness needs -- without re-running the sim.
    """
    n_days: int
    realized_log: list[RealizedRecord]
    potential_log: list[PotentialRecord]   # empty if eval_mode=False
    invoices: list[SimInvoice]
    customers: dict[str, CustomerBehaviorModel]


# invoice, day, context -> chosen action. Deliberately excludes the
# CustomerBehaviorModel: §3 states the model (and its latent archetype) is
# "Never exposed to the agent/policy — only to the simulator engine and the
# eval harness", so the policy only ever sees the invoice + derived context.
PolicyFn = Callable[[SimInvoice, int, dict], Action]


class SimulationEngine:
    """Day-step driver. Advances all customers/invoices by one day,
    invokes the policy-under-test for due invoices, records realized
    outcomes, and (if in eval mode) also records full potential outcomes."""

    def __init__(self, customers: list[CustomerBehaviorModel], seed: int, eval_mode: bool = True):
        # `eval_mode` is an additive keyword (default True) beyond the two
        # positional args in §3's frozen __init__ signature -- needed to
        # implement the class docstring's documented "(if in eval mode)
        # also records full potential outcomes" behavior without changing
        # run()'s signature, which the contract does give exactly.
        self.customers: dict[str, CustomerBehaviorModel] = {c.latent.customer_id: c for c in customers}
        self.seed = seed
        self.eval_mode = eval_mode
        random.seed(seed)
        np.random.seed(seed)

        # One invoice per customer, issued on day 0. §3's __init__ takes no
        # separate invoice list, so the engine generates the invoice
        # population itself from the given customers.
        self.invoices: list[SimInvoice] = [
            SimInvoice(
                invoice_id=f"inv-{c.latent.customer_id}-0",
                customer_id=c.latent.customer_id,
                issue_day=0,
                due_day=DEFAULT_CREDIT_TERMS_DAYS,
            )
            for c in customers
        ]

        self.realized_log: list[RealizedRecord] = []
        self.potential_log: list[PotentialRecord] = []

    def _build_context(self, invoice: SimInvoice, day: int) -> dict:
        return {
            "issue_day": invoice.issue_day,
            "due_day": invoice.due_day,
            "days_since_last_contact": (
                day - invoice.last_contact_day if invoice.last_contact_day is not None else float("inf")
            ),
            "total_contacts": invoice.total_contacts,
        }

    def step(self, day: int, policy_fn: PolicyFn) -> None:
        # Snapshot with list(...): closing an invoice below can append its
        # customer's next billing-cycle invoice to self.invoices, and that
        # new invoice must not be processed within this same day's pass.
        for invoice in list(self.invoices):
            if invoice.status == "paid" or day < invoice.issue_day:
                continue

            customer = self.customers[invoice.customer_id]
            context = self._build_context(invoice, day)
            potential = customer.generate_potential_outcomes(invoice.invoice_id, day, context)

            if self.eval_mode:
                self.potential_log.append(PotentialRecord(
                    day=day, invoice_id=invoice.invoice_id,
                    customer_id=invoice.customer_id, outcomes=potential,
                ))

            action = policy_fn(invoice, day, context)
            outcome = customer.realize(action, day)

            self.realized_log.append(RealizedRecord(
                day=day, customer_id=invoice.customer_id, archetype=customer.latent.archetype,
                invoice_id=invoice.invoice_id, action=action, outcome=outcome,
            ))

            if action != "none":
                invoice.last_contact_day = day
                invoice.total_contacts += 1

            if outcome.will_pay_within_N:
                invoice.status = "paid"
                invoice.paid_day = day + (outcome.days_to_pay or 0)
                self._issue_next_cycle(invoice)

    def _issue_next_cycle(self, paid_invoice: SimInvoice) -> None:
        """Start the customer's next billing cycle the day after this
        invoice closes. §3's __init__ only takes a customer list (no
        recurring-billing concept), so this is an added interpretation: a
        single invoice per customer over a 90-day window closes, for most
        archetypes, well before day 45 (see §4's avg_days_to_pay values),
        which would leave no invoice open to observe behavior in the back
        half of the window at all -- including the "declining reliability"
        trend §4 specifies for the degrading archetype. Recurring
        NET-30-style billing (a new invoice each time the last one is paid)
        is what makes a 90-day simulated window meaningful to observe.
        """
        cycle = sum(1 for i in self.invoices if i.customer_id == paid_invoice.customer_id)
        next_issue_day = paid_invoice.paid_day + 1
        self.invoices.append(SimInvoice(
            invoice_id=f"inv-{paid_invoice.customer_id}-{cycle}",
            customer_id=paid_invoice.customer_id,
            issue_day=next_issue_day,
            due_day=next_issue_day + DEFAULT_CREDIT_TERMS_DAYS,
        ))

    def run(self, n_days: int, policy_fn: PolicyFn) -> SimulationResult:
        for day in range(n_days):
            self.step(day, policy_fn)
        return SimulationResult(
            n_days=n_days,
            realized_log=self.realized_log,
            potential_log=self.potential_log,
            invoices=self.invoices,
            customers=self.customers,
        )
