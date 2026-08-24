# Promise Integrity Agent — Implementation Contract
### Frozen post-architecture spec. No further architecture discussion after this document.

---

## 1. Dependencies

```
python==3.11
fastapi==0.115.*
uvicorn==0.32.*
sqlalchemy==2.0.*
pydantic==2.9.*
lightgbm==4.5.*
scikit-learn==1.5.*
pandas==2.2.*
numpy==1.26.*
razorpay==1.4.*          # official Python SDK
anthropic==0.39.*        # LLM calls (structured output via tool-use/JSON schema)
pytest==8.3.*
matplotlib==3.9.*         # eval harness charts (simple, no frontend dep needed for these)
python-dotenv==1.0.*
```

DB: **SQLite** via SQLAlchemy (file-based, zero setup, sufficient for demo scale). No ORM migrations framework needed at this scale — `Base.metadata.create_all()` on startup is enough.

---

## 2. Database schema (SQLAlchemy models)

```python
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship
import datetime as dt

class Base(DeclarativeBase): pass

class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    archetype = Column(String, nullable=False)          # ground-truth label, hidden from agent
    segment = Column(String)
    credit_terms_days = Column(Integer, default=30)
    onboarding_date = Column(DateTime)
    razorpay_customer_id = Column(String, nullable=True) # set only for live-slice customers

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
```

---

## 3. Simulator interfaces

```python
from dataclasses import dataclass
from typing import Literal, Optional

Action = Literal[
    "none", "soft_reminder", "firm_reminder", "channel_escalation",
    "link_resend", "plan_proposal", "human_escalation"
]

@dataclass
class CustomerLatentState:
    customer_id: str
    archetype: str
    keep_probability_base: float       # promise-keep prob
    avg_days_to_pay: float
    dispute_propensity: float
    response_propensity: float
    fatigue_sensitivity: float         # how much repeated contact reduces response
    trend_slope: float                 # drift in reliability over simulated time

@dataclass
class PotentialOutcome:
    invoice_id: str
    action: Action
    will_pay_within_N: bool
    days_to_pay: Optional[int]
    will_respond: bool
    will_promise: bool
    promise_kept: Optional[bool]

class CustomerBehaviorModel:
    """One instance per customer. Owns the latent state and generates
    potential outcomes. Never exposed to the agent/policy — only to
    the simulator engine and the eval harness."""
    def __init__(self, latent: CustomerLatentState): ...
    def generate_potential_outcomes(
        self, invoice_id: str, day: int, context: dict
    ) -> dict[Action, PotentialOutcome]: ...
    def realize(self, action: Action, day: int) -> PotentialOutcome:
        """Returns the single realized outcome for the action actually taken.
        Used during rollout; generate_potential_outcomes() is used by the
        eval harness to reveal counterfactuals after the fact."""

class SimulationEngine:
    """Day-step driver. Advances all customers/invoices by one day,
    invokes the policy-under-test for due invoices, records realized
    outcomes, and (if in eval mode) also records full potential outcomes."""
    def __init__(self, customers: list[CustomerBehaviorModel], seed: int): ...
    def step(self, day: int, policy_fn) -> None: ...
    def run(self, n_days: int, policy_fn) -> "SimulationResult": ...
```

---

## 4. Seven customer archetypes — exact latent parameters

| Archetype | `keep_probability_base` | `avg_days_to_pay` (vs NET-30) | `dispute_propensity` | `response_propensity` | `fatigue_sensitivity` | `trend_slope` |
|---|---|---|---|---|---|---|
| Reliable-always-late | 0.85 | 40 (fixed, low variance) | 0.02 | 0.7 | low | 0.0 |
| Cash-flow-strained-genuine | 0.65 | 35–55 (high variance) | 0.05 | 0.6 | medium | 0.0 |
| Serial-promiser | 0.20 | 45–70 | 0.03 | 0.8 (responds readily, doesn't pay) | low | 0.0 |
| Disputer | 0.55 | 40 (excl. dispute periods) | 0.35 | 0.5 | medium | 0.0 |
| Non-responsive | 0.40 | 50–90 | 0.02 | 0.15 | high | 0.0 |
| Model-citizen | 0.95 | 28 (before due date) | 0.01 | 0.8 | low | 0.0 |
| Degrading | 0.80 → declines | 30 → increases | 0.05 | 0.6 | medium | **negative, e.g. −0.01/day** |

All per-customer instances draw from a Beta or Normal distribution centered on these base values (not fixed constants) so within-archetype variation exists, seeded for reproducibility.

---

## 5. Potential-outcome generation mechanism

For a given `(customer, invoice, day, action)`:

```
base_pay_prob = f(days_overdue relative to avg_days_to_pay, archetype base rate)

action_effect:
  none              → 0
  soft_reminder      → + small positive shift, scaled by response_propensity
  firm_reminder       → + larger positive shift, but − fatigue_penalty if contacted recently
  channel_escalation   → + response_propensity boost (reaches non-responders better)
  link_resend          → + small positive, near-zero cost/friction removal effect
  plan_proposal         → + large positive IF keep_probability_base > threshold,
                           ELSE negative (serial-promisers exploit plans, don't keep them)
  human_escalation       → + largest positive but simulates real cost, used for eval only
                           (agent doesn't need a "pay probability" for this — it's a backstop)

final_pay_prob = clip(base_pay_prob + action_effect + trend_slope * day, 0, 1)

promise_prob = f(response_propensity, action) 
promise_kept = Bernoulli(keep_probability_base), independent draw, adjusted by trend
```

`will_pay_within_N` = Bernoulli(final_pay_prob). This function is called once per action to populate `generate_potential_outcomes()` (all actions) for eval, and once for the realized action during rollout.

---

## 6. Feature vector (S-learner input)

```python
FEATURE_COLUMNS = [
    "relative_lateness",           # float, days_overdue / avg_days_to_pay, capped at 3.0
    "prs_score",                   # float [0,1]
    "prs_trend",                   # float, slope over last 90 sim-days
    "dispute_rate",                # float, lifetime
    "response_rate",               # float, last 5 contacts
    "partial_payment_rate",        # float, lifetime
    "amount_tier",                 # int 0-4, quantile bucket relative to customer's own history
    "days_since_last_contact",     # int
    "active_promise_flag",         # bool
    "days_until_promised_date",    # int, -1 if no active promise
    "broken_promise_streak",       # int, consecutive most-recent
    "segment",                     # categorical, one-hot or LightGBM native categorical
    "intervention_type",           # categorical: the treatment variable itself
]
TARGET = "paid_within_N"           # bool, N fixed per risk tier (7 or 21 days)
```

`segment` and `intervention_type` passed as LightGBM native categorical features (`categorical_feature` param), not manually one-hot encoded — avoids sparsity blowup.

---

## 7. Training target and intervention encoding

- One model, `GBMClassifier` (LightGBM), target = `paid_within_N` (binary)
- `intervention_type` is a feature column with values `{none, soft_reminder, firm_reminder, channel_escalation, link_resend, plan_proposal}` (`human_escalation` excluded from model — not a "pay probability" decision, always policy-forced)
- **Training data generation**: run the simulator with a **logging policy** that assigns actions with controlled randomization (e.g., 70% follows a reasonable heuristic, 30% random across eligible actions) — this is what prevents the confounding problem from §2b of the architecture. Without this randomization, the model would only ever see "risky customers get aggressive actions," making the intervention-effect uninterpretable.
- At inference: score the same feature vector once per candidate action (varying only `intervention_type`), producing `P(pay | action=a)` for each `a` — this is what EIV consumes.

---

## 8. EIV / action-selection algorithm

```python
def select_action(invoice, customer, model, policy_constraints) -> Action:
    # Step 1: hard constraints first (see §9)
    forced = check_hard_constraints(invoice, customer, policy_constraints)
    if forced is not None:
        return forced   # e.g. "human_escalation", "none" (blocked), etc.

    # Step 2: EIV ranking over eligible actions only
    eligible = get_eligible_actions(invoice, customer, policy_constraints)
    best_action, best_eiv = "none", 0.0
    for action in eligible:
        features = build_feature_vector(invoice, customer, intervention_type=action)
        p = model.predict_proba(features)          # P(pay | action)
        p_none = model.predict_proba(
            build_feature_vector(invoice, customer, intervention_type="none")
        )
        eiv = invoice.amount * (p - p_none) - ACTION_COST[action]
        if eiv > best_eiv:
            best_action, best_eiv = action, eiv
    return best_action
```

`ACTION_COST` is a fixed cost table (e.g., soft_reminder=₹5, firm_reminder=₹15, channel_escalation=₹20, link_resend=₹5, plan_proposal=₹100 [human review time], human_escalation=₹300).

---

## 9. Policy engine — exact rules, priority order

Checked **in this order**; first match wins and short-circuits EIV:

1. `dispute_flag == True and not resolved` → **force `human_escalation`**, rationale `DISPUTE_UNRESOLVED`
2. Customer has `no_contact_requested == True` → **force `none`**, rationale `NO_CONTACT_HONORED` (permanent, never overridden)
3. `active_promise_flag == True and days_until_promised_date > -grace_period` → **force `none`**, rationale `COOLING_PERIOD_ACTIVE`
4. `broken_promise_streak >= 2` → **force `human_escalation`**, rationale `PROMISE_STREAK_EXCEEDED`
5. `contacts_in_last_3_days >= 1` → **remove all contact actions from eligible set**, rationale `FREQUENCY_CAP` (falls through to EIV over remaining eligible = `{none}` effectively, unless link_resend is exempted — link_resend is exempted from the frequency cap since it's low-friction)
6. `total_automated_contacts_this_invoice >= 4` → **force `human_escalation`**, rationale `MAX_ATTEMPTS_REACHED`
7. `invoice.amount >= HIGH_VALUE_THRESHOLD` → **remove `plan_proposal` and `human_escalation`-bypass from EIV auto-execution; require `human_approval_required=True` flag on any action beyond reminders**, rationale `HIGH_VALUE_REQUIRES_APPROVAL`
8. `plan_proposal` eligible only if `prs_score >= PLAN_ELIGIBILITY_FLOOR` (e.g., 0.5) — else removed from eligible set, rationale `PRS_BELOW_PLAN_FLOOR`
9. If none of the above fire → proceed to EIV ranking (§8) over the remaining eligible set

---

## 10. Agent state machine and tool interfaces

```python
class InvoiceAgentState:
    invoice_id: str
    phase: Literal["detect","diagnose","decide","act","verify","reassess","closed"]
    next_reassess_at: Optional[dt.datetime]

class Tool(Protocol):
    name: str
    def call(self, **kwargs) -> dict: ...

# Tools registered with the orchestration loop:
# - FeatureEngineTool: build_feature_vector(invoice_id) -> dict
# - PropensityModelTool: score(features, action) -> float
# - PolicyEngineTool: authorize(invoice_id, candidate_action) -> AuthDecision
# - LLMDraftTool: draft_message(action_type, context) -> str
# - LLMExtractTool: extract_promise(reply_text) -> PromiseExtraction | None
# - RazorpayTool: create_invoice / resend_link / fetch_payment (live slice only)
# - AuditLogTool: write(event: AuditEvent) -> None

def run_agent_cycle(invoice_id: str) -> None:
    state = load_state(invoice_id)
    features = FeatureEngineTool.call(invoice_id=invoice_id)               # detect+diagnose
    candidate = select_action(features, PropensityModelTool, ...)          # decide (EIV)
    auth = PolicyEngineTool.call(invoice_id=invoice_id, candidate=candidate)  # decide (gate)
    if auth.approved:
        if auth.action in CONTACT_ACTIONS:
            msg = LLMDraftTool.call(action_type=auth.action, context=features)
            dispatch(msg)                                                  # act
        AuditLogTool.call(...)
    schedule_reassess(invoice_id, auth.action)                             # reassess
```

State persists in the `audit_log` + invoice `status`/promise tables — no separate in-memory agent state store needed at this scale; the DB *is* the state machine's memory.

---

## 11. LLM promise-extraction structured schema

```json
{
  "type": "object",
  "properties": {
    "commitment_detected": {"type": "boolean"},
    "promised_date": {"type": ["string", "null"], "format": "date"},
    "promised_amount": {"type": ["number", "null"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "notes": {"type": "string"}
  },
  "required": ["commitment_detected", "confidence"]
}
```
Rule: if `confidence < 0.6`, force `commitment_detected = false` regardless of model output, and route to human review queue rather than creating a `Promise` row — this is the concrete implementation of failure case #8.

---

## 12. Audit-log event schema

```python
@dataclass
class AuditEvent:
    invoice_id: str
    customer_id: str
    step: Literal["detect","diagnose","decide","act","verify","reassess"]
    input_snapshot: dict        # feature vector at decision time
    model_output: Optional[dict]   # {action: prob} for each candidate
    decision: Optional[str]        # chosen action
    rationale_code: Optional[str]  # e.g. "DISPUTE_UNRESOLVED", "EIV_MAX"
    constraint_triggered: Optional[str]
    executed_action: Optional[str]
    human_approval_required: bool
```
Every `run_agent_cycle()` invocation writes at least one `AuditEvent` per phase it passes through — partial cycles (e.g., stopped at policy gate) still log up to the point of stopping.

---

## 13. Razorpay live-slice API calls (exact)

- `client.invoice.create({...})` — create the 5–10 demo invoices as real Razorpay Invoice entities
- `client.invoice.fetch(invoice_id)` — pull status for the verify step
- `client.payment_link.create({...})` — used by the `link_resend` action for live-slice invoices
- `client.payment_link.fetch(id)` — check payment status
- `client.payment.fetch(payment_id)` — confirm captured amount for verification
- All calls use **test-mode API keys** from `.env`, official `razorpay` Python SDK, no manual HTTP needed

---

## 14. Evaluation metrics and outputs

**Table 1 — 3-arm comparison** (columns: Arm | Total ₹ Recovered | Cost | Net Recovery | Incremental vs. No-Intervention | Incremental vs. Naive)
Rows: No-Intervention, Naive-Uniform, Promise Integrity Agent.

**Table 2 — Escalation precision**: of invoices the Agent arm escalated to human, % where `true_would_have_paid_without_further_automation == False` (ground truth from potential outcomes) — precision of escalation decisions.

**Chart 1**: bar chart, net recovery per arm (this is the single most important visual for the demo).

**Chart 2**: PRS trajectory for 2–3 illustrative customers over simulated time, annotated with promise-kept/broken events (shows the reliability feedback loop visually).

**Table 3**: failure-injection scenario pass/fail table (scenario name → expected behavior → observed behavior → pass/fail).

All generated by `eval/run_eval.py`, outputting to `eval/results/` as CSV + PNG, re-runnable on demand.

---

## 15. Repository structure (final)

```
promise-integrity-agent/
├── simulator/
│   ├── archetypes.py         # §4 parameter table as data
│   ├── behavior_model.py     # CustomerBehaviorModel, potential outcomes (§3, §5)
│   └── engine.py              # SimulationEngine (§3)
├── features/
│   └── feature_engine.py      # §6 feature vector construction, PRS calculation
├── models/
│   ├── train.py                # trains the S-learner GBM
│   ├── propensity_model.py      # load/predict wrapper
│   └── artifacts/                # saved model files
├── policy/
│   ├── constraints.py             # §9 hard rules, in priority order
│   └── eiv.py                      # §8 EIV ranking
├── agent/
│   ├── state_machine.py             # §10 orchestration loop
│   ├── tools.py                      # tool interfaces/registrations
│   └── llm/
│       ├── draft.py                   # message drafting prompt + call
│       └── extract.py                  # §11 structured extraction
├── integration/
│   └── razorpay_client.py               # §13 wrapper around official SDK
├── backend/
│   ├── main.py                           # FastAPI app
│   ├── db.py                              # §2 SQLAlchemy setup
│   └── routes/                             # dashboard API endpoints
├── frontend/
│   └── (minimal dashboard, framework TBD at build time — plain HTML/JS acceptable)
├── eval/
│   ├── run_eval.py                          # §14 3-arm harness
│   ├── scenarios.py                          # failure-injection scenario defs
│   └── results/
├── tests/
│   └── test_policy_engine.py                  # unit tests against §9 rules directly
├── data/
│   └── (generated synthetic datasets, gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Module interfaces (contract summary — build independently against these)

- `simulator` exposes: `SimulationEngine.run(n_days, policy_fn) -> SimulationResult` and `generate_potential_outcomes(...)`. No other module reaches into simulator internals.
- `features` exposes: `build_feature_vector(invoice_id, intervention_type) -> dict`. Only touches the DB (read-only) and simulator-generated state during offline training data prep.
- `models` exposes: `PropensityModel.predict_proba(feature_vector) -> float`. Pure function, no DB access.
- `policy` exposes: `select_action(invoice, customer, model, constraints) -> (Action, rationale_code)`. Depends on `features` and `models`, nothing else.
- `agent` exposes: `run_agent_cycle(invoice_id)`. Orchestrates calls to `features`, `models`, `policy`, `agent.llm`, `integration` (if live), and writes to `AuditLog`. This is the only module allowed to call multiple other modules — everything else stays a leaf dependency.
- `integration` exposes: `create_invoice`, `resend_link`, `fetch_payment_status`. Wraps the Razorpay SDK, no business logic.
- `eval` depends on `simulator` (for ground truth) and `agent`/`policy` (for the policies under test) — nothing else depends on `eval`.

This means: **simulator, features, models, policy, integration can all be built and unit-tested in parallel and independently**, since none of them depend on `agent` or `eval`. `agent` and `eval` are integration layers built after their dependencies exist.

---

## Implementation sequence and definition of done

**Stage 0 — Scaffolding**
DoD: repo structure exists, `requirements.txt` installs cleanly, empty FastAPI app runs, SQLite DB created with all tables from §2.

**Stage 1 — Simulator**
DoD: `SimulationEngine.run()` produces plausible per-archetype trajectories (spot-check: model-citizens pay near on-time, serial-promisers rarely keep promises) over a synthetic population of ~200–500 customers; potential outcomes generated correctly for all 7 actions per invoice-day.

**Stage 2 — Features + PRS**
DoD: `build_feature_vector()` returns all §6 columns correctly for a sample of invoices; PRS visibly tracks promise-keep history in a manual spot-check.

**Stage 3 — Logging policy + training data + model**
DoD: logging policy generates a training dataset with genuine action variation (not perfectly confounded with risk); GBM trains without error; dev-set AUC/calibration checked and reasonable (not necessarily high — reasonable given synthetic noise).

**Stage 4 — Policy engine**
DoD: all 9 rules from §9 pass unit tests in `tests/test_policy_engine.py` against constructed synthetic states, independent of the model or simulator being wired in yet.

**Stage 5 — EIV + action selection**
DoD: `select_action()` combines policy engine + model scores correctly; manual trace of 3–5 example invoices produces sensible action choices.

**Stage 6 — Evaluation harness (early, per build-order rationale)**
DoD: 3-arm comparison runs end-to-end on the simulator + policy engine (agent loop can be stubbed as "policy engine only" at this stage), produces Table 1 and Chart 1 with non-degenerate numbers (Agent arm recovers more than No-Intervention, ideally more than Naive).

**Stage 7 — Agent orchestration loop**
DoD: `run_agent_cycle()` executes the full detect→diagnose→decide→act→verify→reassess cycle for a batch of invoices over simulated days, writing correct `AuditEvent`s at each phase; a promise made and then verified correctly updates PRS.

**Stage 8 — LLM tools**
DoD: drafting produces on-tone messages for each action type; extraction correctly parses at least the ambiguous-reply failure case (#8) into `commitment_detected: false`.

**Stage 9 — Audit log completeness check**
DoD: every executed action in a full simulation run has a traceable, human-readable rationale chain in the audit log — spot check 10 random invoices end-to-end.

**Stage 10 — Razorpay live-slice integration**
DoD: 5–10 real invoices created via the Invoices API, at least one live payment-link flow completed in test mode, payment status correctly fetched and reflected in the DB/audit log.

**Stage 11 — Dashboard**
DoD: portfolio view, per-invoice audit drill-down, and the 3-arm chart render correctly from real run data — no mock data in the final dashboard.

**Stage 12 — Failure-injection scenarios**
DoD: the 4–5 must-have scenarios from the locked spec pass as automated tests and are demonstrable live.

**Stage 13 — README + demo rehearsal**
DoD: README documents methodology, the causal-evaluation boundary, and limitations; a full dry run of the 5-minute demo script completes without manual intervention.

---

This is the frozen contract. Build against it; if something genuinely doesn't fit during implementation, that's a signal to flag explicitly rather than silently drift from it.
