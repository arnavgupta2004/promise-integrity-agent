# Promise Integrity Agent — Technical Architecture
### Track 03: AI Revenue Recovery — B2B Receivables

**Status:** Locked problem specification → architecture design (pre-implementation)

---

## 0. Core metric definition

**Payment/Promise Reliability Score (PRS)** — a per-customer, continuously updated scalar in [0,1] representing how much weight to place on this customer's stated commitments, derived from:
- rolling promise-keep rate (last K promises, recency-weighted)
- relative lateness trend (is days-overdue/baseline improving or worsening)
- dispute frequency and resolution pattern
- response rate to prior outreach

This is a *behavioral* score, not a credit score — it updates every time a promise resolves (kept/broken) or a payment lands. It is the single most important derived feature, feeding both the risk model and the policy engine.

---

## 1. End-to-end system architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│ Data Layer        │     │ Intelligence Layer  │     │ Action Layer      │
│                    │     │                     │     │                   │
│ Customers          │────▶│ Feature Engine       │────▶│ Policy Engine     │
│ Invoices            │     │ (deterministic)      │     │ (deterministic)   │
│ Payments            │     │                     │     │                   │
│ Communications       │◀───│ Dual Propensity       │────▶│ Intervention       │
│ Promises             │     │ Model (ML)            │     │ Selector (rules)   │
│ Disputes             │     │  P(pay | no-action)    │     │                   │
│                    │     │  P(pay | intervention)  │     │ Message Generator  │
│                    │     │                     │     │ (LLM, scoped)      │
└──────────────────┘     └───────────────────┘     └──────────────────┘
         ▲                                                     │
         │                                                     ▼
         │                                          ┌──────────────────┐
         └──────────────────────────────────────────│ Verification &     │
                                                       │ Audit Layer         │
                                                       │ (deterministic)     │
                                                       └──────────────────┘
                                                                │
                                                                ▼
                                                       ┌──────────────────┐
                                                       │ Eval Harness         │
                                                       │ (3-arm comparison)   │
                                                       └──────────────────┘
```

**Orchestration:** a simple scheduler (not a heavyweight agent framework) sweeps open invoices daily (simulated day-steps), runs each through Detect → Diagnose → Decide → Act → Verify → Reassess, and writes every step to the audit log. This is closer to a **rules-and-models pipeline with one bounded LLM call per action**, not an autonomous multi-step agent — this is a deliberate choice (see §9).

---

## 2. ML formulation and target variables

Two related but distinct prediction targets, per your refinement:

**Target 1 — Baseline propensity:**
`P(pay within N days | no intervention)` — trained on the subset of historical invoice-customer instances where no contact was made (or using the no-intervention arm of the simulator), using pre-intervention features only.

**Target 2 — Intervention-conditional propensity:**
`P(pay within N days | intervention = X)` — trained on instances where a specific intervention type was applied, same feature set plus intervention-type as a feature (or separate models per intervention type if data allows).

**Derived decision quantity — expected incremental value:**
```
EIV(invoice, intervention) = amount × [P(pay | intervention) − P(pay | no intervention)] − cost(intervention)
```
This is what makes intervention selection **economically grounded** rather than a lookup table dressed up as ML: the policy picks the intervention (including "none") that maximizes EIV, subject to hard safety constraints from the policy engine (§8).

**N** (the pay-within-N-days horizon) should be fixed per risk tier, e.g., N=7 for near-term, N=21 for aging receivables — avoids one global horizon papering over very different customer situations.

---

## 2b. Causal formulation (resolved)

The risk flagged is real: an observational propensity model trained on "customers who happened to receive intervention X" is not automatically a causal estimate of intervention X's effect, because the policy that generated the training data already selects who gets which intervention based on risk — classic confounding. Since we control the simulator, we should exploit that rather than pretend we're in a purely observational setting.

**Potential-outcomes formulation:** for each invoice-customer state `s`, the simulator's latent behavior model defines a full set of potential outcomes `{Y(a) : a ∈ actions}` — what would happen under each possible action, including no-action — generated deterministically by the simulator's hidden archetype parameters at that point in time. The **policy** (agent) only ever observes pre-action features and selects one action `a*`; it never sees the other potential outcomes. The **evaluation harness**, running after the fact with privileged access to the simulator's internals, can reveal `Y(a)` for every `a`, not just the one chosen — this is genuine counterfactual ground truth, not something available in a real observational dataset.

This gives a clean separation of roles:

- **Decision-time model** (what the agent actually uses to act): a single **S-learner** — one GBM model, with intervention-type included as a categorical/one-hot feature, predicting `P(pay within N days | features, intervention)`. Trained on simulated rollout data where interventions were assigned with enough variation (e.g., an exploration/logging policy during data generation, not the final greedy policy) to avoid the confounding trap of "risky customers only ever see aggressive interventions in the training data." This is the credible, honestly-caveated proxy for causal effect — it's what the agent would have to rely on in a real deployment where true potential outcomes are unobservable.
- **Evaluation ground truth** (what scores the policy afterward): the simulator's actual `Y(a)` potential outcomes, used only in the eval harness, never fed back into the decision-time model. This is what makes the 3-arm comparison (§7) a genuine causal statement about incremental recovery, not just "the agent's own model agreed with itself."

**Why S-learner over T-learner (separate models per intervention) or a naive simulator-grounded policy:** given a synthetic batch of likely low-thousands of rows, splitting into separate per-intervention models (T-learner) fragments the data too far to train reliably — each intervention type would get a small, noisy slice. A single shared model with intervention as a feature pools statistical strength across interventions while still letting the model learn intervention-specific effects through feature interactions. A "simpler simulator-grounded policy" (just looking up the simulator's known parameters to pick the best action) would be more accurate but scientifically dishonest — it would mean the agent is cheating by reading the answer key, which defeats the entire purpose of the exercise. The chosen design deliberately keeps the decision-time model blind to ground truth and reserves the ground truth exclusively for scoring — that boundary is the single most important integrity property of the evaluation and should be stated explicitly in the demo.

---

## 3. Feature engineering

Deterministic, computed per invoice-customer pair at decision time:

- `relative_lateness` = days_overdue / customer_avg_days_to_pay (capped)
- `PRS` (Payment/Promise Reliability Score, §0)
- `PRS_trend` = slope of PRS over last 90 simulated days
- `dispute_rate` = disputes / total_invoices (customer lifetime)
- `response_rate` = replies / contacts (last 5 contacts)
- `partial_payment_rate`
- `amount_tier` (bucketed, relative to customer's typical invoice size — a ₹50k invoice means different things to different customers)
- `days_since_last_contact`
- `active_promise_flag` + `days_until_promised_date` (if a promise is outstanding)
- `segment` (categorical, from customer archetype in simulation / industry in real data)
- `broken_promise_streak` (consecutive, most recent)

No LLM-derived features feed the propensity models — keeps the ML pipeline fully deterministic and auditable, and keeps the LLM's blast radius contained to generation/extraction (§9).

---

## 4. Model candidates

| Candidate | Verdict |
|---|---|
| **Gradient-boosted trees (LightGBM/XGBoost)** | **Chosen.** Handles tabular, mixed-type features well, gives feature importances/SHAP for explainability (important for "explainable, bounded" bar), trains fast enough to iterate during the buildathon, and matches your existing ensemble experience. |
| Logistic regression | Good baseline/sanity-check model, too weak to capture interactions (e.g., PRS × relative_lateness) — use only as a comparison baseline in the eval, not the production model. |
| Deep learning (tabular NN) | Overkill for this data volume (synthetic batch, likely low thousands of rows); adds complexity without improving credibility — explicitly avoid per "don't over-engineer." |
| LLM-as-scorer | Rejected for the propensity model itself — not calibratable, not reproducible, undermines "honest metrics" credibility. LLM stays out of anything that produces a probability used for money decisions. |

Two GBM models (or one model with intervention-type as a categorical feature, if data is sparse) trained on simulator-generated outcomes.

---

## 5. Synthetic data generation + customer behavior simulator

A **generative simulator**, not a static CSV — this is what makes the counterfactual evaluation (§7) possible at all, since real-world data never gives you "what would have happened without intervention."

- Each customer instantiated from one of the 7 archetypes (see locked spec §8), with per-customer noise drawn from archetype-specific distributions (e.g., serial-promisers draw `keep_probability ~ Beta(2,8)`, model-citizens draw `~ Beta(9,1)`)
- Simulator runs day-by-day: invoices age, the **policy under test** decides whether/how to contact, the customer's latent behavior model (hidden from the agent) decides whether to respond, promise, pay, or ignore, conditioned on the archetype parameters *and* whatever intervention was applied that day
- Critically: the customer's latent response function includes a real (simulated) intervention effect — e.g., firm reminders increase near-term pay probability by archetype-specific amounts, over-contacting decreases response probability (fatigue) — so that "with intervention" outcomes are causally different from "without," not just relabeled
- This is what lets you legitimately claim incremental recovery rather than "customer paid, therefore agent worked" (a trap Track 04's "cherry-picked match" warning is really pointing at)

---

## 6. Train/dev/test methodology

- **Customer-level split** (not invoice-level) into train/dev/test — no customer's data appears in more than one split
- **Temporal ordering within the simulation**: propensity models trained on outcomes from simulated days 1–N, evaluated on days N+1–M, so the model never has access to future promise/payment outcomes for a customer it's scoring
- Dev set for hyperparameter tuning and threshold selection (e.g., PRS floor for payment-plan eligibility), test set touched exactly once for final reported numbers

---

## 7. Counterfactual evaluation — three-arm comparison

Run the **same simulated customer population** (same seed, same underlying latent behavior) through three independent policy arms:

1. **No intervention** — invoices simply age, no contact ever made
2. **Naive/uniform policy** — every overdue invoice gets the same reminder cadence regardless of risk/reliability
3. **Promise Integrity Agent** — full Detect→Diagnose→Decide→Act→Verify loop

Because the underlying customer behavior model is the same across arms (only the intervention differs), this is a valid counterfactual comparison, not an A/B test on different populations. Report:
- Total ₹ recovered per arm
- Incremental recovery: Agent − No-intervention, and Agent − Naive
- Cost-adjusted net recovery per arm
- Escalation precision (Agent arm only): % of human-escalated invoices that were truly unrecoverable via automation (checked against hidden `true_would_have_paid` labels)

This is the single most reviewer-credible artifact in the whole submission — a bar chart of three arms with the same population is unambiguous and hard to fake.

---

## 8. Policy engine and hard safety constraints

**Fully deterministic, no ML or LLM in this layer.** Implemented as an explicit rule table + the EIV-maximization step from §2, with hard constraints that override EIV whenever triggered:

- Contact frequency cap, cooling periods, escalation thresholds, max attempts, mandatory stop/human-approval conditions — all as coded in the locked spec §6
- Constraints are checked *before* EIV-based selection runs; if any hard constraint fires, it short-circuits straight to the mandated action (e.g., dispute flag → route to dispute queue, full stop) regardless of what EIV would have recommended
- This ordering (**safety constraints first, economic optimization second**) is the core of "bounded and gated" — worth stating explicitly in the demo narrative

---

## 9. Agent architecture — four distinct layers

The system genuinely is an agent, in the sense Razorpay's brief asks for ("detects... determines... executes a bounded recovery workflow") — but "agentic" is a property of the **orchestration loop's autonomy over time**, not of who holds financial authority. Collapsing those two ideas is what leads either to an over-permissioned LLM or to a pipeline that undersells its own agentic behavior. Four layers, kept explicitly separate:

**Layer 1 — Agent orchestration (genuinely autonomous):**
A stateful loop, run per invoice, that autonomously: observes current invoice/customer state → invokes tools to gather context (feature engine, PRS lookup, communication/promise history) → invokes the propensity model tool to score candidate actions → submits the ranked candidate to the policy engine for authorization → executes the authorized action → observes the outcome (reply received, promise made, payment landed) → updates state → **autonomously decides when to reassess** (immediately, at the next scheduled sweep, or after a promise's grace period elapses) without a human re-triggering it each time. This loop persisting and re-invoking itself across a multi-day timeline, driven by its own state transitions, is what makes it an agent rather than a one-shot script — this should be named explicitly as the agentic property in the writeup, since it's real and it's exactly what's being asked for.

**Layer 2 — Financial decision authority (deterministic, non-negotiable):**
The policy engine (§8) — hard safety constraints checked first, EIV-based ranking second. The agent's orchestration loop *calls* this layer as a tool and *must* accept its verdict; it cannot act on a candidate the policy engine didn't authorize. This is where "bounded and gated" lives, and it's exactly as strict as it was before — nothing here changes.

**Layer 3 — LLM language capability (narrow, tool-scoped):**
Unchanged in substance from the prior draft — drafting outreach text and extracting promise commitments from free text, invoked by the orchestration loop as one tool among several (alongside the feature engine, the propensity model, and the policy engine). The LLM is one tool the agent calls, not the agent's controller.

**Layer 4 — Hard safety controls (invariant, cross-cutting):**
Contact caps, cooling periods, dispute-triggered stops, no-contact requests, high-value human-approval thresholds — these apply regardless of what any other layer proposes, including a fully-authorized policy decision (e.g., a no-contact request overrides even an EIV-justified action). These are checked at execution time, not just decision time, so a stale authorization can't bypass a control that changed state in between.

**The resulting sentence for the demo/writeup:** *"The agent autonomously perceives, plans, and re-engages over time — but it never has unilateral financial authority; every action it proposes passes through a deterministic policy gate before execution."* This directly answers your concern: it reads as a genuine agent to a reviewer, not "ML model + rules engine + LLM message generator," while keeping every guardrail from the original design intact.

---

## 10. Promise extraction and verification

- **Extraction** (LLM, scoped per §9): structured output, low temperature, with a fallback to `no_commitment_detected` whenever confidence is low — explicitly designed to fail toward "ask a human" rather than hallucinate a date/amount (this is failure case #8 from the locked spec)
- **Verification** (deterministic): on `promised_date + grace_period`, check payments table for matching/partial payment against the invoice; mark promise kept/broken; **this write updates the customer's PRS**, which feeds back into the next Diagnose step — this feedback loop is the "Reassess" part of the loop and should be visually demonstrable (PRS visibly drops after a broken promise, next intervention gets more assertive)

---

## 11. Intervention/action system

Implemented as a small typed action registry (soft reminder, firm reminder, channel escalation, payment link resend, payment-plan proposal, promise capture, human escalation, no-action), each with:
- eligibility preconditions (checked against policy engine constraints)
- a cost value (for net-recovery accounting)
- an execution function (simulated dispatch in MVP; real Payment Links API call in the Razorpay-integrated version)

---

## 12. Audit-log design

Append-only log, one row per action or state change:
```
timestamp | invoice_id | customer_id | step (detect/diagnose/decide/act/verify/reassess)
| input_features_snapshot | model_output (if any) | decision | rationale_code
| constraint_triggered (if any) | executed_action | human_approval_required (bool)
```
Every decision carries a **rationale_code** (e.g., `PRS_BELOW_FLOOR`, `PROMISE_BROKEN_2X`, `HIGH_VALUE_FIRST_LATE`) mapping to a human-readable explanation — this is what makes "every money action explainable" checkable by a reviewer in the demo without reading code.

---

## 13. Razorpay test-mode API integration

- **Invoices API** — back the invoice/customer data model with real Razorpay invoice entities instead of a pure local table, so the demo is visibly hitting real Razorpay infrastructure
- **Payment Links API** — the "payment link resend" intervention becomes a genuine `create`/`resend` API call (test mode allows up to 30 links/business — sufficient for a demo batch, not for the full synthetic simulation, which stays local)
- **Payments API** — verification step fetches real payment status for invoices paid via test-mode payment link, closing the loop with an actual Razorpay signal rather than a mocked one
- Practical split: the **large-batch simulation and evaluation** (hundreds of synthetic customers, for the 3-arm comparison) runs entirely locally against the simulator; a **small live demo subset** (5–10 invoices) runs against real Razorpay test-mode APIs to prove genuine integration during the 5-minute demo

---

## 14. Database / schema design

Relational (SQLite for MVP — zero setup, trivially portable for a demo; Postgres if there's time and multi-user access matters):

Tables: `customers`, `invoices`, `payments`, `communications`, `promises`, `disputes`, `audit_log`, `policy_config` (versioned, so constraint changes are themselves auditable), `model_runs` (tracks which model version/threshold was active for reproducibility).

Foreign keys throughout; `audit_log` references invoice_id/customer_id but is append-only and never updated in place (integrity requirement, not just style).

---

## 15. Backend / frontend components

- **Backend**: Python (FastAPI) — simulation engine, feature pipeline, GBM models (LightGBM), policy engine, LLM tool calls, audit logging, Razorpay API client
- **Frontend**: a single-page dashboard (React or even a well-built HTML/JS artifact for the demo) showing: portfolio risk overview, per-invoice timeline (detect→diagnose→decide→act→verify events pulled from audit_log), 3-arm comparison chart, and a "why did the agent do this" drill-down per decision (rationale_code → explanation)
- Keep frontend genuinely minimal — a clean dashboard that surfaces the audit trail and the 3-arm chart clearly beats a feature-bloated UI; the reviewer's attention should go to the numbers and the reasoning, not UI polish

---

## 16. Evaluation harness

A standalone script, separate from the "live" pipeline, that:
1. Instantiates the customer population (fixed seed)
2. Runs all three policy arms against identical latent behavior
3. Collects ₹ recovered, cost, net recovery, escalation precision per arm
4. Produces the comparison table/chart used in the demo
5. Re-runs on a held-out customer split to confirm numbers aren't an artifact of the dev set

This harness *is* the credibility of the submission — treat it as a first-class deliverable, not an afterthought script.

---

## 17. Failure injection / testing

Directly implement the 9 failure cases from the locked spec (§10) as **unit-style scenario tests** against the policy engine and pipeline — each asserts the expected stop/escalate/proceed behavior. These tests double as demo material (§12 of the locked spec, point 5) and as proof the "bar" (graceful failure handling) is met by design, not by luck.

---

## 18. Repository structure

```
/simulator          # customer archetypes, behavior model, day-step engine
/features            # deterministic feature engineering
/models              # GBM training, evaluation, saved artifacts
/policy              # constraint engine, EIV-based selector
/agent               # LLM tool wrappers (drafting, extraction) — thin, isolated
/integration          # Razorpay API client (invoices, payment links, payments)
/backend              # FastAPI app tying it together, audit logging
/frontend             # dashboard
/eval                # 3-arm harness, failure-injection scenario tests
/data                 # generated synthetic datasets, schema migrations
README.md             # architecture summary + how to reproduce eval numbers
```

---

## 19. Deployment / demo architecture

No real deployment needed — this is a buildathon submission, not a production service. Run locally or on a single cloud VM for the demo: FastAPI backend + SQLite + static frontend build, Razorpay test-mode keys in env vars, evaluation harness output (charts/tables) pre-generated and also re-runnable live if time allows during the panel interview.

---

## 20. Hardest technical risks and mitigations

| Risk | Mitigation |
|---|---|
| Simulator behavior model is *itself* arbitrary — reviewer could ask "why should we trust your simulated customers behave realistically?" | Ground archetype parameters in publicly known B2B DSO/collections statistics where possible; be transparent about assumptions in the README rather than hiding them — credibility comes from honesty about limitations, not pretending the simulation is real data |
| Propensity model overfits due to small synthetic batch size | Report confidence intervals, use customer-level cross-validation, avoid overclaiming precision beyond what batch size supports |
| LLM promise-extraction hallucinates a date/amount not actually stated | Structured output schema + low temperature + explicit `no_commitment_detected` fallback + test this specifically in failure injection (§17) |
| 3-arm comparison accidentally leaks intervention effects across arms (e.g., shared random state causing correlated outcomes) | Separate RNG streams per arm, seeded independently but reproducibly; document this in the eval harness |
| Scope creep — trying to build all 7 intervention types, full frontend polish, and live Razorpay integration in limited time | MVP/stretch split below exists specifically to prevent this |
| Reviewer time is short (a few minutes) — architecture complexity doesn't automatically translate to demo clarity | Audit-log rationale_codes and the 3-arm chart are the two artifacts that must be flawless; everything else is secondary |

---

## 20b. Complexity audit

| Component | Verdict | Rationale |
|---|---|---|
| Simulator (7 archetypes, day-step, hidden potential outcomes) | **Must-have** | Nothing else works without it — it's the source of both training data and causal ground truth |
| PRS (Payment/Promise Reliability Score) | **Must-have** | Core differentiator vs. a generic dunning bot; feeds both model and policy |
| Feature engineering | **Must-have** | Required for the propensity model |
| S-learner propensity model (single GBM, intervention as feature) | **Must-have** | Simplified per §2b — one credible model beats two fragile ones |
| Separate per-intervention (T-learner) models | **Remove** | Data volume doesn't support it; S-learner is more credible, not less |
| EIV computation | **Must-have** | The economic grounding the refinement specifically asked for |
| Policy engine (hard constraints + EIV ranking) | **Must-have** | This is "bounded and gated," the core of the bar |
| Agent orchestration loop (Layer 1, §9) | **Must-have** | This is what makes it an agent, not a script — directly answers the brief's wording |
| LLM drafting + extraction (Layer 3, §9) | **Must-have** | Required by the track, but narrow and cheap to build |
| Audit log with rationale codes | **Must-have** | The single artifact most likely to build reviewer trust fastest |
| 3-arm evaluation harness (with true potential-outcome ground truth) | **Must-have** | The other most-important artifact; this is the causal proof, not just a claim |
| Razorpay Invoices API integration | **Must-have (moved from Stretch)** | Agreed — a submission that visibly touches real Razorpay infrastructure is materially more credible than a pure simulator, and the track explicitly asks for test-mode APIs |
| Razorpay Payment Links API (small live slice) | **Must-have (moved from Stretch)** | Same reasoning; test-mode's 30-link cap is fine for a demo-sized slice |
| Razorpay Payments API verification | **Must-have (moved from Stretch)** | Closes the loop with a real signal instead of a mocked one |
| Failure-injection scenarios | **Strongly valuable** | Build 4–5 of the 9 (the sharpest ones: dispute-stop, no-contact-request, broken-promise-streak-escalation, ambiguous-reply extraction) as must-have; the remaining 4–5 are optional polish |
| Minimal dashboard (portfolio view + audit drill-down + 3-arm chart) | **Strongly valuable** | Needed to present the must-have artifacts clearly, but should stay deliberately plain |
| Rich frontend polish, live in-demo eval re-run | **Optional** | Nice if time allows, doesn't change the core signal |
| Policy config versioning / UI | **Optional** | Good engineering hygiene, not reviewer-facing value |
| SHAP explainability visualization | **Optional** | Adds credibility marginally; cut first if time is short |
| Full multi-channel real dispatch infra (actual WhatsApp/SMS sending) | **Remove** | Simulate the channel effect; don't build real messaging infrastructure for a demo |
| Full human-approval workflow UI | **Remove** | A flagged row in the DB/audit log is sufficient; no need for a workflow UI |

---

## MVP vs. Stretch (revised)

**MVP — required for submission:**
- Synthetic simulator with all 7 archetypes, day-step engine, hidden potential-outcome ground truth (§2b)
- PRS computation, feature engineering
- Single S-learner propensity model (intervention-as-feature), trained on a properly varied logging policy to avoid confounding
- EIV-based intervention selection
- Policy engine with all hard safety constraints
- Agent orchestration loop (Layer 1) — explicit perceive → tool-call → propose → authorize → execute → observe → reassess cycle
- LLM scoped to drafting + extraction only (Layer 3), with guardrail checks before dispatch
- Full audit log with rationale codes
- 3-arm evaluation harness scored against true simulator ground truth, with the comparison chart
- 4–5 sharpest failure-injection scenarios, demonstrable live
- **Real Razorpay test-mode integration**: Invoices API + Payment Links API + Payments API, exercised on a small live demo subset (5–10 invoices) alongside the large-scale local simulation
- Minimal dashboard: portfolio risk view, per-invoice audit drill-down, 3-arm chart
- README documenting methodology, the causal-evaluation boundary (§2b), and known limitations

**Stretch — add only if time remains:**
- Remaining 4–5 failure-injection scenarios
- SHAP feature-importance visualization
- Policy config versioning shown in UI
- Live in-demo re-run of the full evaluation harness
- Frontend visual polish beyond functional clarity

---

## Build order (first things first)

1. **Simulator + archetypes + potential-outcome generation** — everything downstream depends on this; get it producing plausible-looking trajectories before anything else
2. **Feature engineering + PRS** — needed to generate training data from simulator rollouts
3. **Logging policy for data collection** — a simple varied/exploratory policy to generate the training dataset without the confounding problem flagged in §2b
4. **S-learner propensity model** — train and sanity-check against dev set
5. **Policy engine (hard constraints + EIV ranking)** — this is testable in isolation against synthetic states before wiring it to the model
6. **3-arm evaluation harness** — build this early, not last, so you can see incremental recovery numbers as soon as the policy engine exists, and iterate against a real signal rather than guessing
7. **Agent orchestration loop** — wire the above into the perceive→act→observe→reassess cycle
8. **LLM drafting + extraction tools** — bolt on last, since the system is fully functional and testable without them (a placeholder template can stand in until this step)
9. **Audit log + rationale codes** — thread through from step 5 onward, not bolted on at the end
10. **Razorpay test-mode integration** — once the local pipeline works end-to-end, swap the small demo subset over to real API calls
11. **Dashboard** — last, once there's real audit-log and evaluation data to display

This order front-loads the two artifacts that carry the most reviewer credibility (the eval harness, the audit trail) and treats the LLM and the live API integration as late-stage attachments to an already-working deterministic core — which also means the project is demoable at almost every checkpoint after step 6.

---

This architecture is designed so that **every component maps to something the rubric explicitly asks for** (measured ₹ recovered → 3-arm harness with causal ground truth; compliant escalation → policy engine; stopping rules → hard constraints; audit trail → append-only log with rationale codes; agentic behavior → the orchestration loop in §9; AI Judgment → the LLM boundary in §9; Razorpay integration → real test-mode API calls in MVP) rather than existing for its own sake.
