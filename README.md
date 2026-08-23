# Promise Integrity Agent

An AI-driven B2B receivables recovery system (Razorpay Buildathon, Track 03: AI Revenue Recovery). It watches open invoices, decides whether and how to nudge a customer toward paying, and does so under a policy layer that has final say over every action — the model proposes, a deterministic gate disposes.

## The problem

A B2B collections team's real leverage isn't in sending more reminders — it's in knowing *which* invoices need a human-costly touch and which will resolve on their own, and in remembering which customers' promises are worth trusting. Naive collections (a fixed reminder cadence for everyone) wastes contact budget on customers who were going to pay anyway and under-escalates the ones who won't. This project's premise: a per-customer reliability signal, combined with an economically-grounded action-selection rule and a small set of hard safety constraints, recovers more money at lower cost than either doing nothing or nagging everyone uniformly — and can prove it, not just claim it (see [Causal evaluation methodology](#causal-evaluation-methodology) below).

## The Promise Integrity Score (PRS)

PRS is a per-customer, continuously-updated scalar in `[0, 1]` representing how much weight to place on this customer's stated commitments — a *behavioral* score, not a credit score. It's the single most important derived feature: it feeds both the propensity model and the policy engine's rules directly (`policy/constraints.py`'s `PRS_BELOW_PLAN_FLOOR` rule, for one). Computed in [features/feature_engine.py](features/feature_engine.py)'s `compute_prs()` as a weighted combination of four components, each defaulting to a neutral 0.5 when its underlying evidence is absent rather than raising or collapsing to 0:

```
PRS = 0.45 × keep_component      (rolling promise-keep rate, exponentially recency-weighted, half-life 30d)
    + 0.25 × trend_component     (slope of relative-lateness over the customer's paid invoices)
    + 0.15 × dispute_component   (1 − lifetime dispute rate)
    + 0.15 × response_component  (reply rate over the last 5 agent-dispatched contacts)
```

It updates every time a promise resolves or a dispute closes — see the dashboard's [Invoice Audit Trail view](#dashboard) for a real example of a broken promise pulling PRS down and immediately disabling `plan_proposal` eligibility on the very next decision cycle (`PRS_BELOW_PLAN_FLOOR`).

## Four-layer agent architecture

The system is genuinely agentic — a stateful loop that observes, decides, acts, and autonomously re-engages over time without a human re-triggering it — but "agentic" describes the orchestration loop's autonomy, not who holds financial authority. Those are kept in four explicitly separate layers ([agent/state_machine.py](agent/state_machine.py)'s `run_agent_cycle`):

1. **Agent orchestration** (genuinely autonomous) — Detect → Diagnose → Decide → Act → Verify → Reassess, run per invoice, per simulated day. The loop decides for itself when to reassess next: immediately, at the next daily sweep, or once a promise's grace period elapses — driven by its own state transitions, not external re-triggering.
2. **Financial decision authority** (deterministic, non-negotiable) — [policy/constraints.py](policy/constraints.py)'s 9 hard-priority rules, checked first, then [policy/eiv.py](policy/eiv.py)'s expected-incremental-value ranking over whatever the rules left eligible. The orchestration loop calls this layer as a tool and cannot act on anything it didn't authorize.
3. **LLM language capability** (narrow, tool-scoped) — drafting outreach text ([agent/llm/draft.py](agent/llm/draft.py)) and extracting promise commitments from free text ([agent/llm/extract.py](agent/llm/extract.py)), invoked as one tool among several. The LLM is a tool the agent calls, not the agent's controller — and it fails toward *"no commitment detected"* rather than fabricating a date or amount when confidence is low.
4. **Hard safety controls** (invariant, cross-cutting) — contact caps, cooling periods, dispute-triggered stops, no-contact requests, high-value human-approval thresholds. These apply regardless of what any other layer proposes, and are checked at execution time, not just decision time, so a stale authorization can't bypass a control that changed state in between.

The sentence that captures the design: *the agent autonomously perceives, plans, and re-engages over time — but it never has unilateral financial authority; every action it proposes passes through a deterministic policy gate before execution.*

## Causal evaluation methodology

This is the project's core intellectual contribution, and the one property that has to be stated explicitly rather than left implicit: **the model the agent uses to decide is never allowed to see the answer key it's being graded against.**

An observational propensity model trained on "customers who happened to receive intervention X" is not automatically a causal estimate of X's effect — the policy that generated the training data already selected who got which intervention based on risk, so a naive model would just be learning "risky customers get aggressive interventions," not "aggressive interventions work." Since this project controls the simulator generating the data, that confounding trap is avoidable by exploiting a distinction a real deployment doesn't get for free:

- **Potential outcomes.** For every invoice/customer state, `simulator/behavior_model.py`'s `CustomerBehaviorModel.generate_potential_outcomes()` computes the *full* set of outcomes under every possible action — what would happen if the agent sent a soft reminder, a firm one, a payment link, did nothing, and so on — all at once, deterministically, from the customer's hidden archetype parameters. A real deployment only ever observes the one outcome for the one action actually taken; the simulator, because we built it, can also reveal the roads not taken.
- **The decision-time model never gets that privilege.** [models/propensity_model.py](models/propensity_model.py) is a single S-learner GBM (intervention type as a categorical feature, not a separate model per action — deliberately, so a synthetic batch of a few thousand rows doesn't get fragmented too thin to learn from) trained on rollout data with enough exploration variation to avoid the "risky customers only ever see aggressive interventions" confound. It only ever sees pre-action features plus a candidate intervention and predicts `P(pay within N days | features, intervention)` — this is the same thing an agent operating in a real deployment would have to rely on, since true potential outcomes are unobservable there. Everything downstream of this model — `policy/eiv.py`'s `EIV = amount × [P(pay|intervention) − P(pay|no intervention)] − cost(intervention)` ranking, the policy engine's authorization — is built on this same honestly-blind estimate.
- **The evaluation harness gets the privilege the model never does.** [eval/run_eval.py](eval/run_eval.py) runs *after* the fact, with access to the simulator's actual potential outcomes for scoring — never fed back into the decision-time model. This is what makes the 3-arm comparison a genuine causal statement about incremental recovery ("the agent recovered ₹X more than doing nothing, and here's the counterfactual proof"), not "the agent's own model agreed with itself." Table 2's escalation-precision metric goes one step further and explicitly scores the counterfactual for the action *not* taken (`Y("none")` at the moment of escalation) — a genuine "what would have happened otherwise" query that a real observational dataset could never answer directly.

In short: **decision time is blind, evaluation time sees everything, and the two are architecturally incapable of talking to each other.** That boundary — not the specific GBM hyperparameters or the exact EIV formula — is the single most important integrity property of this evaluation.

## Reproducing the 3-arm evaluation numbers

From a clean checkout:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train the decision-time propensity model (self-contained: generates its own synthetic rollout data via the simulator, no external inputs needed):

```bash
python3 models/train.py
```

This writes `data/training_data.csv` and `models/artifacts/propensity_model.joblib` — the artifact `eval/run_eval.py` loads for the Agent arm. Then run the 3-arm evaluation harness itself:

```bash
python3 eval/run_eval.py
```

This prints Table 1 (3-arm comparison), Table 1b (by PRS band), and Table 2 (escalation precision) to stdout, and writes `eval/results/table1_three_arm_comparison.csv`, `table1b_by_prs_band.csv`, `table2_escalation_precision.csv`, `chart1_net_recovery.png`, and `chart2_prs_trajectory.png`. The held-out evaluation population (`eval-` prefixed customer IDs, seed `4242`) is disjoint by construction from Stage 3's training population (`train-` prefixed, seed `7`), so these numbers aren't measuring the model against data it trained on.

To also reproduce the failure-injection scenario table (Table 3) and the Stage 1 archetype-behavior smoke test:

```bash
python3 eval/scenarios.py            # Table 3 -- 5 must-have failure cases, each asserting the exact rationale_code
python3 scripts/simulate_smoke_test.py  # archetype-distinctiveness + dispute-lifecycle assertions
```

To view the dashboard against the fuller 100-invoice/60-day audit-completeness batch (a separate, richer real-data run than the eval harness's population — see `scripts/audit_completeness_check.py`):

```bash
python3 scripts/audit_completeness_check.py   # generates data/audit_completeness_batch.db
uvicorn backend.main:app --reload             # serves the dashboard at http://localhost:8000
```

## Known limitations

**Aging-tier data thinness (Stage 3).** `models/train.py`'s training population is much thinner for the `aging` risk tier than `near_term`: 2,083 rows vs. 15,104 (about 1/7th), per `joint_distribution_table()`. Every individual `(lateness_bucket, action)` cell for bucket ≥ 1 clears the stratified-sampling guardrail (`check_bucket_coverage()`, `MIN_CELL_COUNT=45`) — so no single cell is starved — but the aging tier's overall statistical support is still far smaller than near_term's. The propensity model's calibration should be trusted less for higher-lateness invoices than for near-term ones as a result.

**Single `promised_date`/`promised_amount` per Promise row (Stage 8).** `backend/db.py`'s frozen §2 `Promise` schema carries exactly one `promised_date` and one `promised_amount` column. A customer reply describing multiple distinct commitments (e.g., a split payment across two dates) can only be captured as a single Promise row with one date/amount — `agent/llm/extract.py`'s `PromiseExtraction` is scoped to extract one primary commitment per reply, not a decomposed multi-installment plan.

**PRS clustering in the dashboard's batch dataset (Stage 11).** In the 100-invoice Stage 9 batch backing the dashboard, every unpaid invoice's PRS score clusters tightly in 0.50–0.575 — all land in the "medium risk" tier, with "high risk" and "low risk" empty. Verified directly against `compute_prs` that this is real, not a bucketing bug: with one invoice per customer and sparse payment/promise history in this particular dataset, most PRS components stay near their documented neutral 0.5 default. A batch with richer per-customer payment history would likely spread more across tiers.

**Stage 10 live-slice manual payment step (contract §13).** `scripts/live_slice_demo.py` creates real Razorpay test-mode invoices and, via the real policy engine, a real payment link. Completing the manual test-mode checkout to exercise `fetch_payment` end-to-end is blocked on this account: Razorpay's hosted checkout rejects both documented domestic test cards (Visa `4111 1111 1111 1111` and Mastercard `5104 0155 5555 5558`) with "international cards not supported." Per Razorpay's own docs this traces to KYC/international-payment eligibility gating on the account itself, not to the payment link, the card numbers, or anything in this integration.

Evidence gathered up to that account-level wall: 7/7 real invoices created via `create_invoice`, one real payment link created via `create_payment_link` (reached through the actual decide→act policy path, no forced trigger needed), and `fetch_payment_link` confirmed working via a standalone smoke test. See `scripts/live_slice_demo.py`'s docstring for full details and IDs, and `tests/test_live_slice_verify.py` for a mocked-response test proving the downstream verification logic (`_verify_live_slice_payment`: DB update, invoice settlement, audit event) works correctly from the point a "captured" payment status would arrive, independent of this account-level blocker.

## Dashboard

`backend/main.py` serves a minimal plain-HTML/JS dashboard (`frontend/index.html`) with three views: portfolio risk overview, per-invoice audit-trail drill-down (every `rationale_code` — including compound comma-joined trails — translated to plain language via `backend/rationale_explanations.py`), and the 3-arm comparison chart. See `scripts/demo_script.md` for a guided walkthrough.
