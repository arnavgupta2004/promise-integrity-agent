# 5-minute demo script

Six beats, minute-by-minute, per the original problem specification's §12 demo format (the user's own summary of it: portfolio overview → two-invoices-same-lateness-different-treatment → live loop run → promise tracked/verified → a stop case → close on the 3-arm numbers). Exact commands for every beat — nothing improvised live.

**No beat in this script depends on a live Gemini API call succeeding.** The live-loop-run beat uses `agent.tools.LLMDraftTool` (the free, deterministic stub), not the real Gemini-backed drafting tool — today's quota has been exhausted multiple times during this build, and this script is built to not care. If a reviewer specifically asks to see a real LLM call, see **"If asked: live LLM call"** at the bottom — don't improvise that live either.

---

## Setup (before the reviewer sits down — not part of the 5 minutes)

```bash
cd promise-integrity-agent

# 1. Confirm the batch DB and eval results are current (skip regeneration if they already are).
ls -la data/audit_completeness_batch.db eval/results/table1_three_arm_comparison.csv

# 2. Start the dashboard server.
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in a browser tab, sized reasonably (not full mobile-width) so the audit-trail timeline text doesn't wrap excessively. Leave it on the **Portfolio Risk** tab (the default) — that's beat 1.

**If you regenerate `data/audit_completeness_batch.db` for any reason as part of setup** (e.g. `python3 scripts/audit_completeness_check.py`), **restart the uvicorn server afterward** — `Ctrl+C` then re-run step 2. This isn't optional: the dashboard's SQLite connection is opened once at server startup, and swapping the DB file out from under a running server (via `mv`, which is what the batch script does) leaves it serving stale data from the old file handle until restarted. This bit us for real during Stage 13's build.

Have a second terminal tab open in `promise-integrity-agent/` for beat 3 (the live loop run) — no server needed there, just the repo root.

---

**Timing note**: the beat markers below are calibrated, not just labeled to sum to 5:00. Every blockquoted/spoken line was extracted and word-counted directly (436 words total), plus measured/estimated mechanical time (clicks, dropdown selects, the live script's real ~2.8s runtime, ~32s total). The first draft ran 771 words — ~6.2 minutes once counted this way, over budget — the narration below is the trimmed version:

| Pace | Speaking time | + mechanical (~32s) | Total |
|---|---|---|---|
| 140 wpm (brisk) | 3:07 | | ~3:39 |
| 110 wpm (deliberate, realistic for a demo with numbers spoken aloud) | 3:58 | | ~4:30 |

Either pace lands comfortably under 5:00, with 30–80s of real margin for a stumble or a pointed question — not scripted to the exact edge. This is a word-count/pacing model, calibrated with the one number I could actually measure (the live script's real runtime), not a literal recording — **rehearse it aloud once regardless**; real speech has pauses and variance no model fully captures.

## 0:00 – 0:30 — Portfolio overview

Already on screen (Portfolio Risk tab). Say while pointing:

> "100 invoices under management. ₹34,50,000 at risk across 57 open invoices, ₹7,35,000 already recovered on 43 — every number here is a live read off the real audit-log database, nothing mocked."

> "Risk tiering comes from each customer's live Promise Integrity Score, not just how overdue they are — that's the whole point of the next beat."

*(If asked why "low risk" is empty: this dataset's unpaid invoices all cluster in the medium band — see the README's Known Limitations section. Don't dodge it, it's a documented, verified-real finding, not a bug you're hiding.)*

---

## 0:30 – 1:25 — Two invoices, same lateness, different treatment

Click **Invoice Audit Trail**. Both invoices share the exact same issue date and the exact same ₹15,000 amount — as of day 11, equally overdue by every calendar measure. **The thesis is specifically the PRS gap, not just the differing outcome.**

Run this **before the demo starts** (not live — it's setup, not a beat) so the numbers are already sitting in a terminal to point at:

```bash
python3 -c "
import datetime as dt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from features.feature_engine import compute_prs
engine = create_engine('sqlite:///data/audit_completeness_batch.db')
s = sessionmaker(bind=engine)()
as_of = dt.datetime(2025, 1, 11)
print('inv-batch-016 (serial_promiser) PRS:', compute_prs('batch-016-serial_promiser', as_of=as_of, session=s))
print('inv-batch-035 (reliable_always_late) PRS:', compute_prs('batch-035-reliable_always_late', as_of=as_of, session=s))
"
```

Prints **`0.5`** and **`0.875`**.

1. Select **`inv-batch-016-serial_promiser-0`**, scroll to **day 11** `decide`: `human_escalation`, **`MAX_ATTEMPTS_REACHED`**.
   > "Four contact attempts — day 0, 4, 7, 10 — never once produced a promise. PRS stuck at the neutral 0.5 default: not judged unreliable, just no evidence yet."
2. Select **`inv-batch-035-reliable_always_late-0`**, scroll to **day 11** `verify`/`decide`: promise kept, then `none`/`EIV_MAX`.
   > "A promise made day 0, verified kept that same day. Decide right after: none — left alone because they just proved they follow through."

> "Same amount, same lateness, same day — PRS 0.5 versus 0.875. That's the actual thesis: one customer hasn't earned trust yet, the other just re-earned it, and treatment follows the score, not the invoice."

---

## 1:25 – 2:10 — Live loop run

Switch to the second terminal. Run:

```bash
python3 scripts/demo_live_run.py
```

Runs in ~3 seconds, fully deterministic (seeded), zero network calls.

> "Two fresh invoices — one reliable, one unreliable — run through the real loop live, right now, for 12 simulated days."

While it prints:

> "Not a replay — the actual Detect-through-Reassess loop, executing live. Watch the reliable customer: one plan proposal day 0, paid by day 1. Watch the unreliable one: same opening move, but day 3 they promise, then go quiet — the cooling-period rule holding off contact while it's pending. Same start, completely different trajectory."

---

## 2:10 – 3:10 — Promise tracked and verified

Back to the dashboard, **Invoice Audit Trail**. Select **`inv-batch-009-serial_promiser-0`** (146 events — the richest full promise lifecycle in the batch).

1. **Day 0, `act`**: `PROMISE_CAPTURED` — "plan proposal goes out, customer commits to a date."
2. **Days 1–10**: `COOLING_PERIOD_ACTIVE` — "deliberately silent while the promise is outstanding — not neglect, rule 3's grace period."
3. **Day 11, `verify`**: `PROMISE_VERIFIED`, broken — "grace period elapsed, payment never landed, PRS updates immediately."
4. **Day 11 onward, `decide`**: `PRS_BELOW_PLAN_FLOOR,EIV_MAX` — "the lowered score disqualifies plan_proposal on the very next cycle. That's the Reassess feedback loop, in real data."
5. **Day 22**: `ALREADY_TERMINAL` — "quietly paid, sometime between day 21 and 22."

> "Capture, cooling period, verification, PRS consequence — paid eventually, but the score already reflects the broken promise."

---

## 3:10 – 3:50 — A stop case

Select **`inv-batch-010-disputer-0`**.

> "A real, naturally-occurring dispute from the population — not a constructed test scenario."

Scroll to **day 4 `decide`/`act`**: `DISPUTE_UNRESOLVED`, `human_escalation`.

> "An open dispute escalates immediately, ahead of every other rule — rule 1, highest priority. It overrides everything, including whatever the model ranked highest. The moment a dispute is open, the agent stops making its own calls, full stop."

*(Optional, if there's time: scroll further to show `ALREADY_TERMINAL` repeating — the agent never re-engages once escalated, even though the underlying dispute record is still capable of resolving in the background, verified separately in `tests/test_live_slice_verify.py`'s sibling test.)*

---

## 3:50 – 4:30 — Close on the 3-arm numbers

Click **3-Arm Comparison**.

> "Proof this isn't just a plausible pipeline. Three arms, same held-out population, same behavior: do nothing, nag everyone, or run this agent."

Read off the chart/table:
- **No-Intervention**: ₹1.10 crore net recovery.
- **Naive-Uniform**: ₹1.10 crore.
- **Promise Integrity Agent**: ₹1.20 crore — **+₹9.67 lakh over doing nothing, +₹9.48 lakh over naive**, after ₹2.51 lakh in cost.

> "And the agent isn't grading its own homework — the model that decided never saw these outcomes. It only sees pre-action features; the counterfactual ground truth comes from the simulator's privileged internals, revealed only in the eval harness, never fed back. That's what makes this a causal number, not a self-report — full detail in the README."

Close there. **4:30 on the clock — 30 seconds of buffer before the 5:00 ceiling for a stumble or a pointed question.**

---

## If asked: live LLM call

If a reviewer specifically wants to see a real Gemini-backed draft or extraction call live, don't run one cold — today's quota has failed repeatedly during this build (documented in `eval/results/table3_failure_injection_trails.json`'s `Ambiguous reply` scenario notes). Say so plainly, then show either:
- The real, already-executed extraction trace for `inv-batch-000-reliable_always_late-0`'s amount-mismatch example (dashboard, Invoice Audit Trail, look for `PROMISE_CAPTURED_AMOUNT_MISMATCH`) — genuine Gemini output, just not called live in front of them.
- `tests/test_llm_tools.py` — the real Stage 8 test suite that exercises `agent/llm/draft.py` and `agent/llm/extract.py` against the live API when quota allows.

Offer to actually attempt a fresh live call only if they want to watch it possibly fail — frame it as "let's see if quota's back" rather than promising it'll work.

## If asked: the Razorpay live-slice integration

Real invoices and a real payment link were created via the actual Razorpay test-mode API — genuine `plink_...`/`inv_...` IDs, reachable through the real decide→act policy path, not a forced demo trigger (see `scripts/live_slice_demo.py`'s docstring for the exact IDs). Completing the final checkout is blocked by an account-level KYC/international-payment eligibility gate on this specific test account — documented in the README's Known Limitations as exactly that, not glossed over as a code defect. The downstream settlement logic (what happens once Razorpay reports a payment captured) was independently verified via `tests/test_live_slice_verify.py`'s mocked-response tests, so the gap is specifically "we couldn't complete one manual checkout on this account," not "the integration is unproven."
