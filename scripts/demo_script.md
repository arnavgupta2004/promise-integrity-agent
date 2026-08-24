# Pitch video recording script

Strict, numbered, blind-follow-able. Every step is exactly one of:
- **[DO]** — a physical action (click, switch window, run a command).
- **[SAY]** — exact words to speak, verbatim. Nothing to improvise.
- **[POINT]** — where to point/gesture on screen while speaking.

Steps are numbered sequentially across the whole video, not per-beat. Cumulative time estimates are word-count-verified (see the bottom of this file for the exact methodology and numbers) — treat them as a pacing guide, not a stopwatch to hit exactly.

---

## PRE-RECORDING SETUP (do all of this before you hit record — none of it is a numbered step)

1. Confirm the batch DB and eval results are current:
   ```bash
   cd promise-integrity-agent
   ls -la data/audit_completeness_batch.db eval/results/table1_three_arm_comparison.csv
   ```
2. Start the dashboard server:
   ```bash
   uvicorn backend.main:app --reload
   ```
3. Open **http://localhost:8000** in a browser window, sized so the audit-trail timeline text doesn't wrap excessively. Leave it on the **Portfolio Risk** tab (the default).
4. **Stage the PRS terminal output** (needed for steps 14–15 below) — run this now, in a visible terminal window, and leave the output on screen. Do not run it live during recording:
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
   Confirms `0.5` and `0.875` on screen. Leave this terminal window open and reachable.
5. Open a **second, separate terminal** in `promise-integrity-agent/`, prompt clean, nothing pre-typed. This is for step 18 — the **one and only command run live** during recording.
6. Arrange all three windows (browser, staged-PRS terminal, clean terminal) so you can switch between them with a single click/keystroke — no hunting during recording.
7. If you regenerated `data/audit_completeness_batch.db` as part of any of the above, **restart the uvicorn server** (`Ctrl+C`, re-run step 2) before proceeding — a swapped DB file leaves the running server serving stale data from the old file handle until restarted.
8. ⚠️ **Decide what you're actually going to say for `[name]` in step 2 below, and say it out loud once before recording.** The script still has the literal placeholder in it — it will not fix itself, and recording the bracketed text verbatim is an easy, embarrassing mistake to make on a real take.

Everything below assumes: browser on the dashboard (Portfolio Risk tab), PRS numbers already sitting in terminal 1, a clean terminal 2 ready, recording about to start.

---

## RECORDING STEPS

**1. [DO]** Start screen/video recording. *(0:00)*

**2. [SAY]** ⚠️ **REPLACE `[name]` WITH YOUR ACTUAL NAME BEFORE RECORDING THIS LINE — do not say the placeholder text or the brackets out loud.** "Hi, I'm [name], and this is the Promise Integrity Agent, built for the AI Revenue Recovery track. B2B merchants extend credit terms and then have no real way to chase overdue invoices — someone eventually emails, with no memory of past promises, and no sense of who's actually going to pay versus who's stringing them along." *(0:02–0:26)*

**3. [SAY]** "Our answer: a reliability score built from each customer's actual promise-keeping history — not just how overdue they are — driving every decision the agent makes. It's bounded throughout by hard compliance rules: contact caps, cooling periods, dispute stops. And every decision is fully audited. Let me show you it running on a real 100-invoice portfolio." *(0:26–0:50)*

### Beat 1 — Portfolio overview

**4. [POINT]** The "Total ₹ at risk" and "Total ₹ recovered" stat cards. *(0:50–0:52)*

**5. [SAY]** "100 invoices under management. ₹34,50,000 at risk across 57 open invoices, ₹7,35,000 already recovered on 43 — every number here is a live read off the real audit-log database, nothing mocked." *(0:52–1:05)*

**6. [POINT]** The risk-tier bars (High / Medium / Low). *(1:05–1:06)*

**7. [SAY]** "Risk tiering comes from each customer's live Promise Integrity Score, not just how overdue they are — that's the whole point of the next beat." *(1:06–1:17)*

### Beat 2 — Two invoices, same lateness, different treatment

**8. [DO]** Click the **Invoice Audit Trail** tab; select **`inv-batch-016-serial_promiser-0`** from the dropdown; scroll to the **day 11 `decide`** event. *(1:17–1:19)*

**9. [POINT]** The line `decision=human_escalation`, rationale `MAX_ATTEMPTS_REACHED`. *(1:19–1:20)*

**10. [SAY]** "Four contact attempts — day 0, 4, 7, 10 — never once produced a promise. PRS stuck at the neutral 0.5 default: not judged unreliable, just no evidence yet." *(1:20–1:33)*

**11. [DO]** Select **`inv-batch-035-reliable_always_late-0`** from the dropdown; scroll to the **day 11 `verify`/`decide`** events. *(1:33–1:35)*

**12. [POINT]** The `PROMISE_VERIFIED` / `promise_kept` line, then `decision=none`, rationale `EIV_MAX` right below it. *(1:35–1:36)*

**13. [SAY]** "A promise made day 0, verified kept that same day. Decide right after: none — left alone because they just proved they follow through." *(1:36–1:47)*

**14. [POINT]** Switch to terminal 1 (the pre-staged PRS output); point at the two printed numbers. *(1:47–1:48)*

**15. [SAY]** "Same amount, same lateness, same day — PRS 0.5 versus 0.875. That's the actual thesis: one customer hasn't earned trust yet, the other just re-earned it, and treatment follows the score, not the invoice." *(1:48–2:03)*

### Beat 3 — Live loop run

**16. [DO]** Switch to terminal 2 (the clean one). *(2:03–2:05)*

**17. [SAY]** "Two fresh invoices, two of this system's seven named customer archetypes — model_citizen and serial_promiser — run through the real loop live, right now, for 12 simulated days." *(2:05–2:17)*

**18. [DO]** **Run live**: `python3 scripts/demo_live_run.py` — this is the one command executed during recording, not staged ahead of time. Takes ~3 seconds; let the output finish printing on screen. *(2:17–2:21)*

**19. [SAY]** (spoken over/immediately after the output) "Not a replay — the actual Detect-through-Reassess loop, executing live. Watch the reliable customer: one plan proposal day 0, paid by day 1. Watch the unreliable one: same opening move, but day 3 they promise, then go quiet — the cooling-period rule holding off contact while it's pending. Same start, completely different trajectory." *(2:21–2:44)*

### Beat 4 — Promise tracked and verified

**20. [DO]** Switch back to the browser; select **`inv-batch-009-serial_promiser-0`** from the dropdown (146 events). *(2:44–2:46)*

**21. [POINT]** The **day 0 `act`** event, rationale `PROMISE_CAPTURED`. *(2:46–2:47)*

**22. [SAY]** "A plan proposal goes out on day zero, and the customer commits to a date — that commitment becomes a real Promise row in the database, not just a line in a transcript." *(2:47–3:01)*

**23. [POINT]** The repeated `COOLING_PERIOD_ACTIVE` events, days 1–10. *(3:01–3:03)*

**24. [SAY]** "Ten days of silence while that promise is outstanding. That's not the agent going idle — it's rule three's cooling-period constraint, deliberately holding off contact so a customer who just committed isn't hounded before they've had a chance to follow through." *(3:03–3:20)*

**25. [POINT]** The **day 11 `verify`** event, `PROMISE_VERIFIED` / `promise_broken`. *(3:20–3:22)*

**26. [SAY]** "The grace period elapses, the payment never lands, the promise is marked broken — and that write updates this customer's PRS in the same motion. Not a nightly batch job. Immediate." *(3:22–3:35)*

**27. [POINT]** The **day 11 `decide`** event onward, rationale `PRS_BELOW_PLAN_FLOOR,EIV_MAX`. *(3:35–3:36)*

**28. [SAY]** "On the very next decision cycle, the freshly-lowered score disqualifies this customer from another plan proposal. That's the Reassess feedback loop the architecture calls for — not asserted in a slide, sitting right here in the audit log." *(3:36–3:53)*

**29. [DO]** Scroll down to the **day 22** event. *(3:53–3:55)*

**30. [POINT]** `ALREADY_TERMINAL`; the invoice header showing `status: paid`. *(3:55–3:56)*

**31. [SAY]** "And yet this same invoice quietly gets paid a few days later. The broken promise didn't block recovery — it just changed how the system treats this customer from here on." *(3:56–4:10)*

**32. [SAY]** "Capture, patience, verification, consequence — and none of it needed a human to step in until the record actually demanded one." *(4:10–4:19)*

### Beat 5 — A stop case

**33. [DO]** Select **`inv-batch-010-disputer-0`** from the dropdown. *(4:19–4:21)*

**34. [SAY]** "A real, naturally-occurring dispute from the population — not a constructed test scenario." *(4:21–4:26)*

**35. [DO]** Scroll to the **day 4 `decide`/`act`** pair. *(4:26–4:28)*

**36. [POINT]** `DISPUTE_UNRESOLVED`, `human_escalation`. *(4:28–4:30)*

**37. [SAY]** "An open dispute escalates immediately, ahead of every other rule — rule 1, highest priority. It overrides everything, including whatever the model ranked highest. The moment a dispute is open, the agent stops making its own calls, full stop." *(4:30–4:46)*

### Beat 6 — Close on the 3-arm numbers

**38. [DO]** Click the **3-Arm Comparison** tab. *(4:46–4:48)*

**39. [SAY]** "Proof this isn't just a plausible pipeline. Three arms, same held-out population, same behavior: do nothing, nag everyone, or run this agent." *(4:48–4:58)*

**40. [POINT]** The bar chart, left to right. *(4:58–4:59)*

**41. [SAY]** "No-Intervention: ₹1.10 crore net recovery. Naive-Uniform: ₹1.10 crore. Promise Integrity Agent: ₹1.20 crore — plus ₹9.67 lakh over doing nothing, plus ₹9.48 lakh over naive, after ₹2.51 lakh in cost." *(4:59–5:12)*

**42. [SAY]** "And the agent isn't grading its own homework — the model that decided never saw these outcomes. It only sees pre-action features; the counterfactual ground truth comes from the simulator's privileged internals, revealed only in the eval harness, never fed back. That's what makes this a causal number, not a self-report — full detail in the README." *(5:12–5:37)*

### Wrap

**43. [SAY]** "This wasn't just built — it was tested to try to break it, repeatedly, and what we found got fixed, not buried. A training population that would have quietly confounded the evaluation was caught and replaced with a genuinely held-out one before this eval ever ran. A real bug let 101 messages go out that the policy engine had already flagged as needing human approval — caught not by a checklist of expected failures, but by a generic completeness checker built to catch any violation pattern, not just the ones we anticipated. And two of the policy engine's own hard safety rules — dispute-handling and no-contact-honoring — turned out to be structurally unreachable through the real agent loop, despite passing their unit tests. We found that late, fixed it, and re-ran the full regression — the evaluation, the audit batch, this dashboard — to confirm the fix actually held. That's the real claim here: not that this agent is correct, but that we went looking for the ways it wasn't, and it's on the record. Thanks for watching." *(5:37–6:53)*

**44. [DO]** Stop recording. *(6:53)*

---

## IF SOMETHING GOES WRONG

**A click/tab/dropdown doesn't register.** Pause, click again. Do not narrate the fumble ("let me just—") — dead air edits out cleanly, an on-camera apology doesn't. Since this is a recorded submission, not a live presentation, the simplest fix is usually the right one: stop, back up a few seconds, and redo that [DO] step and everything after it in the same take, or splice in a clean re-take of just that segment in editing.

**`demo_live_run.py` (step 18) errors or hangs.** It's deterministic and has been verified to run in ~3 seconds with zero network calls — a hang means something changed in the environment since this script was written, not bad luck on this run. Do not debug live on camera. Kill it (`Ctrl+C`), say the [SAY] line at step 19 anyway (it describes the *expected* output, which is accurate to what the script does), and note in post-production that beat 3 needs a re-shoot once the script is fixed. Never attempt to fix code live during a recording take.

**You lose your place in the audit trail (wrong day, wrong invoice, scrolled too far).** Every invoice ID and day number in this script is written out explicitly in the [DO]/[POINT] steps above — glance back at the step, not at the dashboard, to reorient. If you're visibly lost for more than a couple of seconds, cut and re-take that beat from its most recent numbered [DO] step rather than pushing through.

**You blank on a [SAY] line.** Every line is short enough to read directly off this document if needed — this script is meant to be visible off-camera (second monitor, printed page, teleprompter) precisely so nothing needs to be memorized. Re-take the line rather than paraphrase; the numbers and rationale codes in each line are exact and shouldn't be improvised.

**Recording runs long.** Don't rush mid-recording — a rushed demo reads worse than a slightly-long one. Finish the take at a natural pace; you can adjust playback speed or trim in editing afterward (the wrap and the beat-6 causal-methodology line are the safest places to tighten if you do trim, since they restate points made earlier).

---

## Timing methodology (for reference, not part of the recording)

Every one of the 21 [SAY] lines above was word-counted directly: 870 words total across 44 steps (up from 581 after this content-quality pass — the wrap and several beat-4 lines were deliberately expanded, not trimmed, once timing stopped being a constraint). The cumulative timestamps are computed from that at a brisk 140 wpm pace plus ~42s of mechanical time (clicks/scrolls/the real `demo_live_run.py` runtime), landing at 6:55. Speak at whatever pace feels natural — since this is a recorded, editable video, the timestamps above are a pacing guide for staying roughly on track while recording, not a target to hit exactly. Playback speed and trimming in post cover the rest.

---

## If asked: live LLM call

If a reviewer specifically wants to see a real Gemini-backed draft or extraction call live, don't run one cold — today's quota has failed repeatedly during this build (documented in `eval/results/table3_failure_injection_trails.json`'s `Ambiguous reply` scenario notes). Say so plainly, then show either:
- The real, already-executed extraction trace for `inv-batch-000-reliable_always_late-0`'s amount-mismatch example (dashboard, Invoice Audit Trail, look for `PROMISE_CAPTURED_AMOUNT_MISMATCH`) — genuine Gemini output, just not called live in front of them.
- `tests/test_llm_tools.py` — the real Stage 8 test suite that exercises `agent/llm/draft.py` and `agent/llm/extract.py` against the live API when quota allows.

Offer to actually attempt a fresh live call only if they want to watch it possibly fail — frame it as "let's see if quota's back" rather than promising it'll work.

## If asked: the Razorpay live-slice integration

Real invoices and a real payment link were created via the actual Razorpay test-mode API — genuine `plink_...`/`inv_...` IDs, reachable through the real decide→act policy path, not a forced demo trigger (see `scripts/live_slice_demo.py`'s docstring for the exact IDs). Completing the final checkout is blocked by an account-level KYC/international-payment eligibility gate on this specific test account — documented in the README's Known Limitations as exactly that, not glossed over as a code defect. The downstream settlement logic (what happens once Razorpay reports a payment captured) was independently verified via `tests/test_live_slice_verify.py`'s mocked-response tests, so the gap is specifically "we couldn't complete one manual checkout on this account," not "the integration is unproven."
