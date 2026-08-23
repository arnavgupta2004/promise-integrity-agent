"""
Smoke test for simulator/archetypes.py, behavior_model.py, and engine.py.

Two separate simulation runs:

  1. "always none" -- no contact is ever sent. Validates payment timing
     (avg_days_to_pay) and promise-keep dynamics (keep_probability_base,
     trend_slope) purely from spontaneous customer behavior.
  2. "always soft_reminder" -- a reminder is sent every day an invoice is
     open. Validates response_propensity specifically, which §4/§5 define
     in terms of responding *to contact* -- something run 1 cannot exercise
     at all, since it never contacts anyone.

Per-archetype stats from both runs are checked against §4's actual target
values (via simulator.archetypes.ARCHETYPES), with documented tolerance
bands -- not just pairwise comparisons between archetypes -- plus a
day-bucketed breakdown of the degrading archetype's keep-rate, since a
single first-half/second-half split turned out to hide its decline behind
an unrelated distributional effect (see the comment on DEGRADING_BUCKETS
below).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from simulator.archetypes import ARCHETYPE_NAMES, ARCHETYPES, sample_customer_latent
from simulator.behavior_model import CustomerBehaviorModel
from simulator.engine import SimInvoice, SimulationEngine

SEED = 42
# Bumped up from the original ~300 (43/archetype): the degrading-archetype
# day-bucket diagnostic below needs enough promise events per 15-day bucket
# to be readable rather than noise (43/archetype left some buckets at n=1).
N_CUSTOMERS = 900  # ~129 per archetype
N_DAYS = 90

# Tolerance bands for the per-archetype target-range assertions (see
# run_assertions). These are deliberately not tight to a single seed's
# exact output -- they exist to catch real breakage (e.g. an archetype
# collapsing to ~0, or two archetypes becoming indistinguishable), not to
# pin down decimal-precision calibration.
KEEP_RATE_ABS_TOLERANCE = 0.15
DAYS_TO_PAY_REL_BAND = (0.5, 1.25)  # observed must be within [0.5x, 1.25x] of the §4 target

DEGRADING_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]


def always_none_policy(invoice: SimInvoice, day: int, context: dict) -> str:
    return "none"


def always_soft_reminder_policy(invoice: SimInvoice, day: int, context: dict) -> str:
    return "soft_reminder"


def build_population(seed: int, n_customers: int, id_prefix: str) -> list[CustomerBehaviorModel]:
    rng = np.random.default_rng(seed)
    customers = []
    for i in range(n_customers):
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
        customer_id = f"{id_prefix}-{i:04d}-{archetype}"
        latent = sample_customer_latent(customer_id, archetype, rng)
        customers.append(CustomerBehaviorModel(latent))
    return customers


# ---------------------------------------------------------------------------
# Run 1: always none -- payment timing + promise-keep dynamics
# ---------------------------------------------------------------------------

def summarize_none_run(engine: SimulationEngine) -> dict:
    # Invoices recur (a new billing cycle starts each time the previous one
    # is paid -- see engine._issue_next_cycle), so "%paid" and
    # "mean_days_to_pay" are computed over all invoice cycles issued during
    # the window, not just one invoice per customer.
    stats = {
        a: {"n_customers": 0, "n_invoices": 0, "paid_days": [], "promises": [],
            "n_days_observed": 0, "n_spontaneous_responses": 0}
        for a in ARCHETYPE_NAMES
    }

    for c in engine.customers.values():
        stats[c.latent.archetype]["n_customers"] += 1

    for inv in engine.invoices:
        archetype = engine.customers[inv.customer_id].latent.archetype
        stats[archetype]["n_invoices"] += 1
        if inv.status == "paid" and inv.paid_day is not None:
            stats[archetype]["paid_days"].append(inv.paid_day - inv.issue_day)

    for rec in engine.realized_log:
        d = stats[rec.archetype]
        d["n_days_observed"] += 1
        # Under this policy every realized action is "none", so any
        # will_respond here is a *spontaneous*, unprompted response -- not
        # a reply to contact. See print_response_rate_note().
        if rec.outcome.will_respond:
            d["n_spontaneous_responses"] += 1
        if rec.outcome.will_promise and rec.outcome.promise_kept is not None:
            d["promises"].append((rec.day, rec.outcome.promise_kept))

    results = {}
    for archetype, d in stats.items():
        promises = d["promises"]
        results[archetype] = {
            "n_customers": d["n_customers"],
            "n_invoices": d["n_invoices"],
            "n_paid": len(d["paid_days"]),
            "pct_paid": len(d["paid_days"]) / d["n_invoices"] if d["n_invoices"] else 0.0,
            "mean_days_to_pay": float(np.mean(d["paid_days"])) if d["paid_days"] else float("nan"),
            "spontaneous_response_rate": (
                d["n_spontaneous_responses"] / d["n_days_observed"] if d["n_days_observed"] else float("nan")
            ),
            "n_promises": len(promises),
            "keep_rate": float(np.mean([k for _, k in promises])) if promises else float("nan"),
        }
    return results


def degrading_diagnostics(engine: SimulationEngine) -> None:
    """Prints exactly what was asked for after the first smoke-test run
    looked wrong: the actual sampled keep_probability_base/trend_slope
    values for the degrading archetype, and a fine-grained (15-day bucket)
    breakdown of realized keep-rate over the window.

    The original version of this test used a single day<45 vs day>=45
    split, and that made degrading look like it *started* at
    serial-promiser-like reliability (0.24) instead of declining from a
    reliable-always-late-like baseline (~0.80). It wasn't a bug in the
    decay mechanism -- the per-day math was correct throughout (verified
    below: keep-rate does start high and erode across the six buckets).
    The half-split was just too coarse: degrading customers' invoices stay
    open *longer* the more their pay probability has already eroded, so
    promise opportunities accumulate disproportionately in the back half of
    any window, and day 30-44 (already ~40% eroded per the trend formula)
    dominated the "first half" bucket, masking the true day-0 starting
    point. Finer buckets make the actual trajectory visible instead.
    """
    kbs, slopes = [], []
    for c in engine.customers.values():
        if c.latent.archetype == "degrading":
            kbs.append(c.latent.keep_probability_base)
            slopes.append(c.latent.trend_slope)
    kbs, slopes = np.array(kbs), np.array(slopes)

    print("\nDegrading archetype -- sampled latent parameters (n=%d customers):" % len(kbs))
    print(f"  keep_probability_base: mean={kbs.mean():.3f} std={kbs.std():.3f} "
          f"min={kbs.min():.3f} max={kbs.max():.3f}  (§4 target: {ARCHETYPES['degrading']['keep_probability_base']})")
    print(f"  trend_slope:           mean={slopes.mean():.5f} std={slopes.std():.5f} "
          f"min={slopes.min():.5f} max={slopes.max():.5f}  (§4 target: ~-0.01/day)")

    promises = [
        (rec.day, rec.outcome.promise_kept)
        for rec in engine.realized_log
        if rec.archetype == "degrading" and rec.outcome.will_promise and rec.outcome.promise_kept is not None
    ]
    days = np.array([p[0] for p in promises])
    kept = np.array([p[1] for p in promises])

    print("\nDegrading archetype -- realized keep-rate by day bucket:")
    print(f"  {'day range':<14}{'n':>5}{'keep_rate':>12}")
    bucket_rates = []
    for lo, hi in DEGRADING_BUCKETS:
        mask = (days >= lo) & (days < hi)
        n = int(mask.sum())
        rate = float(kept[mask].mean()) if n > 0 else float("nan")
        bucket_rates.append(rate)
        rate_str = f"{rate:.2f}" if not np.isnan(rate) else "n/a"
        print(f"  [{lo:>3},{hi:>3})    {n:>5}{rate_str:>12}")

    return bucket_rates


# ---------------------------------------------------------------------------
# Dispute lifecycle (Stage 13) -- disputer archetype vs. the other 6
# ---------------------------------------------------------------------------

def dispute_diagnostics(engine: SimulationEngine, n_days: int) -> dict:
    """CustomerBehaviorModel.dispute_outcome() is a pure function (no DB
    involved), so it can be checked directly against every invoice the
    "always none" run produced -- exactly like every other per-archetype
    check in this file, no separate DB-backed run needed."""
    stats = {a: {"n_invoices": 0, "n_disputed": 0, "n_resolved_within_window": 0} for a in ARCHETYPE_NAMES}
    for inv in engine.invoices:
        customer = engine.customers[inv.customer_id]
        archetype = customer.latent.archetype
        outcome = customer.dispute_outcome(inv.invoice_id, inv.issue_day)
        stats[archetype]["n_invoices"] += 1
        if outcome.will_dispute:
            stats[archetype]["n_disputed"] += 1
            if outcome.resolved_day is not None and outcome.resolved_day <= n_days:
                stats[archetype]["n_resolved_within_window"] += 1

    results = {}
    for archetype, d in stats.items():
        results[archetype] = {
            "n_invoices": d["n_invoices"],
            "n_disputed": d["n_disputed"],
            "dispute_rate": d["n_disputed"] / d["n_invoices"] if d["n_invoices"] else 0.0,
            "n_resolved_within_window": d["n_resolved_within_window"],
            "resolved_fraction_of_disputed": (
                d["n_resolved_within_window"] / d["n_disputed"] if d["n_disputed"] else float("nan")
            ),
        }
    return results


def print_dispute_table(results: dict) -> None:
    header = (
        f"{'archetype':<28}{'dispute_propensity':>19}{'n_invoices':>12}"
        f"{'n_disputed':>12}{'dispute_rate':>14}{'resolved_in_window':>20}"
    )
    print(header)
    print("-" * len(header))
    for archetype in ARCHETYPE_NAMES:
        r = results[archetype]
        target_dp = ARCHETYPES[archetype]["dispute_propensity"]
        resolved_str = f"{r['resolved_fraction_of_disputed']:.2f}" if r["n_disputed"] else "n/a"
        print(
            f"{archetype:<28}{target_dp:>19.2f}{r['n_invoices']:>12}"
            f"{r['n_disputed']:>12}{r['dispute_rate']:>14.3f}{resolved_str:>20}"
        )


# ---------------------------------------------------------------------------
# Run 2: always soft_reminder -- response-to-contact validation
# ---------------------------------------------------------------------------

def summarize_contact_run(engine: SimulationEngine) -> dict:
    stats = {a: {"n_contacts": 0, "n_responses": 0} for a in ARCHETYPE_NAMES}
    for rec in engine.realized_log:
        assert rec.action == "soft_reminder"  # this run's policy never returns anything else
        d = stats[rec.archetype]
        d["n_contacts"] += 1
        if rec.outcome.will_respond:
            d["n_responses"] += 1

    results = {}
    for archetype, d in stats.items():
        results[archetype] = {
            "n_contacts": d["n_contacts"],
            "response_rate": d["n_responses"] / d["n_contacts"] if d["n_contacts"] else float("nan"),
        }
    return results


def print_none_table(results: dict) -> None:
    header = (
        f"{'archetype':<28}{'n':>4}{'%paid':>8}{'mean_days_to_pay':>18}"
        f"{'spontaneous_resp':>18}{'n_promises':>12}{'keep_rate':>11}"
    )
    print(header)
    print("-" * len(header))
    for archetype in ARCHETYPE_NAMES:
        r = results[archetype]
        print(
            f"{archetype:<28}{r['n_customers']:>4}{r['pct_paid']*100:>7.1f}%"
            f"{r['mean_days_to_pay']:>18.2f}{r['spontaneous_response_rate']:>18.3f}"
            f"{r['n_promises']:>12}{r['keep_rate']:>11.2f}"
        )
    print(
        "\nNote: 'spontaneous_resp' is the rate of UNPROMPTED customer contact under "
        "action=none (no reminder was ever sent in this run) -- it measures the "
        "DAILY_SPONTANEOUS_RESPONSE_RATE mechanism, not response_propensity's primary "
        "meaning (reply-to-contact). response_propensity is validated separately below."
    )


def print_contact_table(results: dict) -> None:
    header = f"{'archetype':<28}{'response_propensity':>22}{'n_contacts':>12}{'response_rate':>16}"
    print(header)
    print("-" * len(header))
    for archetype in ARCHETYPE_NAMES:
        r = results[archetype]
        target_rp = ARCHETYPES[archetype]["response_propensity"]
        print(f"{archetype:<28}{target_rp:>22.2f}{r['n_contacts']:>12}{r['response_rate']:>16.3f}")


def run_assertions(none_results: dict, contact_results: dict, degrading_buckets: list[float],
                    dispute_results: dict) -> None:
    # --- Per-archetype target-range checks (not just pairwise) ---------
    for archetype in ARCHETYPE_NAMES:
        if archetype == "degrading":
            continue  # degrading's keep_rate is checked via the bucket trend below, not a single target
        r = none_results[archetype]
        target_keep = ARCHETYPES[archetype]["keep_probability_base"]
        assert abs(r["keep_rate"] - target_keep) <= KEEP_RATE_ABS_TOLERANCE, (
            f"{archetype}: observed keep_rate {r['keep_rate']:.2f} is more than "
            f"{KEEP_RATE_ABS_TOLERANCE} away from its §4 target {target_keep}"
        )

        target_days = ARCHETYPES[archetype]["avg_days_to_pay_mean"]
        lo, hi = DAYS_TO_PAY_REL_BAND[0] * target_days, DAYS_TO_PAY_REL_BAND[1] * target_days
        assert lo <= r["mean_days_to_pay"] <= hi, (
            f"{archetype}: observed mean_days_to_pay {r['mean_days_to_pay']:.2f} is outside "
            f"the [{lo:.1f}, {hi:.1f}] band around its §4 target {target_days}"
        )

    # --- Degrading: must start near the reliable-always-late baseline and
    #     visibly erode -- this is what the day-bucket diagnostic is for.
    early_rate = degrading_buckets[0]
    late_rate = degrading_buckets[-1]
    assert not np.isnan(early_rate), "degrading archetype produced no promise events in the first bucket (day 0-15)"
    baseline = ARCHETYPES["degrading"]["keep_probability_base"]
    assert early_rate >= baseline - 0.35, (
        f"degrading customers should start close to their {baseline} baseline; "
        f"observed day[0,15) keep_rate={early_rate:.2f} is too low even accounting for noise"
    )
    if not np.isnan(late_rate):
        assert late_rate < early_rate - 0.2, (
            f"degrading customers should show clear erosion by the end of the window: "
            f"day[0,15) keep_rate={early_rate:.2f} vs day[75,90) keep_rate={late_rate:.2f}"
        )

    # --- response_propensity, validated against actual contact this time ---
    rp_sorted_archetypes = sorted(ARCHETYPE_NAMES, key=lambda a: ARCHETYPES[a]["response_propensity"])
    lowest_rp, highest_rp = rp_sorted_archetypes[0], rp_sorted_archetypes[-1]
    assert lowest_rp == "non_responsive"
    assert contact_results[highest_rp]["response_rate"] > contact_results[lowest_rp]["response_rate"] + 0.10, (
        f"{highest_rp} (highest response_propensity) should show a clearly higher reply-to-contact "
        f"rate than {lowest_rp} (lowest): {contact_results[highest_rp]['response_rate']:.3f} vs "
        f"{contact_results[lowest_rp]['response_rate']:.3f}"
    )
    # serial-promiser: "responds readily, doesn't pay" -- high reply rate, low keep rate
    assert contact_results["serial_promiser"]["response_rate"] > contact_results["non_responsive"]["response_rate"], (
        "serial-promiser should reply to contact more often than non-responsive"
    )
    assert none_results["serial_promiser"]["keep_rate"] < none_results["model_citizen"]["keep_rate"], (
        "serial-promiser should still keep far fewer promises than model-citizen despite replying readily"
    )

    # --- Stage 13: disputer archetype must generate meaningfully more
    # disputes than every other archetype (dispute_rate, not raw count --
    # rate is what dispute_propensity actually governs; raw count would be
    # confounded by archetypes with different avg_days_to_pay issuing
    # different numbers of recurring invoice cycles over the window), and a
    # nonzero fraction of raised disputes must resolve within the window
    # (not permanently open). ---
    disputer_rate = dispute_results["disputer"]["dispute_rate"]
    other_rates = {a: dispute_results[a]["dispute_rate"] for a in ARCHETYPE_NAMES if a != "disputer"}
    highest_other = max(other_rates, key=other_rates.get)
    assert disputer_rate > other_rates[highest_other] * 3, (
        f"disputer archetype's dispute_rate ({disputer_rate:.3f}) should be at least 3x the highest "
        f"of every other archetype's ({highest_other}: {other_rates[highest_other]:.3f}) -- "
        f"§4's dispute_propensity=0.35 vs <=0.05 elsewhere implies a much larger gap than this"
    )
    for archetype, rate in other_rates.items():
        assert disputer_rate > rate, f"disputer archetype's dispute_rate ({disputer_rate:.3f}) should exceed {archetype}'s ({rate:.3f})"

    assert dispute_results["disputer"]["n_disputed"] > 0, "disputer archetype produced zero disputes -- dispute mechanism is broken"
    resolved_fraction = dispute_results["disputer"]["resolved_fraction_of_disputed"]
    assert resolved_fraction > 0.0, (
        "disputer archetype's disputes never resolve within the simulation window -- every disputed "
        "invoice would be permanently stuck in DISPUTE_UNRESOLVED, which is not a realistic dispute lifecycle"
    )
    assert resolved_fraction < 1.0 or dispute_results["disputer"]["n_disputed"] < 5, (
        "every single disputer-archetype dispute resolved within the window -- suspicious for a "
        "realistic lifecycle unless the disputed sample is too small to tell"
    )

    print("\nAll per-archetype target-range and contact-response assertions passed.")


def main() -> None:
    print(f"=== Run 1: always_none policy ({N_CUSTOMERS} customers, {N_DAYS} days, seed={SEED}) ===\n")
    none_customers = build_population(SEED, N_CUSTOMERS, id_prefix="none")
    none_engine = SimulationEngine(none_customers, seed=SEED)
    none_engine.run(N_DAYS, always_none_policy)
    none_results = summarize_none_run(none_engine)
    print_none_table(none_results)
    degrading_buckets = degrading_diagnostics(none_engine)

    print(f"\n\n=== Dispute lifecycle (Stage 13) -- {N_CUSTOMERS} customers, {N_DAYS}-day window ===\n")
    dispute_results = dispute_diagnostics(none_engine, N_DAYS)
    print_dispute_table(dispute_results)

    print(f"\n\n=== Run 2: always_soft_reminder policy ({N_CUSTOMERS} customers, {N_DAYS} days, seed={SEED}) ===\n")
    contact_customers = build_population(SEED, N_CUSTOMERS, id_prefix="contact")
    contact_engine = SimulationEngine(contact_customers, seed=SEED)
    contact_engine.run(N_DAYS, always_soft_reminder_policy)
    contact_results = summarize_contact_run(contact_engine)
    print_contact_table(contact_results)

    print()
    run_assertions(none_results, contact_results, degrading_buckets, dispute_results)


if __name__ == "__main__":
    main()
