"""
§4 archetype parameter table, encoded as data, plus per-customer latent-state
sampling.

Per-customer noise around each archetype's base values is drawn from Beta
distributions for the four probability-valued fields (keep_probability_base,
dispute_propensity, response_propensity, fatigue_sensitivity) and Normal
distributions for the two day/slope-valued fields (avg_days_to_pay,
trend_slope), matching §4's "draw from a Beta or Normal distribution
centered on these base values" instruction:
  - Beta for the [0,1]-bounded probabilities: it respects the bounds
    automatically (no post-hoc clipping) and its concentration parameter
    (kappa = alpha + beta) gives direct, interpretable control over spread
    -- higher kappa means tighter clustering around the mean.
  - Normal for avg_days_to_pay (an unbounded count of days) and trend_slope
    (a small signed drift), each clipped to a sane range after sampling.

fatigue_sensitivity is given only qualitatively in §4 (low/medium/high); it
is mapped to numeric base rates (0.15 / 0.35 / 0.60) below, then sampled the
same Beta way as the other [0,1] fields.
"""
from __future__ import annotations

import numpy as np

from simulator.behavior_model import CustomerLatentState

FATIGUE_LEVEL_BASE = {"low": 0.15, "medium": 0.35, "high": 0.60}

# Beta concentration (kappa = alpha + beta) per field family. Higher kappa =
# tighter spread around the base value. Chosen so within-archetype variation
# is visible without archetypes bleeding into each other (checked by the
# smoke test's distinctiveness assertions).
BETA_KAPPA_KEEP = 40.0
BETA_KAPPA_DISPUTE = 50.0
BETA_KAPPA_RESPONSE = 25.0
BETA_KAPPA_FATIGUE = 25.0

# §4 table, verbatim base values; only the noise-distribution shape/spread
# around them is an added implementation choice (documented above).
ARCHETYPES: dict[str, dict] = {
    "reliable_always_late": {
        "keep_probability_base": 0.85,
        "avg_days_to_pay_mean": 40.0, "avg_days_to_pay_std": 2.0,   # "fixed, low variance"
        "dispute_propensity": 0.02,
        "response_propensity": 0.70,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["low"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "cash_flow_strained_genuine": {
        "keep_probability_base": 0.65,
        "avg_days_to_pay_mean": 45.0, "avg_days_to_pay_std": 8.0,   # "35-55, high variance"
        "dispute_propensity": 0.05,
        "response_propensity": 0.60,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["medium"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "serial_promiser": {
        "keep_probability_base": 0.20,
        "avg_days_to_pay_mean": 57.5, "avg_days_to_pay_std": 8.0,   # "45-70"
        "dispute_propensity": 0.03,
        "response_propensity": 0.80,                                # "responds readily, doesn't pay"
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["low"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "disputer": {
        "keep_probability_base": 0.55,
        "avg_days_to_pay_mean": 40.0, "avg_days_to_pay_std": 4.0,   # "40, excl. dispute periods"
        "dispute_propensity": 0.35,
        "response_propensity": 0.50,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["medium"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "non_responsive": {
        "keep_probability_base": 0.40,
        "avg_days_to_pay_mean": 70.0, "avg_days_to_pay_std": 12.0,  # "50-90"
        "dispute_propensity": 0.02,
        "response_propensity": 0.15,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["high"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "model_citizen": {
        "keep_probability_base": 0.95,
        "avg_days_to_pay_mean": 28.0, "avg_days_to_pay_std": 2.0,   # "28, before due date"
        "dispute_propensity": 0.01,
        "response_propensity": 0.80,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["low"],
        "trend_slope_mean": 0.0, "trend_slope_std": 0.0008,
    },
    "degrading": {
        "keep_probability_base": 0.80,   # starting point; declines via trend_slope
        "avg_days_to_pay_mean": 30.0, "avg_days_to_pay_std": 3.0,   # starting point; increases via trend_slope
        "dispute_propensity": 0.05,
        "response_propensity": 0.60,
        "fatigue_sensitivity": FATIGUE_LEVEL_BASE["medium"],
        # "negative, e.g. -0.01/day" per §4; sampled per-customer and then
        # clipped to stay negative so every degrading customer actually
        # degrades (never accidentally drifts positive) over a run.
        "trend_slope_mean": -0.01, "trend_slope_std": 0.003,
        "trend_slope_clip": (-0.03, -0.002),
    },
}

ARCHETYPE_NAMES: tuple[str, ...] = tuple(ARCHETYPES.keys())


def _sample_beta(rng: np.random.Generator, mean: float, kappa: float) -> float:
    mean = min(max(mean, 1e-6), 1 - 1e-6)
    alpha = mean * kappa
    beta = (1 - mean) * kappa
    return float(rng.beta(alpha, beta))


def sample_customer_latent(customer_id: str, archetype: str, rng: np.random.Generator) -> CustomerLatentState:
    """Draw one customer's latent state from the named archetype's
    distribution. `rng` must be a seeded np.random.Generator supplied by the
    caller (e.g. the population-builder script), so the whole population is
    reproducible end to end.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype: {archetype}")
    p = ARCHETYPES[archetype]

    keep_probability_base = _sample_beta(rng, p["keep_probability_base"], BETA_KAPPA_KEEP)
    dispute_propensity = _sample_beta(rng, p["dispute_propensity"], BETA_KAPPA_DISPUTE)
    response_propensity = _sample_beta(rng, p["response_propensity"], BETA_KAPPA_RESPONSE)
    fatigue_sensitivity = _sample_beta(rng, p["fatigue_sensitivity"], BETA_KAPPA_FATIGUE)

    avg_days_to_pay = float(rng.normal(p["avg_days_to_pay_mean"], p["avg_days_to_pay_std"]))
    avg_days_to_pay = max(avg_days_to_pay, 1.0)

    trend_slope = float(rng.normal(p["trend_slope_mean"], p["trend_slope_std"]))
    if "trend_slope_clip" in p:
        lo, hi = p["trend_slope_clip"]
        trend_slope = min(max(trend_slope, lo), hi)

    return CustomerLatentState(
        customer_id=customer_id,
        archetype=archetype,
        keep_probability_base=keep_probability_base,
        avg_days_to_pay=avg_days_to_pay,
        dispute_propensity=dispute_propensity,
        response_propensity=response_propensity,
        fatigue_sensitivity=fatigue_sensitivity,
        trend_slope=trend_slope,
    )
