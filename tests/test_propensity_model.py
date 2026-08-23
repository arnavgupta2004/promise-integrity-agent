"""
tests/test_propensity_model.py — unit tests for models/propensity_model.py
plus a sanity check that the logging policy (models/train.py) produces a
non-degenerate action distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from features.feature_engine import FEATURE_COLUMNS
from models.propensity_model import DEFAULT_ARTIFACT_PATH, PropensityModel
from models.train import TRAINING_DATA_CSV

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.exists(),
    reason="model artifact not found; run `python3 models/train.py` first",
)


def make_feature_vector(**overrides) -> dict:
    base = {
        "relative_lateness": 0.5,
        "prs_score": 0.5,
        "prs_trend": 0.0,
        "dispute_rate": 0.0,
        "response_rate": 0.0,
        "partial_payment_rate": 0.0,
        "amount_tier": 2,
        "days_since_last_contact": 9999,
        "active_promise_flag": False,
        "days_until_promised_date": -1,
        "broken_promise_streak": 0,
        "segment": "SMB",
        "intervention_type": "none",
    }
    base.update(overrides)
    return base


SAMPLE_FEATURE_VECTORS = [
    make_feature_vector(),  # neutral / no-history-like row
    make_feature_vector(relative_lateness=2.9, prs_score=0.1, broken_promise_streak=3, intervention_type="channel_escalation"),
    make_feature_vector(relative_lateness=0.1, prs_score=0.95, response_rate=0.8, intervention_type="none"),
    make_feature_vector(relative_lateness=1.5, prs_score=0.3, dispute_rate=0.4, intervention_type="plan_proposal"),
    make_feature_vector(active_promise_flag=True, days_until_promised_date=4, intervention_type="soft_reminder"),
]


class TestPropensityModelLoadsAndPredicts:
    def test_model_file_loads(self):
        model = PropensityModel()
        assert model.model is not None
        assert set(model.feature_columns) == set(FEATURE_COLUMNS)

    @pytest.mark.parametrize("feature_vector", SAMPLE_FEATURE_VECTORS)
    def test_predict_proba_in_unit_interval(self, feature_vector):
        model = PropensityModel()
        p = model.predict_proba(feature_vector)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_predict_proba_varies_by_intervention_type(self):
        model = PropensityModel()
        fv = make_feature_vector(relative_lateness=1.0, prs_score=0.6)
        probs = {
            action: model.predict_proba({**fv, "intervention_type": action})
            for action in ["none", "soft_reminder", "firm_reminder", "channel_escalation", "link_resend", "plan_proposal"]
        }
        # not asserting a specific ordering (that's what the eval harness is
        # for) -- just that intervention_type actually moves the prediction,
        # i.e. the model didn't learn to ignore its own treatment feature
        assert len(set(probs.values())) > 1


class TestLoggingPolicyActionDistribution:
    """§7's DoD: the logging policy's action distribution must not be
    degenerate (no single action dominating >90% of assignments).
    """

    @pytest.mark.skipif(not TRAINING_DATA_CSV.exists(), reason="training data not found; run models/train.py first")
    def test_action_distribution_not_degenerate(self):
        df = pd.read_csv(TRAINING_DATA_CSV)
        dist = df["intervention_type"].value_counts(normalize=True)
        assert dist.max() < 0.90, f"{dist.idxmax()} accounts for {dist.max():.1%} of assignments"

    @pytest.mark.skipif(not TRAINING_DATA_CSV.exists(), reason="training data not found; run models/train.py first")
    def test_all_logged_actions_appear(self):
        df = pd.read_csv(TRAINING_DATA_CSV)
        from models.train import LOGGED_ACTIONS
        assert set(df["intervention_type"].unique()) == set(LOGGED_ACTIONS)

    @pytest.mark.skipif(not TRAINING_DATA_CSV.exists(), reason="training data not found; run models/train.py first")
    def test_intervention_type_not_perfectly_confounded_with_risk_tier(self):
        df = pd.read_csv(TRAINING_DATA_CSV)
        joint = pd.crosstab(df["risk_tier"], df["intervention_type"])
        # every risk tier should see at least 2 distinct actions with
        # non-trivial counts -- a perfectly confounded assignment would
        # show each tier mapped to (essentially) one action only
        for tier in joint.index:
            nonzero = (joint.loc[tier] > 0).sum()
            assert nonzero >= 2, f"risk_tier={tier} only ever saw {nonzero} distinct action(s) -- looks confounded"
