"""
models/propensity_model.py — load/predict wrapper (module-interfaces
contract: "models exposes PropensityModel.predict_proba(feature_vector) ->
float. Pure function, no DB access.").
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "propensity_model.joblib"


class PropensityModel:
    """Loads a trained LightGBM S-learner (models/train.py) and scores a
    single feature vector. No DB access anywhere in this class -- the
    artifact bundles everything needed (model, categorical-column
    categories captured at training time, expected column order).
    """

    def __init__(self, artifact_path: Optional[Path] = None):
        artifact = joblib.load(artifact_path or DEFAULT_ARTIFACT_PATH)
        self.model = artifact["model"]
        self.categories: dict[str, list] = artifact["categories"]
        self.feature_columns: list[str] = artifact["feature_columns"]

    def predict_proba(self, feature_vector: dict) -> float:
        """feature_vector must contain (at least) every key in
        FEATURE_COLUMNS (features/feature_engine.py). Extra keys are
        ignored; categorical columns are re-encoded against the exact
        category set seen at training time so an unseen category becomes a
        missing value to LightGBM rather than erroring.
        """
        row = {col: feature_vector[col] for col in self.feature_columns}
        df = pd.DataFrame([row])
        for col, cats in self.categories.items():
            df[col] = pd.Categorical(df[col], categories=cats)
        if "active_promise_flag" in df.columns:
            df["active_promise_flag"] = df["active_promise_flag"].astype(int)
        proba = self.model.predict_proba(df)[:, 1][0]
        return float(proba)
