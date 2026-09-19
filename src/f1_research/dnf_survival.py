"""Censoring-aware DNF survival challengers.

The target is race-progress-to-retirement. Finishers are right-censored at the
race end rather than being dropped, so the model can be evaluated without the
classic survivor-only selection mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import validate
from .features import CATEGORICAL, FEATURES, NUMERIC, build_features

SURVIVAL_REFERENCES = {
    "xgboost_aft": (
        "https://xgboost.readthedocs.io/en/stable/python/"
        "survival-examples/aft_survival_demo.html"
    ),
    "deephit_competing_risks": "https://doi.org/10.1609/aaai.v32i1.11842",
}


@dataclass(frozen=True)
class AFTTargets:
    lower: np.ndarray
    upper: np.ndarray
    observed: np.ndarray
    progress: np.ndarray


def build_aft_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, AFTTargets]:
    validated = validate(frame)
    required = {"laps_completed", "race_total_laps", "dnf"}
    missing = required - set(validated)
    if missing:
        raise ValueError(
            "DNF survival requires source lap exposure columns: "
            f"{sorted(missing)}"
        )
    featured = build_features(validated)
    completed = pd.to_numeric(featured["laps_completed"], errors="raise").to_numpy(dtype=float)
    total = pd.to_numeric(featured["race_total_laps"], errors="raise").to_numpy(dtype=float)
    observed = pd.to_numeric(featured["dnf"], errors="raise").to_numpy(dtype=int).astype(bool)
    if np.any(total <= 0) or np.any(completed < 0) or np.any(completed > total):
        raise ValueError("Invalid lap exposure for DNF survival")

    # Avoid zero-duration AFT labels for lap-0 incidents while preserving ordering.
    progress = np.clip(np.maximum(completed, 0.5) / total, 1e-6, 1.0)
    lower = np.where(observed, progress, 1.0)
    upper = np.where(observed, progress, np.inf)
    return featured, AFTTargets(
        lower=lower.astype(float),
        upper=upper.astype(float),
        observed=observed,
        progress=progress.astype(float),
    )


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


class XGBoostAFTDNF:
    """XGBoost accelerated-failure-time model for DNF risk ordering."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(params or {})
        self.preprocess = _preprocessor()
        self.model: Any | None = None

    def fit(self, frame: pd.DataFrame) -> "XGBoostAFTDNF":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Install the modern extra to evaluate XGBoost AFT") from exc
        featured, targets = build_aft_frame(frame)
        x = np.asarray(self.preprocess.fit_transform(featured[FEATURES]), dtype=float)
        matrix = xgb.DMatrix(x)
        matrix.set_float_info("label_lower_bound", targets.lower)
        matrix.set_float_info("label_upper_bound", targets.upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 8.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 8.0,
            "alpha": 0.05,
            "aft_loss_distribution": "logistic",
            "aft_loss_distribution_scale": 1.0,
            "seed": 42,
        }
        params.update(self.params)
        rounds = int(params.pop("num_boost_round", 500))
        self.model = xgb.train(params, matrix, num_boost_round=rounds)
        return self

    def predict_survival_progress(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostAFTDNF must be fitted before prediction")
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Install the modern extra to evaluate XGBoost AFT") from exc
        missing = set(FEATURES) - set(frame)
        if missing:
            raise ValueError(f"Missing DNF survival features: {sorted(missing)}")
        x = np.asarray(self.preprocess.transform(frame[FEATURES]), dtype=float)
        values = np.asarray(self.model.predict(xgb.DMatrix(x)), dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError("XGBoost AFT produced invalid survival predictions")
        return values

    def predict_risk_score(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.predict_survival_progress(frame)


def concordance_index(targets: AFTTargets, risk_score: np.ndarray) -> float:
    """Harrell-style concordance using only comparable censored pairs."""
    risk = np.asarray(risk_score, dtype=float)
    if risk.shape != targets.progress.shape or not np.isfinite(risk).all():
        raise ValueError("Invalid survival risk scores")
    concordant = 0.0
    comparable = 0
    for i in range(len(risk)):
        if not targets.observed[i]:
            continue
        for j in range(len(risk)):
            if i == j or targets.progress[i] >= targets.progress[j]:
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("No comparable survival pairs")
    return float(concordant / comparable)
