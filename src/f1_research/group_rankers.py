"""Native learning-to-rank challengers for whole-race finishing order.

These models are deliberately optional. They operate on complete event groups and
return lower-is-better scores so they can be evaluated by the existing Plackett-
Luce probability/calibration layer without changing score semantics.
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

from .features import CATEGORICAL, FEATURES, NUMERIC

RANKING_REFERENCES = {
    "xgboost": "https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-learning-to-rank",
    "lightgbm": "https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#lambdarank",
    "catboost": "https://catboost.ai/docs/en/concepts/loss-functions-ranking",
}


@dataclass(frozen=True)
class RankingSpec:
    name: str
    params: dict[str, Any]


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


def group_arrays(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int], np.ndarray]:
    required = set(FEATURES) | {"event_id", "finish_position"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    ordered = frame.copy()
    sort_columns = ["event_id", "finish_position"]
    if "date" in ordered:
        sort_columns = ["date", "event_id", "finish_position"]
    ordered = ordered.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if ordered["finish_position"].isna().any():
        raise ValueError("Ranking training cannot contain unknown finish positions")

    group_sizes: list[int] = []
    relevance = np.empty(len(ordered), dtype=float)
    group_ids = np.empty(len(ordered), dtype=int)
    offset = 0
    for group_index, (_, event) in enumerate(ordered.groupby("event_id", sort=False)):
        positions = pd.to_numeric(event["finish_position"], errors="raise").to_numpy(dtype=float)
        n = len(event)
        if n < 2 or len(np.unique(positions)) != n:
            raise ValueError("Each ranking event requires a complete unique order")
        if sorted(positions.tolist()) != [float(value) for value in range(1, n + 1)]:
            raise ValueError("Ranking event positions must be consecutive from 1..N")
        group_sizes.append(n)
        relevance[offset : offset + n] = n + 1.0 - positions
        group_ids[offset : offset + n] = group_index
        offset += n
    return ordered, relevance, group_sizes, group_ids


class NativeGroupRanker:
    """Adapter for XGBoost LambdaMART, LightGBM LambdaRank and CatBoost YetiRank."""

    def __init__(self, spec: RankingSpec, random_state: int = 42):
        self.spec = spec
        self.random_state = int(random_state)
        self.preprocess = _preprocessor()
        self.model: Any | None = None

    def fit(self, frame: pd.DataFrame) -> "NativeGroupRanker":
        ordered, target, group_sizes, group_ids = group_arrays(frame)
        x = np.asarray(self.preprocess.fit_transform(ordered[FEATURES]), dtype=float)
        params = dict(self.spec.params)

        if self.spec.name == "xgboost_rank_ndcg":
            try:
                from xgboost import XGBRanker
            except ImportError as exc:
                raise RuntimeError("Install the modern extra to evaluate XGBoost ranking") from exc
            defaults = {
                "objective": "rank:ndcg",
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 8.0,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 8.0,
                "lambdarank_pair_method": "topk",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            defaults.update(params)
            self.model = XGBRanker(**defaults)
            self.model.fit(x, target, group=group_sizes)
        elif self.spec.name == "lightgbm_lambdarank":
            try:
                from lightgbm import LGBMRanker
            except ImportError as exc:
                raise RuntimeError("Install the modern extra to evaluate LightGBM ranking") from exc
            defaults = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "n_estimators": 500,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "reg_lambda": 8.0,
                "random_state": self.random_state,
                "verbosity": -1,
                "n_jobs": -1,
            }
            defaults.update(params)
            self.model = LGBMRanker(**defaults)
            self.model.fit(x, target, group=group_sizes)
        elif self.spec.name == "catboost_yetirank":
            try:
                from catboost import CatBoostRanker, Pool
            except ImportError as exc:
                raise RuntimeError("Install the modern extra to evaluate CatBoost ranking") from exc
            defaults = {
                "loss_function": "YetiRankPairwise",
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 7,
                "l2_leaf_reg": 8.0,
                "random_seed": self.random_state,
                "verbose": False,
                "allow_writing_files": False,
            }
            defaults.update(params)
            self.model = CatBoostRanker(**defaults)
            self.model.fit(Pool(x, label=target, group_id=group_ids))
        else:
            raise ValueError(f"Unknown native ranking model: {self.spec.name}")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NativeGroupRanker must be fitted before prediction")
        missing = set(FEATURES) - set(frame)
        if missing:
            raise ValueError(f"Missing ranking features: {sorted(missing)}")
        x = np.asarray(self.preprocess.transform(frame[FEATURES]), dtype=float)
        raw = np.asarray(self.model.predict(x), dtype=float)
        if raw.shape != (len(frame),) or not np.isfinite(raw).all():
            raise ValueError("Native ranking model produced invalid scores")
        return -raw
