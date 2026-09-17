"""Modern tabular challengers evaluated with whole-event forward validation.

Optional libraries are lazy imports.  A model only earns a place in reports when it
beats baselines on held-out F1 events; availability or benchmark hype is not treated
as evidence of F1 performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, NUMERIC


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    params: dict[str, Any]


@dataclass
class CandidateScore:
    name: str
    params: dict[str, Any]
    mean_position_mae: float
    events: int
    available: bool = True
    error: str | None = None


def normalized_rank_target(frame: pd.DataFrame) -> np.ndarray:
    size = frame.groupby("event_id")["driver"].transform("size")
    return ((frame["finish_position"] - 1) / (size - 1).clip(lower=1)).to_numpy(dtype=float)


def _preprocess(scale: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("category", categorical, CATEGORICAL)])


def available_candidates() -> dict[str, bool]:
    result = {"hist_gradient_boosting": True, "extra_trees": True}
    for name, module in (("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                         ("catboost", "catboost"), ("tabicl_v2", "tabicl")):
        try:
            __import__(module)
            result[name] = True
        except ImportError:
            result[name] = False
    return result


def build_estimator(spec: CandidateSpec, random_state: int = 42) -> Pipeline:
    name, params = spec.name, dict(spec.params)
    if name == "hist_gradient_boosting":
        learner = HistGradientBoostingRegressor(
            loss="huber", random_state=random_state, early_stopping=True, **params)
    elif name == "extra_trees":
        learner = ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=4, max_features=0.8,
            n_jobs=-1, random_state=random_state, **params)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; install .[modern]") from exc
        learner = XGBRegressor(
            objective="reg:squarederror", n_estimators=500, learning_rate=0.03,
            max_depth=3, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5.0, n_jobs=-1, random_state=random_state, **params)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed; install .[modern]") from exc
        learner = LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.03,
            num_leaves=15, min_child_samples=20, reg_lambda=5.0,
            verbosity=-1, random_state=random_state, **params)
    elif name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("catboost is not installed; install .[modern]") from exc
        learner = CatBoostRegressor(
            loss_function="MAE", iterations=500, depth=5, learning_rate=0.03,
            l2_leaf_reg=5.0, verbose=False, random_seed=random_state, **params)
    elif name == "tabicl_v2":
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise RuntimeError("TabICLv2 is not installed; install .[foundation]") from exc
        learner = TabICLRegressor(**params)
    else:
        raise ValueError(f"Unknown model candidate: {name}")
    return Pipeline([("preprocess", _preprocess(scale=False)), ("model", learner)])


def candidate_specs(names: tuple[str, ...] | None = None) -> list[CandidateSpec]:
    names = names or tuple(available_candidates())
    grids: dict[str, list[dict[str, Any]]] = {
        "hist_gradient_boosting": [
            {"max_iter": 150, "learning_rate": lr, "max_leaf_nodes": leaves,
             "l2_regularization": l2, "min_samples_leaf": 12}
            for lr, leaves, l2 in product((0.03, 0.06), (7, 15), (1.0, 5.0))
        ],
        "extra_trees": [
            {"min_samples_leaf": leaf, "max_features": feature}
            for leaf, feature in product((3, 6, 10), (0.6, 0.9))
        ],
        "xgboost": [
            {"max_depth": depth, "min_child_weight": child, "reg_alpha": alpha}
            for depth, child, alpha in product((2, 3), (5, 15), (0.0, 0.5))
        ],
        "lightgbm": [
            {"num_leaves": leaves, "min_child_samples": child, "reg_alpha": alpha}
            for leaves, child, alpha in product((7, 15), (15, 30), (0.0, 0.5))
        ],
        "catboost": [
            {"depth": depth, "random_strength": strength}
            for depth, strength in product((4, 6), (0.5, 1.5))
        ],
        # Foundation model: no F1-specific hyperparameter search by design.
        "tabicl_v2": [{}],
    }
    output = []
    for name in names:
        for params in grids.get(name, []):
            output.append(CandidateSpec(name, params))
    return output


def _rank_mae(event: pd.DataFrame, scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(event) or not np.isfinite(scores).all():
        raise ValueError("Model produced invalid rank scores")
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    truth = event["finish_position"].to_numpy(dtype=float)
    return float(np.mean(np.abs(ranks - truth)))


def tune_forward_events(frame: pd.DataFrame, specs: list[CandidateSpec] | None = None,
                        tuning_events: int = 6, min_fit_events: int = 12,
                        max_specs_per_model: int | None = None) -> tuple[CandidateSpec, pd.DataFrame]:
    """Select hyperparameters only from past whole events.

    For each tuning event the estimator is refit using only strictly earlier rows.
    The returned winner must still be probability-calibrated on a later disjoint
    block before it can be evaluated on an outer test event.
    """
    required = set(FEATURES) | {"event_id", "date", "finish_position", "driver"}
    if required - set(frame):
        raise ValueError(f"Missing tuning columns: {sorted(required - set(frame))}")
    events = frame[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    if tuning_events < 1 or len(events) < min_fit_events + tuning_events:
        raise ValueError("Not enough events for disjoint fit and tuning blocks")
    tune_ids = events.iloc[-tuning_events:].event_id.tolist()
    availability = available_candidates()
    specs = specs or candidate_specs()
    if max_specs_per_model is not None:
        kept, counts = [], {}
        for spec in specs:
            counts[spec.name] = counts.get(spec.name, 0) + 1
            if counts[spec.name] <= max_specs_per_model:
                kept.append(spec)
        specs = kept

    scores: list[CandidateScore] = []
    for spec in specs:
        if not availability.get(spec.name, False):
            scores.append(CandidateScore(spec.name, spec.params, np.inf, 0, False,
                                         "optional dependency is not installed"))
            continue
        event_losses = []
        try:
            for event_id in tune_ids:
                target_event = frame[frame.event_id == event_id]
                cutoff = target_event.date.min()
                train = frame[frame.date < cutoff]
                if train.event_id.nunique() < min_fit_events:
                    continue
                estimator = build_estimator(spec)
                estimator.fit(train[FEATURES], normalized_rank_target(train))
                event_losses.append(_rank_mae(target_event, estimator.predict(target_event[FEATURES])))
            if not event_losses:
                raise ValueError("candidate had no eligible tuning events")
            scores.append(CandidateScore(spec.name, spec.params, float(np.mean(event_losses)),
                                         len(event_losses)))
        except (ValueError, RuntimeError, MemoryError) as exc:
            scores.append(CandidateScore(spec.name, spec.params, np.inf, len(event_losses), True, str(exc)))

    valid = [item for item in scores if np.isfinite(item.mean_position_mae)]
    if not valid:
        raise ValueError("No modern candidate completed forward-event tuning")
    best_score = min(valid, key=lambda item: (item.mean_position_mae, item.name))
    best = CandidateSpec(best_score.name, best_score.params)
    table = pd.DataFrame([
        {**asdict(item), "params": jsonable_params(item.params)} for item in scores
    ]).sort_values(["mean_position_mae", "name"], na_position="last").reset_index(drop=True)
    return best, table


def jsonable_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={params[key]}" for key in sorted(params)) or "default"


def fit_selected(frame: pd.DataFrame, spec: CandidateSpec) -> Pipeline:
    model = build_estimator(spec)
    model.fit(frame[FEATURES], normalized_rank_target(frame))
    return model
