"""Modern tabular challengers evaluated with whole-event forward validation.

Optional libraries are lazy imports. A model only earns a place in reports when it
beats baselines on held-out F1 events; availability or benchmark hype is not treated
as evidence of F1 performance.

Hyperparameter selection uses walk-forward predictions from the designated tuning
block. Because the product exposes race probabilities, winner log loss is the default
selection metric; rank MAE is retained as a transparent secondary metric. A single
scalar temperature is optimized on the tuning predictions for candidate selection,
then discarded. Final probability calibration is repeated independently on the later
outer calibration block in ``evaluation_v2``.
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

TUNING_TEMPERATURES = np.geomspace(0.02, 3.0, 45)
SELECTION_METRICS = {"winner_log_loss", "position_mae"}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    params: dict[str, Any]


@dataclass
class CandidateScore:
    name: str
    params: dict[str, Any]
    mean_winner_log_loss: float
    mean_position_mae: float
    tuning_temperature: float
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


class _CatBoostNativeRegressor:
    """Small adapter that preserves CatBoost's native categorical feature handling.

    Other tree challengers receive ordinal-encoded categories because their sklearn
    interfaces expect numeric matrices. CatBoost should not: its ordered target/stat
    machinery is specifically designed to consume categorical columns directly. This
    adapter therefore performs only leakage-safe numeric median imputation and string
    normalization for categorical values, then passes the original feature names to
    CatBoost via ``cat_features``.
    """

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42):
        self.params = dict(params or {})
        self.random_state = int(random_state)
        self.numeric_fill_: dict[str, float] | None = None
        self.model_: Any | None = None

    def _prepare(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("CatBoost native adapter requires a pandas DataFrame")
        missing = [name for name in FEATURES if name not in frame]
        if missing:
            raise ValueError(f"Missing CatBoost features: {missing}")
        output = frame[FEATURES].copy()
        numeric = output[NUMERIC].apply(pd.to_numeric, errors="coerce")
        if fit:
            medians = numeric.median(axis=0, skipna=True).fillna(0.0)
            self.numeric_fill_ = {name: float(medians[name]) for name in NUMERIC}
        if self.numeric_fill_ is None:
            raise RuntimeError("CatBoost adapter must be fitted before prediction")
        for name in NUMERIC:
            output[name] = numeric[name].fillna(self.numeric_fill_[name]).astype(float)
        for name in CATEGORICAL:
            # CatBoost categorical values cannot contain float NaN. StringDtype keeps
            # missing values explicit before conversion and also preserves unseen
            # categories at prediction time instead of mapping them to an arbitrary id.
            output[name] = output[name].astype("string").fillna("__MISSING__").astype(str)
        return output

    def fit(self, frame: pd.DataFrame, target: Any) -> "_CatBoostNativeRegressor":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("catboost is not installed; install .[modern]") from exc
        settings = _with_overrides({
            "loss_function": "MAE",
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": self.random_state,
            "cat_features": list(CATEGORICAL),
        }, self.params)
        prepared = self._prepare(frame, fit=True)
        self.model_ = CatBoostRegressor(**settings)
        self.model_.fit(prepared, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CatBoost adapter must be fitted before prediction")
        prepared = self._prepare(frame, fit=False)
        return np.asarray(self.model_.predict(prepared), dtype=float)


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


def _with_overrides(defaults: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(params)
    return merged


def build_estimator(spec: CandidateSpec, random_state: int = 42) -> Any:
    name, params = spec.name, dict(spec.params)
    if name == "catboost":
        # Keep CatBoost on the raw named DataFrame so it can use native categorical
        # statistics. OrdinalEncoder is deliberately bypassed for this challenger.
        return _CatBoostNativeRegressor(params, random_state)
    if name == "hist_gradient_boosting":
        settings = _with_overrides({
            "loss": "absolute_error", "random_state": random_state, "early_stopping": True,
            "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "l2_regularization": 5.0, "min_samples_leaf": 12,
        }, params)
        learner = HistGradientBoostingRegressor(**settings)
    elif name == "extra_trees":
        settings = _with_overrides({
            "n_estimators": 400, "min_samples_leaf": 4, "max_features": 0.8,
            "n_jobs": -1, "random_state": random_state,
        }, params)
        learner = ExtraTreesRegressor(**settings)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; install .[modern]") from exc
        settings = _with_overrides({
            "objective": "reg:squarederror", "n_estimators": 500, "learning_rate": 0.03,
            "max_depth": 3, "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_lambda": 5.0, "n_jobs": -1, "random_state": random_state,
        }, params)
        learner = XGBRegressor(**settings)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed; install .[modern]") from exc
        settings = _with_overrides({
            "objective": "regression_l1", "n_estimators": 500, "learning_rate": 0.03,
            "num_leaves": 15, "min_child_samples": 20, "reg_lambda": 5.0,
            "verbosity": -1, "random_state": random_state,
        }, params)
        learner = LGBMRegressor(**settings)
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


def _winner_log_loss(event: pd.DataFrame, scores: np.ndarray, temperature: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(event) or not np.isfinite(scores).all():
        raise ValueError("Model produced invalid probability scores")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    winner = event["finish_position"].to_numpy(dtype=float) == 1
    if winner.sum() != 1:
        raise ValueError("Tuning event must contain exactly one winner")
    logits = -scores / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    return float(-np.log(max(float(probabilities[winner][0]), 1e-15)))


def _select_tuning_temperature(
    predictions: list[tuple[pd.DataFrame, np.ndarray]],
    temperatures: np.ndarray = TUNING_TEMPERATURES,
) -> tuple[float, float]:
    if not predictions:
        raise ValueError("No tuning predictions available for temperature selection")
    losses = []
    for temperature in temperatures:
        event_losses = [
            _winner_log_loss(event, scores, float(temperature))
            for event, scores in predictions
        ]
        losses.append(float(np.mean(event_losses)))
    best_index = int(np.argmin(losses))
    return float(temperatures[best_index]), float(losses[best_index])


def tune_forward_events(
    frame: pd.DataFrame,
    specs: list[CandidateSpec] | None = None,
    tuning_events: int = 6,
    min_fit_events: int = 12,
    max_specs_per_model: int | None = None,
    selection_metric: str = "winner_log_loss",
) -> tuple[CandidateSpec, pd.DataFrame]:
    """Select hyperparameters only from the designated past tuning block.

    Each tuning event is predicted by a model fit strictly before that event. Candidate
    temperature and candidate ranking are both selected using only these tuning-block
    predictions. The resulting temperature is *not* reused for the final model; the
    later outer calibration block independently calibrates the selected candidate.

    The tuning log loss is a model-selection score, not an unbiased generalization
    estimate, because the scalar temperature is optimized on the same tuning block.
    The sealed outer test remains untouched and is the evidence-bearing evaluation.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
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
            scores.append(CandidateScore(
                spec.name,
                spec.params,
                np.inf,
                np.inf,
                np.nan,
                0,
                False,
                "optional dependency is not installed",
            ))
            continue
        event_predictions: list[tuple[pd.DataFrame, np.ndarray]] = []
        event_rank_losses: list[float] = []
        try:
            for event_id in tune_ids:
                target_event = frame[frame.event_id == event_id]
                cutoff = target_event.date.min()
                train = frame[frame.date < cutoff]
                if train.event_id.nunique() < min_fit_events:
                    continue
                model = build_estimator(spec)
                model.fit(train[FEATURES], normalized_rank_target(train))
                predicted_scores = np.asarray(model.predict(target_event[FEATURES]), dtype=float)
                event_rank_losses.append(_rank_mae(target_event, predicted_scores))
                event_predictions.append((target_event, predicted_scores))
            if not event_predictions:
                raise ValueError("candidate had no eligible tuning events")
            temperature, winner_loss = _select_tuning_temperature(event_predictions)
            scores.append(CandidateScore(
                spec.name,
                spec.params,
                winner_loss,
                float(np.mean(event_rank_losses)),
                temperature,
                len(event_predictions),
            ))
        except (ValueError, RuntimeError, MemoryError) as exc:
            scores.append(CandidateScore(
                spec.name,
                spec.params,
                np.inf,
                float(np.mean(event_rank_losses)) if event_rank_losses else np.inf,
                np.nan,
                len(event_predictions),
                True,
                str(exc),
            ))

    valid = [
        item for item in scores
        if np.isfinite(item.mean_position_mae) and np.isfinite(item.mean_winner_log_loss)
    ]
    if not valid:
        raise ValueError("No modern candidate completed forward-event tuning")
    if selection_metric == "winner_log_loss":
        best_score = min(
            valid,
            key=lambda item: (item.mean_winner_log_loss, item.mean_position_mae, item.name),
        )
        sort_columns = ["mean_winner_log_loss", "mean_position_mae", "name"]
    else:
        best_score = min(
            valid,
            key=lambda item: (item.mean_position_mae, item.mean_winner_log_loss, item.name),
        )
        sort_columns = ["mean_position_mae", "mean_winner_log_loss", "name"]
    best = CandidateSpec(best_score.name, best_score.params)
    table = pd.DataFrame([
        {**asdict(item), "params": jsonable_params(item.params)} for item in scores
    ]).sort_values(sort_columns, na_position="last").reset_index(drop=True)
    table.attrs["selection_metric"] = selection_metric
    return best, table


def jsonable_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={params[key]}" for key in sorted(params)) or "default"


def fit_selected(frame: pd.DataFrame, spec: CandidateSpec) -> Any:
    model = build_estimator(spec)
    model.fit(frame[FEATURES], normalized_rank_target(frame))
    return model
