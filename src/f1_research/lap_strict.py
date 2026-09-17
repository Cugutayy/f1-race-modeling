"""Leakage-strict next-lap mixture model.

Historical OpenF1 stint rows do not expose an as-published timestamp, therefore
compound, tyre age and stint number are intentionally excluded from this module.
The strict artifact is the default evidence-grade live pace model; richer stint
features remain available in the exploratory pipeline but cannot improve the sealed
benchmark by using retrospectively joined state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .lap_intelligence import LapModelSpec, live_feature_rows

STRICT_NUMERIC = [
    "lap_number",
    "last_lap_s",
    "recent_median_3_s",
    "recent_median_5_s",
    "recent_trend_s_per_lap",
    "recent_variability_s",
    "pit_stops_before",
    "air_temperature_c",
    "track_temperature_c",
    "humidity_pct",
    "rainfall",
    "safety_car_active",
    "yellow_recent",
]
STRICT_CATEGORICAL = ["driver_number"]
STRICT_FEATURES = STRICT_NUMERIC + STRICT_CATEGORICAL
REGIMES = ("green", "neutralized", "pit")


@dataclass(frozen=True)
class StrictBenchmark:
    test_session: int
    selected_regressor: str
    green_rows: int
    green_mae_s: float
    green_rmse_s: float
    recent_median_mae_s: float
    last_lap_mae_s: float
    interval_coverage: float
    interval_mean_width_s: float
    regime_accuracy: float
    regime_log_loss: float


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, STRICT_NUMERIC),
        ("categorical", categorical, STRICT_CATEGORICAL),
    ])


def build_strict_regressor(spec: LapModelSpec) -> Pipeline:
    params = dict(spec.params)
    if spec.name == "hist_gradient_boosting":
        defaults = {
            "loss": "absolute_error",
            "max_iter": 220,
            "learning_rate": 0.035,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 5.0,
            "random_state": 42,
        }
        defaults.update(params)
        model = HistGradientBoostingRegressor(**defaults)
    elif spec.name == "extra_trees":
        defaults = {
            "n_estimators": 600,
            "min_samples_leaf": 4,
            "max_features": 0.8,
            "n_jobs": -1,
            "random_state": 42,
        }
        defaults.update(params)
        model = ExtraTreesRegressor(**defaults)
    elif spec.name == "tabicl_v2":
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise RuntimeError("Install the foundation extra to evaluate TabICLv2") from exc
        model = TabICLRegressor(**params)
    else:
        raise ValueError(f"Unknown strict lap model: {spec.name}")
    return Pipeline([("preprocess", _preprocessor()), ("model", model)])


def build_strict_classifier() -> Pipeline:
    classifier = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=4,
        max_features=0.8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", _preprocessor()), ("model", classifier)])


def _green(frame: pd.DataFrame) -> pd.DataFrame:
    if "lap_regime" not in frame:
        raise ValueError("lap_regime labels are required")
    return frame[frame.target_valid & frame.lap_regime.astype(str).eq("green")].copy()


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        raise ValueError("No finite strict pace predictions")
    return float(np.mean(np.abs(predicted[mask] - actual[mask])))


def _pace_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        raise ValueError("No finite strict pace predictions")
    error = predicted[mask] - actual[mask]
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))


def _conformal_radius(actual: np.ndarray, predicted: np.ndarray, alpha: float) -> float:
    residual = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    residual = residual[np.isfinite(residual)]
    if len(residual) < 20:
        raise ValueError("Need at least 20 calibration green laps")
    rank = min(1.0, np.ceil((len(residual) + 1) * (1 - alpha)) / len(residual))
    return float(np.quantile(residual, rank, method="higher"))


def _full_probabilities(classifier: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    observed = classifier.predict_proba(frame[STRICT_FEATURES])
    classes = list(classifier.classes_)
    full = np.full((len(frame), len(REGIMES)), 1e-12, dtype=float)
    for source_index, name in enumerate(classes):
        if name in REGIMES:
            full[:, REGIMES.index(name)] = observed[:, source_index]
    full /= full.sum(axis=1, keepdims=True)
    return full


def _fit_green(train: pd.DataFrame, spec: LapModelSpec) -> Pipeline:
    green = _green(train)
    if len(green) < 50:
        raise ValueError("Insufficient green laps for strict pace regression")
    model = build_strict_regressor(spec)
    model.fit(green[STRICT_FEATURES], green.target_s)
    return model


def _select(train: pd.DataFrame, tuning: pd.DataFrame,
            specs: tuple[LapModelSpec, ...]) -> tuple[LapModelSpec, list[dict[str, Any]]]:
    target = _green(tuning)
    if target.empty:
        raise ValueError("Tuning race has no green laps")
    trials: list[dict[str, Any]] = []
    for spec in specs:
        try:
            model = _fit_green(train, spec)
            predicted = model.predict(target[STRICT_FEATURES])
            mae, rmse = _pace_metrics(target.target_s.to_numpy(), predicted)
            trials.append({"model": spec.name, "params": dict(spec.params), "mae_s": mae, "rmse_s": rmse})
        except (ValueError, RuntimeError, MemoryError) as exc:
            trials.append({
                "model": spec.name,
                "params": dict(spec.params),
                "mae_s": np.inf,
                "rmse_s": np.inf,
                "error": str(exc),
            })
    valid = [row for row in trials if np.isfinite(row["mae_s"])]
    if not valid:
        raise ValueError("No strict next-lap candidate completed tuning")
    winner = min(valid, key=lambda row: (row["mae_s"], row["model"]))
    return next(spec for spec in specs if spec.name == winner["model"]), trials


def fit_strict_mixture(
    datasets: list[pd.DataFrame],
    *,
    specs: tuple[LapModelSpec, ...] | None = None,
    alpha: float = 0.10,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Older races fit -> one tune -> one conformal calibration -> sealed test."""
    if len(datasets) < 4:
        raise ValueError("Need at least four chronological race datasets")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    specs = specs or (
        LapModelSpec("hist_gradient_boosting", {}),
        LapModelSpec("extra_trees", {}),
    )
    for frame in datasets:
        missing = set(STRICT_FEATURES + ["target_valid", "target_s", "lap_regime", "session_key"]) - set(frame)
        if missing:
            raise ValueError(f"Strict dataset missing columns: {sorted(missing)}")

    train = pd.concat(datasets[:-3], ignore_index=True)
    tuning = datasets[-3].copy()
    calibration = datasets[-2].copy()
    test = datasets[-1].copy()

    selected, trials = _select(train, tuning, specs)
    pre_cal = pd.concat([train, tuning], ignore_index=True)
    regressor = _fit_green(pre_cal, selected)
    classifier = build_strict_classifier().fit(pre_cal[STRICT_FEATURES], pre_cal.lap_regime.astype(str))

    cal_green = _green(calibration)
    cal_prediction = regressor.predict(cal_green[STRICT_FEATURES])
    radius = _conformal_radius(cal_green.target_s.to_numpy(), cal_prediction, alpha)

    test_green = _green(test)
    prediction = regressor.predict(test_green[STRICT_FEATURES])
    green_mae, green_rmse = _pace_metrics(test_green.target_s.to_numpy(), prediction)
    recent_median_mae = _mae(test_green.target_s.to_numpy(), test_green.recent_median_5_s.to_numpy())
    last_lap_mae = _mae(test_green.target_s.to_numpy(), test_green.last_lap_s.to_numpy())
    lower, upper = prediction - radius, prediction + radius
    actual = test_green.target_s.to_numpy(dtype=float)
    coverage = float(((actual >= lower) & (actual <= upper)).mean())

    regime_probability = _full_probabilities(classifier, test)
    regime_prediction = np.asarray(REGIMES, dtype=object)[np.argmax(regime_probability, axis=1)]
    truth = test.lap_regime.astype(str).to_numpy()
    regime_accuracy = float(accuracy_score(truth, regime_prediction))
    regime_loss = float(log_loss(truth, regime_probability, labels=list(REGIMES)))

    summary = StrictBenchmark(
        test_session=int(test.session_key.iloc[0]),
        selected_regressor=selected.name,
        green_rows=len(test_green),
        green_mae_s=green_mae,
        green_rmse_s=green_rmse,
        recent_median_mae_s=recent_median_mae,
        last_lap_mae_s=last_lap_mae,
        interval_coverage=coverage,
        interval_mean_width_s=2 * radius,
        regime_accuracy=regime_accuracy,
        regime_log_loss=regime_loss,
    )
    artifact = {
        "schema_version": 3,
        "task": "next_lap_strict_mixture",
        "features": STRICT_FEATURES,
        "regimes": list(REGIMES),
        "regime_classifier": classifier,
        "pace_regressor": regressor,
        "selected_regressor": selected.name,
        "conformal_alpha": alpha,
        "conformal_radius_s": radius,
        "trained_through_session": int(tuning.session_key.iloc[0]),
        "calibration_session": int(calibration.session_key.iloc[0]),
        "sealed_test_session": int(test.session_key.iloc[0]),
        "retrospective_stint_features_used": False,
    }
    audit = {
        "protocol": "older races fit -> one race tune -> one race conformal calibration -> sealed test",
        "feature_policy": "strict_asof_only",
        "excluded_retrospective_features": ["compound", "tyre_age", "stint_number"],
        "regressor_trials": [
            {key: (None if isinstance(value, float) and np.isinf(value) else value) for key, value in row.items()}
            for row in trials
        ],
        "conformal_alpha": alpha,
        "conformal_radius_s": radius,
        "test_updates_model": False,
        "baseline_comparison": {
            "recent_median_5_mae_s": recent_median_mae,
            "last_lap_mae_s": last_lap_mae,
        },
        "limitation": (
            "Historical OpenF1 timing is still a retrospective source snapshot; strict means no known "
            "future-dependent stint join is used, not that provider publication latency was independently measured."
        ),
    }
    return artifact, pd.DataFrame([asdict(summary)]), audit


def predict_live_strict(artifact: dict[str, Any], snapshot: dict[str, Any]) -> pd.DataFrame:
    if artifact.get("task") != "next_lap_strict_mixture" or artifact.get("features") != STRICT_FEATURES:
        raise ValueError("Strict next-lap artifact schema mismatch")
    frame = live_feature_rows(snapshot)
    if frame.empty:
        return frame
    probabilities = _full_probabilities(artifact["regime_classifier"], frame)
    pace = artifact["pace_regressor"].predict(frame[STRICT_FEATURES])
    radius = float(artifact["conformal_radius_s"])
    output = frame.copy()
    output["predicted_green_lap_s"] = pace
    output["green_lap_lower_s"] = pace - radius
    output["green_lap_upper_s"] = pace + radius
    for index, name in enumerate(REGIMES):
        output[f"p_{name}"] = probabilities[:, index]
    output["feature_policy"] = "strict_asof_only"
    return output
