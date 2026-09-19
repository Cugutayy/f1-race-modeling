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
BASELINE_REGRESSOR = "recent_median_5_baseline"
BASELINE_MODE = "recent_median_5_baseline"
RESIDUAL_MODE = "recent_median_5_residual"
MIN_BASELINE_RELATIVE_IMPROVEMENT = 0.01


@dataclass(frozen=True)
class StrictBenchmark:
    test_session: int
    selected_regressor: str
    pace_prediction_mode: str
    challenger_selected: bool
    green_rows: int
    calibration_events: int
    tuning_baseline_mae_s: float
    tuning_selected_mae_s: float
    green_mae_s: float
    green_rmse_s: float
    recent_median_mae_s: float
    last_lap_mae_s: float
    interval_coverage: float
    interval_mean_width_s: float
    coverage_50: float
    coverage_80: float
    coverage_90: float
    coverage_95: float
    interval_width_50_s: float
    interval_width_80_s: float
    interval_width_90_s: float
    interval_width_95_s: float
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
    elif spec.name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install the modern extra to evaluate XGBoost") from exc
        defaults = {
            "objective": "reg:absoluteerror",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 8.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 8.0,
            "reg_alpha": 0.05,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        model = XGBRegressor(**defaults)
    elif spec.name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Install the modern extra to evaluate LightGBM") from exc
        defaults = {
            "objective": "regression_l1",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 25,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 8.0,
            "reg_alpha": 0.05,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        defaults.update(params)
        model = LGBMRegressor(**defaults)
    elif spec.name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("Install the modern extra to evaluate CatBoost") from exc
        defaults = {
            "loss_function": "MAE",
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 7,
            "l2_leaf_reg": 8.0,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        defaults.update(params)
        model = CatBoostRegressor(**defaults)
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
    return frame[frame.target_valid & frame.lap_regime.eq("green")].copy()


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
    """Fit a circuit-scale-invariant residual over the recent-median baseline."""
    green = _green(train)
    if len(green) < 50:
        raise ValueError("Insufficient green laps for strict pace regression")
    baseline = pd.to_numeric(green.recent_median_5_s, errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(green.target_s, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(baseline) & np.isfinite(target)
    if int(mask.sum()) < 50:
        raise ValueError("Insufficient finite green-lap residual targets")
    model = build_strict_regressor(spec)
    model.fit(green.loc[mask, STRICT_FEATURES], target[mask] - baseline[mask])
    return model


def _predict_green_pace(
    model: Pipeline | None,
    frame: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    baseline = pd.to_numeric(frame.recent_median_5_s, errors="coerce").to_numpy(dtype=float)
    if mode == BASELINE_MODE:
        predicted = baseline
    elif mode == RESIDUAL_MODE:
        if model is None:
            raise ValueError("Residual pace mode requires a fitted regressor")
        residual = np.asarray(model.predict(frame[STRICT_FEATURES]), dtype=float)
        predicted = baseline + residual
    else:
        raise ValueError(f"Unsupported strict pace prediction mode: {mode}")
    if not np.isfinite(predicted).all() or np.any(predicted <= 0):
        raise ValueError("Strict pace prediction produced invalid lap durations")
    return predicted


def _select(
    train: pd.DataFrame,
    tuning: pd.DataFrame,
    specs: tuple[LapModelSpec, ...],
) -> tuple[LapModelSpec | None, str, list[dict[str, Any]], float, float]:
    """Select a residual challenger only when it beats the live-available baseline."""
    target = _green(tuning)
    if target.empty:
        raise ValueError("Tuning race has no green laps")

    actual = target.target_s.to_numpy(dtype=float)
    baseline_prediction = target.recent_median_5_s.to_numpy(dtype=float)
    baseline_mae, baseline_rmse = _pace_metrics(actual, baseline_prediction)
    trials: list[dict[str, Any]] = [{
        "model": BASELINE_REGRESSOR,
        "params": {},
        "pace_prediction_mode": BASELINE_MODE,
        "mae_s": baseline_mae,
        "rmse_s": baseline_rmse,
        "relative_improvement_vs_baseline": 0.0,
        "beats_recent_median_baseline": True,
    }]

    fitted_specs: dict[str, LapModelSpec] = {}
    for spec in specs:
        try:
            model = _fit_green(train, spec)
            predicted = _predict_green_pace(model, target, RESIDUAL_MODE)
            mae, rmse = _pace_metrics(actual, predicted)
            relative = (baseline_mae - mae) / max(baseline_mae, 1e-12)
            beats = relative >= MIN_BASELINE_RELATIVE_IMPROVEMENT
            trials.append({
                "model": spec.name,
                "params": dict(spec.params),
                "pace_prediction_mode": RESIDUAL_MODE,
                "mae_s": mae,
                "rmse_s": rmse,
                "relative_improvement_vs_baseline": relative,
                "beats_recent_median_baseline": beats,
            })
            fitted_specs[spec.name] = spec
        except (ValueError, RuntimeError, MemoryError) as exc:
            trials.append({
                "model": spec.name,
                "params": dict(spec.params),
                "pace_prediction_mode": RESIDUAL_MODE,
                "mae_s": np.inf,
                "rmse_s": np.inf,
                "relative_improvement_vs_baseline": None,
                "beats_recent_median_baseline": False,
                "error": str(exc),
            })

    eligible = [
        row
        for row in trials
        if row["model"] != BASELINE_REGRESSOR
        and np.isfinite(row["mae_s"])
        and row["beats_recent_median_baseline"]
    ]
    if not eligible:
        return None, BASELINE_MODE, trials, baseline_mae, baseline_mae

    winner = min(eligible, key=lambda row: (row["mae_s"], row["model"]))
    return (
        fitted_specs[winner["model"]],
        RESIDUAL_MODE,
        trials,
        baseline_mae,
        float(winner["mae_s"]),
    )


def fit_strict_mixture(
    datasets: list[pd.DataFrame],
    *,
    specs: tuple[LapModelSpec, ...] | None = None,
    alpha: float = 0.10,
    calibration_events: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Older races fit -> one tune -> multi-event conformal calibration -> sealed test."""
    if calibration_events < 2:
        raise ValueError("At least two calibration events are required")
    if len(datasets) < calibration_events + 3:
        raise ValueError(
            f"Need at least {calibration_events + 3} chronological race datasets "
            "for fit, tuning, multi-event calibration and sealed test"
        )
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

    tuning_index = len(datasets) - calibration_events - 2
    train = pd.concat(datasets[:tuning_index], ignore_index=True)
    tuning = datasets[tuning_index].copy()
    calibration_frames = [
        frame.copy()
        for frame in datasets[tuning_index + 1:-1]
    ]
    test = datasets[-1].copy()

    selected, pace_mode, trials, tuning_baseline_mae, tuning_selected_mae = _select(
        train,
        tuning,
        specs,
    )
    pre_cal = pd.concat([train, tuning], ignore_index=True)
    regressor = _fit_green(pre_cal, selected) if selected is not None else None
    classifier_train = pre_cal[pre_cal.lap_regime.notna()].copy()
    if classifier_train.empty:
        raise ValueError("No known strict lap-regime labels are available for classifier training")
    classifier = build_strict_classifier().fit(
        classifier_train[STRICT_FEATURES], classifier_train.lap_regime.astype(str)
    )

    calibration_green = [_green(frame) for frame in calibration_frames]
    if any(frame.empty for frame in calibration_green):
        raise ValueError("A conformal calibration event has no green laps")
    cal_green = pd.concat(calibration_green, ignore_index=True)
    cal_prediction = _predict_green_pace(regressor, cal_green, pace_mode)
    coverage_levels = (0.50, 0.80, 0.90, 0.95)
    conformal_radii = {
        f"{coverage:.2f}": _conformal_radius(
            cal_green.target_s.to_numpy(),
            cal_prediction,
            1.0 - coverage,
        )
        for coverage in coverage_levels
    }
    requested_coverage = 1.0 - alpha
    radius = _conformal_radius(cal_green.target_s.to_numpy(), cal_prediction, alpha)

    test_green = _green(test)
    prediction = _predict_green_pace(regressor, test_green, pace_mode)
    green_mae, green_rmse = _pace_metrics(test_green.target_s.to_numpy(), prediction)
    recent_median_mae = _mae(test_green.target_s.to_numpy(), test_green.recent_median_5_s.to_numpy())
    last_lap_mae = _mae(test_green.target_s.to_numpy(), test_green.last_lap_s.to_numpy())
    lower, upper = prediction - radius, prediction + radius
    actual = test_green.target_s.to_numpy(dtype=float)
    coverage = float(((actual >= lower) & (actual <= upper)).mean())
    coverage_evidence = {}
    for nominal in coverage_levels:
        key = f"{nominal:.2f}"
        level_radius = conformal_radii[key]
        empirical = float(
            ((actual >= prediction - level_radius) & (actual <= prediction + level_radius)).mean()
        )
        coverage_evidence[key] = {
            "nominal": nominal,
            "empirical": empirical,
            "radius_s": level_radius,
            "mean_width_s": 2.0 * level_radius,
        }

    regime_test = test[test.lap_regime.notna()].copy()
    if regime_test.empty:
        raise ValueError("Sealed strict test has no known lap-regime labels")
    regime_probability = _full_probabilities(classifier, regime_test)
    regime_prediction = np.asarray(REGIMES, dtype=object)[np.argmax(regime_probability, axis=1)]
    truth = regime_test.lap_regime.astype(str).to_numpy()
    regime_accuracy = float(accuracy_score(truth, regime_prediction))
    regime_loss = float(log_loss(truth, regime_probability, labels=list(REGIMES)))

    summary = StrictBenchmark(
        test_session=int(test.session_key.iloc[0]),
        selected_regressor=(selected.name if selected is not None else BASELINE_REGRESSOR),
        pace_prediction_mode=pace_mode,
        challenger_selected=selected is not None,
        green_rows=len(test_green),
        calibration_events=len(calibration_frames),
        tuning_baseline_mae_s=tuning_baseline_mae,
        tuning_selected_mae_s=tuning_selected_mae,
        green_mae_s=green_mae,
        green_rmse_s=green_rmse,
        recent_median_mae_s=recent_median_mae,
        last_lap_mae_s=last_lap_mae,
        interval_coverage=coverage,
        interval_mean_width_s=2 * radius,
        coverage_50=coverage_evidence["0.50"]["empirical"],
        coverage_80=coverage_evidence["0.80"]["empirical"],
        coverage_90=coverage_evidence["0.90"]["empirical"],
        coverage_95=coverage_evidence["0.95"]["empirical"],
        interval_width_50_s=coverage_evidence["0.50"]["mean_width_s"],
        interval_width_80_s=coverage_evidence["0.80"]["mean_width_s"],
        interval_width_90_s=coverage_evidence["0.90"]["mean_width_s"],
        interval_width_95_s=coverage_evidence["0.95"]["mean_width_s"],
        regime_accuracy=regime_accuracy,
        regime_log_loss=regime_loss,
    )
    artifact = {
        "schema_version": 4,
        "task": "next_lap_strict_mixture",
        "features": STRICT_FEATURES,
        "regimes": list(REGIMES),
        "regime_classifier": classifier,
        "pace_regressor": regressor,
        "selected_regressor": (selected.name if selected is not None else BASELINE_REGRESSOR),
        "pace_prediction_mode": pace_mode,
        "pace_regressor_target": (
            "target_s_minus_recent_median_5_s"
            if pace_mode == RESIDUAL_MODE
            else "recent_median_5_s"
        ),
        "baseline_guard": {
            "baseline": BASELINE_REGRESSOR,
            "minimum_relative_improvement": MIN_BASELINE_RELATIVE_IMPROVEMENT,
            "tuning_baseline_mae_s": tuning_baseline_mae,
            "tuning_selected_mae_s": tuning_selected_mae,
            "challenger_selected": selected is not None,
        },
        "conformal_alpha": alpha,
        "conformal_nominal_coverage": requested_coverage,
        "conformal_radius_s": radius,
        "conformal_radii_s": conformal_radii,
        "trained_through_session": int(tuning.session_key.iloc[0]),
        "calibration_sessions": [
            int(frame.session_key.iloc[0]) for frame in calibration_frames
        ],
        "sealed_test_session": int(test.session_key.iloc[0]),
        "retrospective_stint_features_used": False,
    }
    audit = {
        "protocol": (
            "older races fit -> one race tune -> multi-event pooled conformal calibration "
            "-> sealed test"
        ),
        "feature_policy": "strict_asof_only",
        "excluded_retrospective_features": ["compound", "tyre_age", "stint_number"],
        "calibration_sessions": [
            int(frame.session_key.iloc[0]) for frame in calibration_frames
        ],
        "pace_prediction_mode": pace_mode,
        "baseline_guard": {
            "baseline": BASELINE_REGRESSOR,
            "minimum_relative_improvement": MIN_BASELINE_RELATIVE_IMPROVEMENT,
            "tuning_baseline_mae_s": tuning_baseline_mae,
            "tuning_selected_mae_s": tuning_selected_mae,
            "challenger_selected": selected is not None,
        },
        "regressor_trials": [
            {key: (None if isinstance(value, float) and np.isinf(value) else value) for key, value in row.items()}
            for row in trials
        ],
        "conformal_alpha": alpha,
        "conformal_nominal_coverage": requested_coverage,
        "conformal_radius_s": radius,
        "conformal_radii_s": conformal_radii,
        "sealed_test_coverage": coverage_evidence,
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
    pace = _predict_green_pace(
        artifact.get("pace_regressor"),
        frame,
        str(artifact.get("pace_prediction_mode") or ""),
    )
    radius = float(artifact["conformal_radius_s"])
    output = frame.copy()
    output["predicted_green_lap_s"] = pace
    output["green_lap_lower_s"] = pace - radius
    output["green_lap_upper_s"] = pace + radius
    for index, name in enumerate(REGIMES):
        output[f"p_{name}"] = probabilities[:, index]
    output["feature_policy"] = "strict_asof_only"
    return output
