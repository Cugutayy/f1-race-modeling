"""Leakage-resistant expanding/rolling walk-forward backtest for strict F1 pace.

Each scored race is treated as unseen. Model selection and conformal calibration
use only earlier races. Per-lap predictions are emitted with timestamps so future
feature/label leakage can be mechanically audited.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .lap_intelligence import LapModelSpec
from .lap_strict import _green, _predict_green_pace, fit_strict_mixture

RESEARCH_REFERENCES = {
    "time_series_validation": "https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split",
    "adaptive_conformal": (
        "https://papers.neurips.cc/paper/2021/hash/"
        "0d441de75945e5acbc865406fc9a2559-Abstract.html"
    ),
    "xgboost_survival_aft": (
        "https://xgboost.readthedocs.io/en/stable/python/survival-examples/aft_survival_demo.html"
    ),
    "catboost_ranking": "https://catboost.ai/docs/en/concepts/loss-functions-ranking",
    "lightgbm_lambdarank": "https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#lambdarank",
    "temporal_fusion_transformer": "https://arxiv.org/abs/1912.09363",
    "deephit_competing_risks": "https://doi.org/10.1609/aaai.v32i1.11842",
}


@dataclass(frozen=True)
class WalkForwardConfig:
    calibration_events: int = 3
    alpha: float = 0.10
    max_events: int | None = None

    @property
    def minimum_prefix_events(self) -> int:
        return self.calibration_events + 3


def _session_identity(frame: pd.DataFrame) -> int:
    if frame.empty or "session_key" not in frame:
        raise ValueError("Every walk-forward event must be a non-empty session dataset")
    values = pd.to_numeric(frame["session_key"], errors="coerce").dropna().unique()
    if len(values) != 1 or values[0] <= 0 or float(values[0]).is_integer() is False:
        raise ValueError("Each walk-forward dataset must contain exactly one valid session_key")
    return int(values[0])


def _event_start(frame: pd.DataFrame) -> pd.Timestamp:
    if "forecast_at" not in frame:
        raise ValueError("Walk-forward datasets require forecast_at")
    values = pd.to_datetime(frame["forecast_at"], utc=True, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Walk-forward event has no valid forecast_at timestamps")
    return values.min()


def _max_label_available_at(frame: pd.DataFrame) -> pd.Timestamp | None:
    if "target_available_at" not in frame:
        return None
    values = pd.to_datetime(frame["target_available_at"], utc=True, errors="coerce").dropna()
    return None if values.empty else values.max()


def validate_event_order(datasets: list[pd.DataFrame]) -> list[dict[str, Any]]:
    """Reject duplicate, overlapping or non-chronological event inputs."""
    events: list[dict[str, Any]] = []
    seen: set[int] = set()
    previous_start: pd.Timestamp | None = None
    previous_label_end: pd.Timestamp | None = None
    for frame in datasets:
        session_key = _session_identity(frame)
        if session_key in seen:
            raise ValueError(f"Duplicate walk-forward session_key: {session_key}")
        seen.add(session_key)
        start = _event_start(frame)
        label_end = _max_label_available_at(frame)
        if previous_start is not None and start <= previous_start:
            raise ValueError("Walk-forward datasets must be supplied in strict chronological order")
        if previous_label_end is not None and previous_label_end >= start:
            raise ValueError("Walk-forward events overlap in label-availability time")
        events.append(
            {
                "session_key": session_key,
                "forecast_start": start,
                "label_available_end": label_end,
            }
        )
        previous_start = start
        previous_label_end = label_end
    return events


def _assert_row_cutoffs(frame: pd.DataFrame) -> dict[str, int]:
    forecast = pd.to_datetime(frame["forecast_at"], utc=True, errors="coerce")
    feature = pd.to_datetime(frame["feature_available_at"], utc=True, errors="coerce")
    target = pd.to_datetime(frame["target_available_at"], utc=True, errors="coerce")
    invalid_forecast = int(forecast.isna().sum())
    future_features = int((feature.notna() & forecast.notna() & (feature > forecast)).sum())
    early_targets = int((target.notna() & forecast.notna() & (target <= forecast)).sum())
    checks = {
        "invalid_forecast_at": invalid_forecast,
        "features_after_forecast": future_features,
        "targets_available_at_or_before_forecast": early_targets,
    }
    if any(checks.values()):
        raise AssertionError(f"Walk-forward as-of contract failed: {checks}")
    return checks


def _score_test_event(
    artifact: dict[str, Any],
    test: pd.DataFrame,
    *,
    fold_index: int,
    trained_session_keys: list[int],
) -> pd.DataFrame:
    green = _green(test)
    if green.empty:
        raise ValueError("Walk-forward test event has no scoreable green laps")
    predicted = _predict_green_pace(
        artifact.get("pace_regressor"),
        green,
        str(artifact["pace_prediction_mode"]),
    )
    actual = pd.to_numeric(green["target_s"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(green["recent_median_5_s"], errors="coerce").to_numpy(dtype=float)
    radius = float(artifact["conformal_radius_s"])
    result = pd.DataFrame(
        {
            "fold": fold_index,
            "session_key": green["session_key"].astype(int).to_numpy(),
            "driver_number": green["driver_number"].astype(str).to_numpy(),
            "lap_number": green["lap_number"].astype(int).to_numpy(),
            "forecast_at": pd.to_datetime(green["forecast_at"], utc=True).to_numpy(),
            "feature_available_at": pd.to_datetime(
                green["feature_available_at"], utc=True
            ).to_numpy(),
            "target_available_at": pd.to_datetime(
                green["target_available_at"], utc=True
            ).to_numpy(),
            "selected_regressor": artifact["selected_regressor"],
            "pace_prediction_mode": artifact["pace_prediction_mode"],
            "predicted_s": predicted,
            "baseline_s": baseline,
            "actual_s": actual,
            "lower_s": predicted - radius,
            "upper_s": predicted + radius,
        }
    )
    result["error_s"] = result["predicted_s"] - result["actual_s"]
    result["absolute_error_s"] = result["error_s"].abs()
    result["baseline_absolute_error_s"] = (result["baseline_s"] - result["actual_s"]).abs()
    result["covered"] = (
        (result["actual_s"] >= result["lower_s"])
        & (result["actual_s"] <= result["upper_s"])
    )
    result["trained_session_count"] = len(trained_session_keys)
    result["trained_session_keys"] = ",".join(map(str, trained_session_keys))
    return result


def walk_forward_strict(
    datasets: list[pd.DataFrame],
    *,
    specs: tuple[LapModelSpec, ...] | None = None,
    config: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Backtest strict pace with a model that never sees the scored race.

    For fold t the full pipeline (fit, challenger selection and conformal
    calibration) is rebuilt only from races < t. The final race in each prefix is
    scored once and never used to update that fold's fitted model.
    """
    cfg = config or WalkForwardConfig()
    if cfg.calibration_events < 2:
        raise ValueError("Walk-forward requires at least two calibration events")
    if not 0 < cfg.alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if cfg.max_events is not None and cfg.max_events < cfg.minimum_prefix_events:
        raise ValueError("max_events is too small for fit/tune/calibration/test protocol")
    events = validate_event_order(datasets)
    if len(datasets) < cfg.minimum_prefix_events:
        raise ValueError(
            f"Need at least {cfg.minimum_prefix_events} chronological races for walk-forward"
        )

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_index in range(cfg.minimum_prefix_events - 1, len(datasets)):
        start = 0
        if cfg.max_events is not None:
            start = max(0, test_index + 1 - cfg.max_events)
        prefix = datasets[start : test_index + 1]
        train_sessions = [events[index]["session_key"] for index in range(start, test_index)]
        test_session = events[test_index]["session_key"]
        test_start = events[test_index]["forecast_start"]

        prior_label_ends = [
            events[index]["label_available_end"]
            for index in range(start, test_index)
            if events[index]["label_available_end"] is not None
        ]
        if prior_label_ends and max(prior_label_ends) >= test_start:
            raise AssertionError("A training label was not available before the test race began")

        _assert_row_cutoffs(datasets[test_index])
        artifact, metrics, audit = fit_strict_mixture(
            prefix,
            specs=specs,
            alpha=cfg.alpha,
            calibration_events=cfg.calibration_events,
        )
        if int(artifact["sealed_test_session"]) != test_session:
            raise AssertionError("Strict fold did not seal the intended test session")
        calibration_sessions = [int(value) for value in artifact["calibration_sessions"]]
        if test_session in calibration_sessions:
            raise AssertionError("Test session leaked into conformal calibration")
        if int(artifact["trained_through_session"]) == test_session:
            raise AssertionError("Test session leaked into fitted pace model")

        metric = metrics.iloc[0].to_dict()
        fold_rows.append(
            {
                "fold": len(fold_rows),
                "test_session": test_session,
                "train_window_start_session": train_sessions[0],
                "train_window_end_session": train_sessions[-1],
                "train_session_count": len(train_sessions),
                "selected_regressor": artifact["selected_regressor"],
                "pace_prediction_mode": artifact["pace_prediction_mode"],
                "challenger_selected": bool(artifact["baseline_guard"]["challenger_selected"]),
                "green_rows": int(metric["green_rows"]),
                "green_mae_s": float(metric["green_mae_s"]),
                "green_rmse_s": float(metric["green_rmse_s"]),
                "baseline_mae_s": float(metric["recent_median_mae_s"]),
                "last_lap_mae_s": float(metric["last_lap_mae_s"]),
                "coverage_90": float(metric["coverage_90"]),
                "interval_width_90_s": float(metric["interval_width_90_s"]),
                "regime_accuracy": float(metric["regime_accuracy"]),
                "regime_log_loss": float(metric["regime_log_loss"]),
                "calibration_sessions": ",".join(map(str, calibration_sessions)),
                "future_feature_violations": 0,
                "future_label_violations": 0,
                "test_updates_model": bool(audit.get("test_updates_model", True)),
            }
        )
        if fold_rows[-1]["test_updates_model"]:
            raise AssertionError("Walk-forward fold updated model on the sealed test race")
        prediction_frames.append(
            _score_test_event(
                artifact,
                datasets[test_index],
                fold_index=fold_rows[-1]["fold"],
                trained_session_keys=train_sessions,
            )
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = folds["green_rows"].to_numpy(dtype=float)
    selection = Counter(folds["selected_regressor"].astype(str))
    audit = {
        "schema_version": 1,
        "protocol": "chronological expanding/rolling walk-forward; fit+tune+calibrate strictly before each test race",
        "folds": len(folds),
        "first_test_session": int(folds.iloc[0]["test_session"]),
        "last_test_session": int(folds.iloc[-1]["test_session"]),
        "max_events": cfg.max_events,
        "calibration_events": cfg.calibration_events,
        "alpha": cfg.alpha,
        "future_rows_used": 0,
        "post_event_updates": 0,
        "selection_frequency": dict(selection),
        "aggregate": {
            "row_weighted_mae_s": float(np.average(folds["green_mae_s"], weights=weights)),
            "row_weighted_baseline_mae_s": float(
                np.average(folds["baseline_mae_s"], weights=weights)
            ),
            "mean_coverage_90": float(folds["coverage_90"].mean()),
            "mean_interval_width_90_s": float(folds["interval_width_90_s"].mean()),
            "challenger_fold_win_rate": float(folds["challenger_selected"].mean()),
        },
        "research_references": RESEARCH_REFERENCES,
        "survivorship_policy": (
            "pace scoring is conditional on scoreable green laps; race-outcome/DNF evaluation must "
            "retain all starters and treat retirement as an outcome rather than dropping rows"
        ),
        "scope": (
            "This module backtests next-green-lap pace only. Ranking and competing-risk survival "
            "models are research challengers for race-outcome/DNF layers, not claimed as implemented here."
        ),
    }
    return folds, predictions, audit
