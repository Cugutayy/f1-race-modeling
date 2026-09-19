from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from f1_research.lap_intelligence import LapModelSpec, build_lap_dataset
from f1_research.lap_mixture import attach_regime_labels
from f1_research.walk_forward import WalkForwardConfig, validate_event_order, walk_forward_strict
from f1_research.walk_forward_pipeline import model_specs


def _event(session_key: int, day: int) -> pd.DataFrame:
    base = datetime(2026, 1, 1, 12, tzinfo=UTC) + timedelta(days=day)
    laps = []
    for driver in range(1, 7):
        for lap in range(1, 17):
            pace = 88.0 + driver * 0.12 + lap * 0.015 + day * 0.001
            laps.append(
                {
                    "session_key": session_key,
                    "driver_number": driver,
                    "lap_number": lap,
                    "date_start": (
                        base + timedelta(seconds=(lap - 1) * 92 + driver * 0.1)
                    ).isoformat(),
                    "lap_duration": pace,
                    "is_pit_out_lap": False,
                }
            )
    weather = [
        {
            "session_key": session_key,
            "date": base.isoformat(),
            "air_temperature": 24.0,
            "track_temperature": 36.0,
            "humidity": 48.0,
            "rainfall": 0,
        }
    ]
    frame = build_lap_dataset(laps, weather_rows=weather, latency_s=1.0)
    return attach_regime_labels(frame, laps, [])


def _specs():
    return (
        LapModelSpec(
            "hist_gradient_boosting",
            {"max_iter": 12, "min_samples_leaf": 5, "max_leaf_nodes": 7},
        ),
    )


def test_walk_forward_seals_each_future_event_and_emits_per_lap_predictions():
    datasets = [_event(1001 + index, index * 7) for index in range(7)]
    folds, predictions, audit = walk_forward_strict(
        datasets,
        specs=_specs(),
        config=WalkForwardConfig(calibration_events=3),
    )

    assert folds.test_session.tolist() == [1006, 1007]
    assert folds.future_feature_violations.eq(0).all()
    assert folds.future_label_violations.eq(0).all()
    assert folds.test_updates_model.eq(False).all()
    assert not predictions.empty
    assert set(predictions.session_key.unique()) == {1006, 1007}
    assert (
        pd.to_datetime(predictions.feature_available_at, utc=True)
        <= pd.to_datetime(predictions.forecast_at, utc=True)
    ).all()
    assert (
        pd.to_datetime(predictions.target_available_at, utc=True)
        > pd.to_datetime(predictions.forecast_at, utc=True)
    ).all()
    assert audit["future_rows_used"] == 0
    assert audit["post_event_updates"] == 0
    assert 0 <= audit["aggregate"]["challenger_fold_win_rate"] <= 1


def test_mutating_future_test_targets_cannot_change_its_predictions():
    datasets = [_event(1101 + index, index * 7) for index in range(7)]
    _, original, _ = walk_forward_strict(datasets, specs=_specs())

    changed = [frame.copy() for frame in datasets]
    green = changed[-1]["lap_regime"].eq("green") & changed[-1]["target_valid"]
    changed[-1].loc[green, "target_s"] = changed[-1].loc[green, "target_s"] + 20.0
    _, mutated, _ = walk_forward_strict(changed, specs=_specs())

    left = original[original.session_key == 1107].reset_index(drop=True)
    right = mutated[mutated.session_key == 1107].reset_index(drop=True)
    np.testing.assert_allclose(left.predicted_s, right.predicted_s)
    np.testing.assert_allclose(left.lower_s, right.lower_s)
    np.testing.assert_allclose(left.upper_s, right.upper_s)
    assert not np.allclose(left.actual_s, right.actual_s)


def test_walk_forward_rejects_nonchronological_or_overlapping_events():
    first = _event(1201, 7)
    second = _event(1202, 0)
    with pytest.raises(ValueError, match="chronological"):
        validate_event_order([first, second])


def test_model_registry_exposes_modern_challengers_without_importing_them():
    names = [spec.name for spec in model_specs(include_modern=True, include_foundation=True)]
    assert names == [
        "hist_gradient_boosting",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "tabicl_v2",
    ]
