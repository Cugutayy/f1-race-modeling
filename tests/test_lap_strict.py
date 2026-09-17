from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from f1_research.lap_intelligence import LapModelSpec, build_lap_dataset
from f1_research.lap_mixture import attach_regime_labels
from f1_research.lap_strict import STRICT_FEATURES, fit_strict_mixture, predict_live_strict


def _dataset(session_key: int, day: int) -> pd.DataFrame:
    base = datetime(2026, 3, 1, 12, tzinfo=UTC) + timedelta(days=day)
    laps, stints, pits = [], [], []
    for driver in range(1, 7):
        for lap in range(1, 18):
            pace = 88.7 + driver * 0.16 + lap * 0.018 + day * 0.002
            laps.append({
                "session_key": session_key,
                "driver_number": driver,
                "lap_number": lap,
                "date_start": (base + timedelta(seconds=(lap - 1) * 92 + driver * 0.1)).isoformat(),
                "lap_duration": pace + (21.0 if lap == 10 and driver in (1, 3) else 0.0),
                "is_pit_out_lap": lap == 11 and driver in (1, 3),
            })
        stints.extend([
            {
                "session_key": session_key,
                "driver_number": driver,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 10,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            },
            {
                "session_key": session_key,
                "driver_number": driver,
                "stint_number": 2,
                "lap_start": 11,
                "lap_end": 17,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
        ])
        if driver in (1, 3):
            pits.append({
                "session_key": session_key,
                "driver_number": driver,
                "lap_number": 10,
                "date": (base + timedelta(seconds=9 * 92 + 70 + driver * 0.1)).isoformat(),
            })
    weather = [{
        "session_key": session_key,
        "date": base.isoformat(),
        "air_temperature": 23.0,
        "track_temperature": 36.0,
        "humidity": 50.0,
        "rainfall": 0,
    }]
    frame = build_lap_dataset(laps, stint_rows=stints, weather_rows=weather, pit_rows=pits)
    return attach_regime_labels(frame, laps, pits)


def _specs():
    return (LapModelSpec("hist_gradient_boosting", {"max_iter": 25, "min_samples_leaf": 5}),)


def test_strict_artifact_excludes_retrospective_stint_features_and_has_baselines():
    datasets = [_dataset(501 + index, index * 7) for index in range(4)]
    artifact, metrics, audit = fit_strict_mixture(datasets, specs=_specs())
    assert artifact["task"] == "next_lap_strict_mixture"
    assert artifact["retrospective_stint_features_used"] is False
    assert artifact["features"] == STRICT_FEATURES
    assert not {"compound", "tyre_age", "stint_number"} & set(artifact["features"])
    assert audit["feature_policy"] == "strict_asof_only"
    row = metrics.iloc[0]
    assert np.isfinite(row.green_mae_s)
    assert np.isfinite(row.recent_median_mae_s)
    assert np.isfinite(row.last_lap_mae_s)
    assert 0 <= row.interval_coverage <= 1


def test_mutating_retrospective_stint_columns_cannot_change_strict_benchmark():
    datasets = [_dataset(601 + index, index * 7) for index in range(4)]
    _, original, original_audit = fit_strict_mixture(datasets, specs=_specs())
    changed = [frame.copy() for frame in datasets]
    for frame in changed:
        frame["compound"] = np.where(frame.lap_number.mod(2).eq(0), "SOFT", "WET")
        frame["tyre_age"] = 999 - frame.lap_number
        frame["stint_number"] = 99
    _, mutated, mutated_audit = fit_strict_mixture(changed, specs=_specs())
    pd.testing.assert_frame_equal(original, mutated)
    assert original_audit["conformal_radius_s"] == mutated_audit["conformal_radius_s"]
    assert original_audit["baseline_comparison"] == mutated_audit["baseline_comparison"]


def test_strict_live_probabilities_are_coherent():
    datasets = [_dataset(701 + index, index * 7) for index in range(4)]
    artifact, _, _ = fit_strict_mixture(datasets, specs=_specs())
    state = {
        "current_lap": 13,
        "flag": "GREEN",
        "safety_car": None,
        "weather": {
            "air_temperature_c": 23.0,
            "track_temperature_c": 36.0,
            "humidity_pct": 50.0,
            "rainfall": False,
        },
        "drivers": [
            {
                "driver_number": driver,
                "lap_number": 13,
                "recent_laps_s": [
                    89.0 + driver * 0.1,
                    89.05 + driver * 0.1,
                    89.02 + driver * 0.1,
                    89.08 + driver * 0.1,
                ],
                "compound": "HARD",
                "tyre_age": 3,
                "stint_number": 2,
                "pit_stops": 1,
            }
            for driver in range(1, 4)
        ],
    }
    predicted = predict_live_strict(artifact, state)
    assert len(predicted) == 3
    np.testing.assert_allclose(
        predicted[["p_green", "p_neutralized", "p_pit"]].sum(axis=1).to_numpy(),
        1.0,
        atol=1e-10,
    )
    assert (predicted.green_lap_lower_s < predicted.predicted_green_lap_s).all()
    assert (predicted.predicted_green_lap_s < predicted.green_lap_upper_s).all()
    assert predicted.feature_policy.eq("strict_asof_only").all()
