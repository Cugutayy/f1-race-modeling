from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from f1_research.lap_intelligence import LapModelSpec, build_lap_dataset
from f1_research.lap_mixture import attach_regime_labels
from f1_research.lap_strict import (
    BASELINE_MODE,
    BASELINE_REGRESSOR,
    RESIDUAL_MODE,
    STRICT_FEATURES,
    fit_strict_mixture,
    predict_live_strict,
)
from f1_research.live_intelligence import combined_live_report
from f1_research.strategy import SimulationConfig


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


def _live_state():
    return {
        "session_key": 999,
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
                "acronym": f"D{driver}",
                "position": driver,
                "gap_to_leader_s": 0.0 if driver == 1 else 2.5 * (driver - 1),
                "lap_number": 13,
                "recent_laps_s": [
                    89.0 + driver * 0.1,
                    89.05 + driver * 0.1,
                    89.02 + driver * 0.1,
                    89.08 + driver * 0.1,
                ],
                "last_lap_s": 89.08 + driver * 0.1,
                "compound": "HARD",
                "tyre_age": 3,
                "stint_number": 2,
                "pit_stops": 1,
            }
            for driver in range(1, 4)
        ],
    }


def test_strict_artifact_excludes_retrospective_stint_features_and_has_baselines():
    datasets = [_dataset(501 + index, index * 7) for index in range(6)]
    artifact, metrics, audit = fit_strict_mixture(datasets, specs=_specs())
    assert artifact["schema_version"] == 4
    assert artifact["task"] == "next_lap_strict_mixture"
    assert artifact["retrospective_stint_features_used"] is False
    assert artifact["pace_prediction_mode"] in {BASELINE_MODE, RESIDUAL_MODE}
    assert isinstance(artifact["baseline_guard"], dict)
    assert artifact["features"] == STRICT_FEATURES
    assert not {"compound", "tyre_age", "stint_number"} & set(artifact["features"])
    assert audit["feature_policy"] == "strict_asof_only"
    row = metrics.iloc[0]
    assert np.isfinite(row.green_mae_s)
    assert np.isfinite(row.recent_median_mae_s)
    assert np.isfinite(row.last_lap_mae_s)
    assert row.calibration_events == 3
    assert 0 <= row.interval_coverage <= 1
    for column in ("coverage_50", "coverage_80", "coverage_90", "coverage_95"):
        assert 0 <= row[column] <= 1
    assert row.interval_width_50_s <= row.interval_width_80_s
    assert row.interval_width_80_s <= row.interval_width_90_s
    assert row.interval_width_90_s <= row.interval_width_95_s
    assert len(artifact["calibration_sessions"]) == 3
    assert set(artifact["conformal_radii_s"]) == {"0.50", "0.80", "0.90", "0.95"}
    assert audit["calibration_sessions"] == artifact["calibration_sessions"]
    assert set(audit["sealed_test_coverage"]) == {"0.50", "0.80", "0.90", "0.95"}


def test_strict_pace_is_invariant_to_cross_circuit_absolute_time_shift():
    datasets = [_dataset(551 + index, index * 7) for index in range(6)]
    shifted = [frame.copy() for frame in datasets]
    for frame in shifted[1:]:
        for column in ("target_s", "last_lap_s", "recent_median_3_s", "recent_median_5_s"):
            frame[column] = frame[column] + 25.0

    artifact, metrics, audit = fit_strict_mixture(shifted, specs=_specs())
    predicted = predict_live_strict(
        artifact,
        {
            **_live_state(),
            "drivers": [
                {
                    **driver,
                    "recent_laps_s": [value + 25.0 for value in driver["recent_laps_s"]],
                    "last_lap_s": driver["last_lap_s"] + 25.0,
                }
                for driver in _live_state()["drivers"]
            ],
        },
    )
    assert artifact["schema_version"] == 4
    assert artifact["pace_prediction_mode"] in {BASELINE_MODE, RESIDUAL_MODE}
    assert predicted.predicted_green_lap_s.mean() > 108.0
    row = metrics.iloc[0]
    assert row.green_mae_s < 3.0
    assert audit["baseline_guard"]["tuning_selected_mae_s"] <= (
        audit["baseline_guard"]["tuning_baseline_mae_s"] * 1.01 + 1e-12
    )


def test_baseline_guard_rejects_challenger_that_cannot_beat_recent_median():
    datasets = [_dataset(571 + index, index * 7) for index in range(6)]
    tuning = datasets[1].copy()
    green = tuning.lap_regime.eq("green") & tuning.target_valid
    tuning.loc[green, "target_s"] = tuning.loc[green, "recent_median_5_s"]
    datasets[1] = tuning

    artifact, metrics, audit = fit_strict_mixture(datasets, specs=_specs())
    assert artifact["selected_regressor"] == BASELINE_REGRESSOR
    assert artifact["pace_prediction_mode"] == BASELINE_MODE
    assert artifact["pace_regressor"] is None
    assert artifact["baseline_guard"]["challenger_selected"] is False
    assert audit["baseline_guard"]["tuning_baseline_mae_s"] == 0.0
    assert metrics.iloc[0].challenger_selected == False  # noqa: E712


def test_mutating_retrospective_stint_columns_cannot_change_strict_benchmark():
    datasets = [_dataset(601 + index, index * 7) for index in range(6)]
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


def test_mutating_sealed_test_targets_cannot_change_selection_or_conformal_calibration():
    datasets = [_dataset(651 + index, index * 7) for index in range(6)]
    artifact, _, audit = fit_strict_mixture(datasets, specs=_specs())
    changed = [frame.copy() for frame in datasets]
    changed[-1]["target_s"] = changed[-1]["target_s"] + 25.0
    changed_artifact, _, changed_audit = fit_strict_mixture(changed, specs=_specs())
    assert artifact["selected_regressor"] == changed_artifact["selected_regressor"]
    assert artifact["conformal_radii_s"] == changed_artifact["conformal_radii_s"]
    assert artifact["calibration_sessions"] == changed_artifact["calibration_sessions"]
    assert audit["regressor_trials"] == changed_audit["regressor_trials"]


def test_strict_live_probabilities_are_coherent():
    datasets = [_dataset(701 + index, index * 7) for index in range(6)]
    artifact, _, _ = fit_strict_mixture(datasets, specs=_specs())
    predicted = predict_live_strict(artifact, _live_state())
    assert len(predicted) == 3
    np.testing.assert_allclose(
        predicted[["p_green", "p_neutralized", "p_pit"]].sum(axis=1).to_numpy(),
        1.0,
        atol=1e-10,
    )
    assert (predicted.green_lap_lower_s < predicted.predicted_green_lap_s).all()
    assert (predicted.predicted_green_lap_s < predicted.green_lap_upper_s).all()
    assert predicted.feature_policy.eq("strict_asof_only").all()


def test_strict_pace_drives_race_simulation_and_is_exposed_in_audit():
    datasets = [_dataset(801 + index, index * 7) for index in range(6)]
    artifact, _, _ = fit_strict_mixture(datasets, specs=_specs())
    config = SimulationConfig(
        samples=1000,
        seed=11,
        safety_car_hazard_per_lap=0.0,
        dnf_hazard_per_lap=0.0,
    )
    report = combined_live_report(_live_state(), 18, artifact, config)
    assert len(report["pace_predictions"]) == 3
    assert report["pace_model"]["selected_regressor"] == artifact["selected_regressor"]
    assert report["pace_model"]["override_drivers"] == [1, 2, 3]
    expected_source = (
        "strict_residual_next_lap_conformal"
        if artifact["pace_prediction_mode"] == RESIDUAL_MODE
        else "strict_recent_median_baseline_conformal"
    )
    assert set(report["audit"]["pace_sources"].values()) == {expected_source}
    np.testing.assert_allclose(
        sum(row["win_probability"] for row in report["predictions"]),
        1.0,
        atol=1e-10,
    )
