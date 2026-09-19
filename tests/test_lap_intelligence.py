from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from f1_research.lap_intelligence import (
    LapModelSpec,
    benchmark_lap_models,
    build_lap_dataset,
    live_feature_rows,
)


def _session(session_key: int, start_offset: int = 0, drivers: int = 4, laps: int = 12):
    base = datetime(2026, 1, 1, 12, tzinfo=UTC) + timedelta(days=start_offset)
    lap_rows = []
    stint_rows = []
    for driver in range(1, drivers + 1):
        for lap in range(1, laps + 1):
            pace = 89.0 + driver * 0.18 + lap * 0.018
            lap_rows.append({
                "session_key": session_key,
                "driver_number": driver,
                "lap_number": lap,
                "date_start": (base + timedelta(seconds=(lap - 1) * 92 + driver * 0.2)).isoformat(),
                "lap_duration": pace,
                "is_pit_out_lap": False,
            })
        stint_rows.append({
            "session_key": session_key,
            "driver_number": driver,
            "stint_number": 1,
            "lap_start": 1,
            "lap_end": laps,
            "compound": "MEDIUM",
            "tyre_age_at_start": 0,
        })
    weather = [{
        "session_key": session_key,
        "date": base.isoformat(),
        "air_temperature": 25.0,
        "track_temperature": 38.0,
        "humidity": 50.0,
        "rainfall": 0,
    }]
    return lap_rows, stint_rows, weather


def test_lap_dataset_respects_cutoff_and_enriches_state():
    laps, stints, weather = _session(100)
    frame = build_lap_dataset(laps, stint_rows=stints, weather_rows=weather, latency_s=1.0)
    assert not frame.empty
    assert (pd.to_datetime(frame.feature_available_at, utc=True)
            <= pd.to_datetime(frame.forecast_at, utc=True)).all()
    assert (pd.to_datetime(frame.target_available_at, utc=True)
            > pd.to_datetime(frame.forecast_at, utc=True)).all()
    assert frame.compound.eq("MEDIUM").all()
    assert frame.tyre_age.notna().all()
    assert frame.track_temperature_c.eq(38.0).all()
    assert frame.stint_feature_provenance.eq("historical_rest_no_publication_timestamp").all()
    assert frame.pace_history_policy.eq(
        "event_time_nonpit_nonneutralized_same_rain_slow_outlier_guard"
    ).all()


def test_slow_available_restart_lap_remains_truth_but_not_pace_history():
    base = datetime(2026, 1, 1, 12, tzinfo=UTC)
    rows = []
    starts = [0, 100, 200, 250, 400, 500, 600]
    durations = [90, 90, 90, 200, 90, 90, 90]
    for lap, (offset, duration) in enumerate(zip(starts, durations), start=1):
        rows.append({"session_key": 1, "driver_number": 1, "lap_number": lap,
                     "date_start": (base + timedelta(seconds=offset)).isoformat(),
                     "lap_duration": duration, "is_pit_out_lap": False})
    frame = build_lap_dataset(rows, latency_s=1.0, minimum_history=1)

    # The 200s target remains in the historical truth table. Once it becomes available,
    # the model-specific pace buffer rejects it as an extreme slow outlier instead of
    # poisoning subsequent green-pace features.
    lap4 = frame[frame.lap_number == 4].iloc[0]
    lap5 = frame[frame.lap_number == 5].iloc[0]
    lap6 = frame[frame.lap_number == 6].iloc[0]
    assert lap4.target_s == 200
    assert lap5.last_lap_s == 90
    assert lap6.last_lap_s == 90
    assert lap6.recent_median_3_s == 90
    assert lap6.recent_variability_s < 1
    assert pd.Timestamp(base + timedelta(seconds=451)) <= lap6.forecast_at


def test_modern_lap_benchmark_is_whole_session_and_finite():
    datasets = []
    for key, day in ((101, 0), (102, 7), (103, 14)):
        laps, stints, weather = _session(key, day)
        datasets.append(build_lap_dataset(laps, stint_rows=stints, weather_rows=weather))
    metrics, audit = benchmark_lap_models(
        datasets,
        specs=(LapModelSpec("hist_gradient_boosting", {"max_iter": 25, "min_samples_leaf": 5}),),
    )
    assert set(metrics.model) == {"hist_gradient_boosting", "recent_median_5", "last_lap"}
    assert np.isfinite(metrics[["mae_s", "rmse_s", "median_ae_s", "p90_ae_s"]].to_numpy()).all()
    assert audit["test_updates_model"] is False
    assert audit["selected_model"] == "hist_gradient_boosting"


def test_live_feature_rows_match_training_schema_and_prefer_clean_pace_history():
    state = {
        "current_lap": 20,
        "flag": "GREEN",
        "safety_car": None,
        "weather": {"air_temperature_c": 24.0, "track_temperature_c": 35.0,
                    "humidity_pct": 55.0, "rainfall": False},
        "drivers": [{"driver_number": 1, "lap_number": 20,
                     "recent_laps_s": [90.2, 180.0, 90.1, 90.0, 90.05],
                     "pace_laps_s": [90.2, 90.1, 90.0, 90.05],
                     "compound": "MEDIUM", "tyre_age": 12, "stint_number": 1,
                     "pit_stops": 0}],
    }
    frame = live_feature_rows(state)
    assert len(frame) == 1
    assert frame.iloc[0].lap_number == 21
    assert frame.iloc[0].compound == "MEDIUM"
    assert frame.iloc[0].recent_median_5_s < 91.0
    assert frame.iloc[0].last_lap_s == 90.05


def test_lap_dataset_parses_string_false_without_python_truthiness():
    laps, _, _ = _session(901, drivers=1, laps=7)
    for row in laps:
        row["is_pit_out_lap"] = "false"
    frame = build_lap_dataset(laps, latency_s=0.0, minimum_history=1)
    assert not frame.empty
    assert frame["is_pit_out_lap"].eq(False).all()


def test_lap_dataset_rejects_malformed_pit_out_boolean():
    laps, _, _ = _session(902, drivers=1, laps=7)
    laps[0]["is_pit_out_lap"] = "maybe"
    import pytest
    with pytest.raises(ValueError, match="is_pit_out_lap"):
        build_lap_dataset(laps, latency_s=0.0, minimum_history=1)
