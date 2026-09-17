from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from f1_research.lap_intelligence import LapModelSpec, build_lap_dataset
from f1_research.lap_mixture import attach_regime_labels, fit_mixture, predict_live_mixture


def _dataset(session_key: int, day: int):
    base = datetime(2026, 2, 1, 12, tzinfo=UTC) + timedelta(days=day)
    laps = []
    stints = []
    pits = []
    for driver in range(1, 7):
        for lap in range(1, 17):
            pace = 89.0 + driver * 0.14 + lap * 0.015 + day * 0.002
            laps.append({
                "session_key": session_key,
                "driver_number": driver,
                "lap_number": lap,
                "date_start": (base + timedelta(seconds=(lap - 1) * 92 + driver * 0.1)).isoformat(),
                "lap_duration": pace + (22.0 if lap == 10 and driver in (1, 3) else 0.0),
                "is_pit_out_lap": lap == 11 and driver in (1, 3),
            })
        stints.append({
            "session_key": session_key, "driver_number": driver, "stint_number": 1,
            "lap_start": 1, "lap_end": 10, "compound": "MEDIUM", "tyre_age_at_start": 0,
        })
        stints.append({
            "session_key": session_key, "driver_number": driver, "stint_number": 2,
            "lap_start": 11, "lap_end": 16, "compound": "HARD", "tyre_age_at_start": 0,
        })
        if driver in (1, 3):
            pits.append({
                "session_key": session_key, "driver_number": driver, "lap_number": 10,
                "date": (base + timedelta(seconds=9 * 92 + 70 + driver * 0.1)).isoformat(),
                "lane_duration": 22.0,
            })
    weather = [{
        "session_key": session_key, "date": base.isoformat(), "air_temperature": 24.0,
        "track_temperature": 37.0, "humidity": 52.0, "rainfall": 0,
    }]
    dataset = build_lap_dataset(laps, stint_rows=stints, weather_rows=weather, pit_rows=pits)
    return attach_regime_labels(dataset, laps, pits)


def test_regime_labels_use_pit_lap_and_pit_out_targets():
    frame = _dataset(201, 0)
    assert set(frame.lap_regime.astype(str)) == {"green", "pit"}
    assert frame[(frame.driver_number == "1") & (frame.lap_number == 10)].iloc[0].lap_regime == "pit"
    assert frame[(frame.driver_number == "1") & (frame.lap_number == 11)].iloc[0].lap_regime == "pit"
    assert frame[(frame.driver_number == "2") & (frame.lap_number == 10)].iloc[0].lap_regime == "green"


def test_mixture_uses_disjoint_calibration_and_sealed_test():
    datasets = [_dataset(201 + index, index * 7) for index in range(4)]
    artifact, metrics, audit = fit_mixture(
        datasets,
        specs=(LapModelSpec("hist_gradient_boosting", {"max_iter": 25, "min_samples_leaf": 5}),),
        alpha=0.10,
    )
    row = metrics.iloc[0]
    assert artifact["task"] == "next_lap_mixture"
    assert artifact["trained_through_session"] == 202
    assert artifact["calibration_session"] == 203
    assert artifact["sealed_test_session"] == 204
    assert artifact["conformal_radius_s"] > 0
    assert 0 <= row.interval_coverage <= 1
    assert np.isfinite(row.green_mae_s)
    assert np.isfinite(row.regime_log_loss)
    assert audit["test_updates_model"] is False


def test_live_mixture_probabilities_are_coherent_and_interval_is_ordered():
    datasets = [_dataset(301 + index, index * 7) for index in range(4)]
    artifact, _, _ = fit_mixture(
        datasets,
        specs=(LapModelSpec("hist_gradient_boosting", {"max_iter": 20, "min_samples_leaf": 5}),),
    )
    state = {
        "current_lap": 12,
        "flag": "GREEN",
        "safety_car": None,
        "weather": {"air_temperature_c": 24.0, "track_temperature_c": 37.0,
                    "humidity_pct": 52.0, "rainfall": False},
        "drivers": [{
            "driver_number": driver, "lap_number": 12,
            "recent_laps_s": [89.2 + driver * 0.1, 89.25 + driver * 0.1,
                              89.22 + driver * 0.1, 89.28 + driver * 0.1],
            "compound": "HARD", "tyre_age": 2, "stint_number": 2, "pit_stops": 1,
        } for driver in range(1, 4)],
    }
    predicted = predict_live_mixture(artifact, state)
    assert len(predicted) == 3
    np.testing.assert_allclose(
        predicted[["p_green", "p_neutralized", "p_pit"]].sum(axis=1).to_numpy(), 1.0,
        atol=1e-10,
    )
    assert (predicted.green_lap_lower_s < predicted.predicted_green_lap_s).all()
    assert (predicted.predicted_green_lap_s < predicted.green_lap_upper_s).all()
    assert (predicted.p_neutralized >= 0).all()


def test_mixture_requires_four_races():
    with pytest.raises(ValueError, match="four|train"):
        fit_mixture([_dataset(401, 0), _dataset(402, 7), _dataset(403, 14)])
