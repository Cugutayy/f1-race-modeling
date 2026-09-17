import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from f1_research.traffic_calibration import (
    as_payload,
    calibrate_traffic_prior,
    config_values_from_payload,
)


def _session_frame(session_key: int, start: datetime, close_residual: float = 0.24) -> pd.DataFrame:
    rows = []
    for lap in range(1, 13):
        close = lap <= 6
        rows.append({
            "session_key": session_key,
            "driver_number": "1",
            "lap_number": lap,
            "forecast_at": start + timedelta(seconds=90 * lap),
            "target_s": 90.0 + (close_residual if close else 0.0),
            "recent_median_5_s": 90.0,
            "target_valid": True,
            "lap_regime": "green",
            "safety_car_active": 0.0,
            "is_pit_out_lap": False,
        })
    return pd.DataFrame(rows)


def _write_intervals(root, session_key: int, start: datetime, close_s: float = 0.7):
    rows = []
    for lap in range(1, 13):
        close = lap <= 6
        rows.append({
            "session_key": session_key,
            "driver_number": 1,
            "date": (start + timedelta(seconds=90 * lap - 2)).isoformat(),
            "interval": close_s if close else 4.0,
        })
    directory = root / str(session_key)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "intervals.json").write_text(json.dumps(rows), encoding="utf-8")


def test_traffic_calibration_recovers_within_driver_close_following_delta(tmp_path):
    raw = tmp_path / "raw"
    datasets = []
    session_keys = []
    for offset in range(6):
        session_key = 800 + offset
        start = datetime(2026, 1, 1 + offset, 12, tzinfo=UTC)
        datasets.append(_session_frame(session_key, start))
        _write_intervals(raw, session_key, start)
        session_keys.append(session_key)

    model, audit = calibrate_traffic_prior(datasets, raw, session_keys)
    assert model.enabled is True
    assert model.driver_session_contrasts == 6
    assert model.close_laps == 36
    assert model.clear_laps == 36
    assert model.raw_close_minus_clear_s == pytest.approx(0.24)
    assert model.penalty_mean_s == pytest.approx(0.24)
    assert model.penalty_sd_s >= 0.02
    assert audit["sessions_with_intervals"] == 6

    window, mean, sd = config_values_from_payload(as_payload(model))
    assert window == pytest.approx(1.2)
    assert mean == pytest.approx(0.24)
    assert sd == pytest.approx(model.penalty_sd_s)


def test_traffic_calibration_falls_back_when_intervals_are_unavailable(tmp_path):
    start = datetime(2026, 2, 1, 12, tzinfo=UTC)
    dataset = _session_frame(900, start)
    model, audit = calibrate_traffic_prior(
        [dataset],
        tmp_path / "raw",
        [900],
        fallback_mean_s=0.15,
        fallback_sd_s=0.07,
    )
    assert model.enabled is False
    assert model.source == "fallback_insufficient_interval_contrasts"
    assert model.penalty_mean_s == pytest.approx(0.15)
    assert model.penalty_sd_s == pytest.approx(0.07)
    assert audit["joined_laps"] == 0


def test_negative_observed_close_effect_never_invents_simulator_speed_boost(tmp_path):
    raw = tmp_path / "raw"
    datasets = []
    keys = []
    for offset in range(6):
        key = 950 + offset
        start = datetime(2026, 3, 1 + offset, 12, tzinfo=UTC)
        datasets.append(_session_frame(key, start, close_residual=-0.18))
        _write_intervals(raw, key, start)
        keys.append(key)

    model, audit = calibrate_traffic_prior(datasets, raw, keys)
    assert model.enabled is True
    assert model.raw_close_minus_clear_s == pytest.approx(-0.18)
    assert model.penalty_mean_s == pytest.approx(0.0)
    assert audit["raw_close_minus_clear_s"] == pytest.approx(-0.18)
