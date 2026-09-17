import json
from pathlib import Path

import pandas as pd
import pytest

from f1_research.strategy_calibration import (
    calibrate_strategy_priors,
    load_simulation_config,
    save_strategy_priors,
)


def _dataset(session_key: int) -> pd.DataFrame:
    rows = []
    for driver in range(1, 5):
        for lap in range(6, 16):
            pit = lap == 10 and driver in (1, 3)
            rows.append({
                "session_key": session_key,
                "driver_number": str(driver),
                "lap_number": lap,
                "target_valid": True,
                "lap_regime": "pit" if pit else "green",
                "is_pit_out_lap": False,
                "target_s": 112.0 + driver * 0.1 if pit else 90.0 + driver * 0.1,
                "recent_median_5_s": 90.0 + driver * 0.1,
            })
    return pd.DataFrame(rows)


def _write_raw(root: Path, session_key: int, max_lap: int, sc_laps: tuple[int, ...]):
    session = root / str(session_key)
    session.mkdir(parents=True)
    laps = [{"session_key": session_key, "driver_number": 1, "lap_number": lap}
            for lap in range(1, max_lap + 1)]
    control = []
    for lap in sc_laps:
        control.append({
            "session_key": session_key,
            "lap_number": lap,
            "date": f"2026-01-01T12:{lap:02d}:00Z",
            "category": "SafetyCar",
            "message": "SAFETY CAR DEPLOYED",
        })
    (session / "laps.json").write_text(json.dumps(laps), encoding="utf-8")
    (session / "race_control.json").write_text(json.dumps(control), encoding="utf-8")


def test_calibrates_public_pit_and_sc_priors_but_not_dnf(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, 501, 50, (10,))
    _write_raw(raw, 502, 60, (20, 40))
    datasets = [_dataset(501), _dataset(502), _dataset(503)]

    priors, audit = calibrate_strategy_priors(datasets, raw, [501, 502])
    assert priors.pit_observations == 6
    assert priors.pit_loss_mean_s == pytest.approx(22.0, abs=0.25)
    assert 0.5 <= priors.pit_loss_sd_s <= 2.0
    assert priors.safety_car_starts == 3
    assert priors.race_laps_observed == 110
    assert priors.safety_car_hazard_per_lap == pytest.approx(3 / 110)
    assert priors.dnf_source == "default_not_calibrated"
    assert "not calibrated" in audit["dnf_hazard"]


def test_strategy_prior_roundtrip_builds_simulation_config(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, 601, 55, (12,))
    datasets = [_dataset(601), _dataset(602), _dataset(603)]
    priors, audit = calibrate_strategy_priors(datasets, raw, [601])
    path = tmp_path / "strategy_priors.json"
    save_strategy_priors(priors, audit, path)

    config, payload = load_simulation_config(path, samples=4321, seed=9)
    assert config.samples == 4321
    assert config.seed == 9
    assert config.pit_loss_mean_s == pytest.approx(priors.pit_loss_mean_s)
    assert config.safety_car_hazard_per_lap == pytest.approx(priors.safety_car_hazard_per_lap)
    assert payload["priors"]["dnf_source"] == "default_not_calibrated"
