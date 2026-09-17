import json
from pathlib import Path

import pandas as pd
import pytest

from f1_research.strategy import SimulationConfig
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


def _write_raw(root: Path, session_key: int, max_lap: int, sc_laps: tuple[int, ...],
               results: list[dict] | None = None):
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
    (session / "session_result.json").write_text(json.dumps(results or []), encoding="utf-8")


def _results(session_key: int, finishers: int = 18, dnfs: int = 2, laps: int = 55):
    rows = []
    for driver in range(1, finishers + 1):
        rows.append({"session_key": session_key, "driver_number": driver,
                     "number_of_laps": laps, "dnf": False, "dns": False, "dsq": False})
    for offset in range(dnfs):
        rows.append({"session_key": session_key, "driver_number": 100 + offset,
                     "number_of_laps": 20 + offset * 5, "dnf": True, "dns": False, "dsq": False})
    return rows


def test_calibrates_public_pit_sc_and_dnf_priors(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, 501, 50, (10,), _results(501))
    _write_raw(raw, 502, 60, (20, 40), _results(502))
    datasets = [_dataset(501), _dataset(502), _dataset(503)]

    priors, audit = calibrate_strategy_priors(datasets, raw, [501, 502])
    assert priors.pit_observations == 6
    assert priors.pit_loss_mean_s == pytest.approx(22.0, abs=0.25)
    assert 0.5 <= priors.pit_loss_sd_s <= 2.0
    assert priors.safety_car_starts == 3
    assert priors.race_laps_observed == 110
    assert priors.safety_car_hazard_per_lap == pytest.approx(3 / 110)
    assert priors.dnf_events == 4
    assert priors.car_laps_observed > 2000
    assert priors.dnf_source == "session_result_dnf_per_car_lap_exposure"
    assert priors.dnf_hazard_per_lap == pytest.approx(priors.dnf_events / priors.car_laps_observed)
    assert audit["dnf_source"] == priors.dnf_source


def test_insufficient_result_exposure_keeps_explicit_dnf_fallback(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, 551, 20, (), [
        {"session_key": 551, "driver_number": 1, "number_of_laps": 10,
         "dnf": True, "dns": False, "dsq": False},
    ])
    priors, audit = calibrate_strategy_priors([_dataset(551)], raw, [551])
    assert priors.dnf_source == "fallback_insufficient_session_result_exposure"
    assert priors.dnf_hazard_per_lap == pytest.approx(SimulationConfig().dnf_hazard_per_lap)
    assert audit["dnf_source"] == priors.dnf_source


def test_strategy_prior_roundtrip_builds_simulation_config(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, 601, 55, (12,), _results(601))
    datasets = [_dataset(601), _dataset(602), _dataset(603)]
    priors, audit = calibrate_strategy_priors(datasets, raw, [601])
    path = tmp_path / "strategy_priors.json"
    save_strategy_priors(priors, audit, path)

    config, payload = load_simulation_config(path, samples=4321, seed=9)
    assert config.samples == 4321
    assert config.seed == 9
    assert config.pit_loss_mean_s == pytest.approx(priors.pit_loss_mean_s)
    assert config.safety_car_hazard_per_lap == pytest.approx(priors.safety_car_hazard_per_lap)
    assert config.dnf_hazard_per_lap == pytest.approx(priors.dnf_hazard_per_lap)
    assert payload["priors"]["dnf_source"] == priors.dnf_source
