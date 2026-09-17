import json

import pandas as pd
import pytest

from f1_research.strategy import DriverInput, SimulationConfig, Strategy, simulate
from f1_research.strategy_calibration import load_simulation_config
from f1_research.tyre_calibration import as_payload, calibrate_tyre_priors, maps_from_payload


def _mixed_compound_sessions() -> list[pd.DataFrame]:
    frames = []
    for session_key in range(700, 706):
        rows = []
        for driver in range(1, 5):
            soft_first = driver <= 2
            for lap in range(1, 13):
                first = lap <= 6
                stint = 1 if first else 2
                age = lap if first else lap - 6
                compound = (
                    "SOFT"
                    if (first and soft_first) or (not first and not soft_first)
                    else "MEDIUM"
                )
                compound_delta = -0.40 if compound == "SOFT" else 0.0
                degradation = 0.10 if compound == "SOFT" else 0.02
                common_track = -0.03 * lap
                target = 90.0 + common_track + compound_delta + degradation * age
                rows.append({
                    "session_key": session_key,
                    "driver_number": str(driver),
                    "lap_number": lap,
                    "target_s": target,
                    "target_valid": True,
                    "lap_regime": "green",
                    "compound": compound,
                    "tyre_age": age,
                    "stint_number": stint,
                    "safety_car_active": 0.0,
                    "is_pit_out_lap": False,
                })
        frames.append(pd.DataFrame(rows))
    return frames


def test_tyre_calibration_recovers_relative_compound_signal_and_pit_age():
    model, audit = calibrate_tyre_priors(_mixed_compound_sessions())
    assert model.enabled is True
    assert audit["clean_green_laps"] > 200
    assert audit["eligible_stints"] >= 40
    assert model.pace_delta_s["MEDIUM"] == pytest.approx(0.0)
    # Field-median residualization deliberately removes common race pace and can
    # attenuate the raw synthetic -0.40 s offset. The evidence claim is therefore
    # directional/relative, not recovery of a causal tyre coefficient.
    assert model.pace_delta_s["SOFT"] < -0.10
    assert model.pace_delta_s["SOFT"] < model.pace_delta_s["MEDIUM"]
    assert model.relative_degradation_s_per_lap["SOFT"] > model.relative_degradation_s_per_lap["MEDIUM"]
    assert model.observed_pit_age_p50["SOFT"] == pytest.approx(6.0)
    assert model.observed_pit_age_p50["MEDIUM"] == pytest.approx(6.0)
    assert model.pace_delta_s["HARD"] == pytest.approx(0.35)

    pace, degradation, pit_age = maps_from_payload(as_payload(model))
    assert pace["SOFT"] == pytest.approx(model.pace_delta_s["SOFT"])
    assert degradation["SOFT"] == pytest.approx(model.relative_degradation_s_per_lap["SOFT"])
    assert pit_age["SOFT"] == pytest.approx(6.0)


def test_tyre_calibration_gracefully_falls_back_when_stint_fields_are_unavailable():
    legacy = pd.DataFrame({
        "session_key": [1, 1],
        "driver_number": ["1", "2"],
        "lap_number": [5, 5],
        "target_s": [90.0, 91.0],
        "target_valid": [True, True],
        "lap_regime": ["green", "green"],
    })
    model, audit = calibrate_tyre_priors([legacy])
    assert model.enabled is False
    assert audit["clean_green_laps"] == 0
    assert audit["eligible_stints"] == 0
    pace, degradation, pit_age = maps_from_payload(as_payload(model))
    assert pace["SOFT"] == pytest.approx(-0.35)
    assert degradation["MEDIUM"] == pytest.approx(0.06)
    assert pit_age["HARD"] == pytest.approx(40.0)


def test_strategy_prior_loader_applies_tyre_maps(tmp_path):
    model, _ = calibrate_tyre_priors(_mixed_compound_sessions())
    payload = {
        "schema_version": 4,
        "priors": {
            "pit_loss_mean_s": 22.0,
            "pit_loss_sd_s": 1.5,
            "dnf_hazard_per_lap": 0.001,
            "safety_car_hazard_per_lap": 0.01,
        },
        "tyre": as_payload(model),
    }
    path = tmp_path / "strategy_priors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config, loaded = load_simulation_config(path, samples=2000, seed=11)
    assert loaded["schema_version"] == 4
    assert config.compound_pace_delta_s["SOFT"] == pytest.approx(model.pace_delta_s["SOFT"])
    assert config.compound_degradation_s_per_lap["SOFT"] == pytest.approx(
        model.relative_degradation_s_per_lap["SOFT"]
    )
    assert config.compound_stint_target_laps["SOFT"] == pytest.approx(6.0)


def test_simulator_uses_post_pit_compound_pace_prior():
    drivers = [
        DriverInput(1, "A", 1, 0.0, 90.0, 0.01, 0.05, 10, "MEDIUM", dnf_hazard_per_lap=0.0),
        DriverInput(2, "B", 2, 3.0, 90.2, 0.01, 0.05, 10, "MEDIUM", dnf_hazard_per_lap=0.0),
    ]
    config = SimulationConfig(
        samples=3000,
        seed=5,
        lap_noise_s=0.0,
        pit_loss_mean_s=22.0,
        pit_loss_sd_s=0.0,
        safety_car_hazard_per_lap=0.0,
        dnf_hazard_per_lap=0.0,
        compound_pace_delta_s={
            "SOFT": -1.0, "MEDIUM": 0.0, "HARD": 1.0, "INTERMEDIATE": 2.5, "WET": 5.0,
        },
        compound_degradation_s_per_lap={
            "SOFT": 0.02, "MEDIUM": 0.05, "HARD": 0.02, "INTERMEDIATE": 0.06, "WET": 0.05,
        },
    )
    soft, _ = simulate(drivers, 8, {1: Strategy(1, "SOFT")}, config)
    hard, _ = simulate(drivers, 8, {1: Strategy(1, "HARD")}, config)
    soft_driver = next(row for row in soft if row.driver_number == 1)
    hard_driver = next(row for row in hard if row.driver_number == 1)
    assert soft_driver.mean_remaining_time_s + 10.0 < hard_driver.mean_remaining_time_s
