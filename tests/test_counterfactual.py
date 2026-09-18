import pytest

from f1_research.counterfactual import pit_window
from f1_research.strategy import DriverInput, SimulationConfig, Strategy, predict_from_state, simulate


def _state():
    return {
        "session_key": 1, "current_lap": 10,
        "drivers": [
            {"driver_number": 1, "position": 1, "gap_to_leader_s": 0.0,
             "recent_laps_s": [90.0, 90.1, 90.2, 90.3], "last_lap_s": 90.3,
             "compound": "MEDIUM", "tyre_age": 10, "pit_stops": 0},
            {"driver_number": 2, "position": 2, "gap_to_leader_s": 2.0,
             "recent_laps_s": [90.2, 90.3, 90.4, 90.5], "last_lap_s": 90.5,
             "compound": "MEDIUM", "tyre_age": 10, "pit_stops": 0},
        ],
    }


def test_counterfactual_pit_window_is_ranked_and_complete():
    result = pit_window(
        _state(), total_laps=15, driver_number=1, offsets=(0, 2),
        compounds=("SOFT", "HARD"), config=SimulationConfig(samples=1000, seed=7),
    )
    assert len(result["scenarios"]) == 4
    assert result["best_by_expected_position"] == result["scenarios"][0]
    assert {row["compound"] for row in result["scenarios"]} == {"SOFT", "HARD"}


def test_pit_now_is_applied_before_first_future_lap():
    config = SimulationConfig(samples=1000, seed=11)
    baseline = predict_from_state(_state(), total_laps=15, config=config)
    pit_now = predict_from_state(
        _state(),
        total_laps=15,
        strategies={1: Strategy(pit_in_laps=0, next_compound="MEDIUM", label="pit-now")},
        config=config,
    )
    baseline_driver = next(row for row in baseline["predictions"] if row["driver_number"] == 1)
    pit_driver = next(row for row in pit_now["predictions"] if row["driver_number"] == 1)
    assert pit_now["audit"]["strategy_overrides"]["1"]["pit_in_laps"] == 0
    assert pit_driver["mean_remaining_time_s"] > baseline_driver["mean_remaining_time_s"] + 10.0



def test_pit_timing_reuses_same_latent_pit_loss_shock():
    drivers = [
        DriverInput(
            driver_number=1,
            label="A",
            current_position=1,
            gap_to_leader_s=0.0,
            pace_s=90.0,
            pace_uncertainty_s=0.0,
            degradation_s_per_lap=0.0,
            tyre_age=0,
            compound="MEDIUM",
            dnf_hazard_per_lap=0.0,
        ),
        DriverInput(
            driver_number=2,
            label="B",
            current_position=2,
            gap_to_leader_s=2.0,
            pace_s=90.0,
            pace_uncertainty_s=0.0,
            degradation_s_per_lap=0.0,
            tyre_age=0,
            compound="MEDIUM",
            dnf_hazard_per_lap=0.0,
        ),
    ]
    config = SimulationConfig(
        samples=1000,
        seed=19,
        lap_noise_s=0.0,
        pit_loss_mean_s=22.0,
        pit_loss_sd_s=1.6,
        dnf_hazard_per_lap=0.0,
        safety_car_hazard_per_lap=0.0,
        traffic_penalty_mean_s=0.0,
        traffic_penalty_sd_s=0.0,
        compound_degradation_s_per_lap={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    )
    now, _ = simulate(
        drivers,
        laps_remaining=4,
        strategies={1: Strategy(pit_in_laps=0, next_compound="MEDIUM")},
        config=config,
    )
    later, _ = simulate(
        drivers,
        laps_remaining=4,
        strategies={1: Strategy(pit_in_laps=1, next_compound="MEDIUM")},
        config=config,
    )
    now_driver = next(row for row in now if row.driver_number == 1)
    later_driver = next(row for row in later if row.driver_number == 1)
    assert now_driver.mean_remaining_time_s == pytest.approx(later_driver.mean_remaining_time_s, abs=1e-12)
