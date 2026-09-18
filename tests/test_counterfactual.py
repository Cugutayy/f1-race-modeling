from f1_research.counterfactual import pit_window
from f1_research.strategy import SimulationConfig, Strategy, predict_from_state


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
