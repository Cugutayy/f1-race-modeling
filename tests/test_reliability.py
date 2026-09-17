import numpy as np
import pytest

from f1_research.reliability import (
    ReliabilityRecord,
    calibrate_reliability,
    predict_hazard,
    reliability_overrides_from_state,
)
from f1_research.strategy import SimulationConfig, predict_from_state


def _records() -> list[ReliabilityRecord]:
    rows = []
    # Six chronological races. Team Fragile has more failures, especially driver 1;
    # Team Solid mostly finishes. Exposure remains large enough for the model to turn on.
    for session_order in range(6):
        session_key = 100 + session_order
        for driver, team in ((1, "Fragile"), (2, "Fragile"), (3, "Solid"), (4, "Solid")):
            failure = 0
            completed = 55
            if driver == 1 and session_order in (1, 3, 5):
                failure, completed = 1, 24 + session_order
            elif driver == 2 and session_order == 4:
                failure, completed = 1, 37
            elif driver == 4 and session_order == 2:
                failure, completed = 1, 48
            rows.append(ReliabilityRecord(
                session_key=session_key,
                session_order=session_order,
                driver_number=driver,
                team_name=team,
                failure=failure,
                exposure=completed + failure,
            ))
    return rows


def test_hierarchical_reliability_shrinks_team_and_driver_rates():
    model, audit = calibrate_reliability(_records(), pooled_fallback=0.0018)
    assert model.enabled is True
    assert audit["tuning_source"] == "forward_survival_likelihood"
    assert audit["trial_count"] > 0

    fragile_1 = predict_hazard(model, 1, "Fragile")
    fragile_2 = predict_hazard(model, 2, "Fragile")
    solid_3 = predict_hazard(model, 3, "Solid")
    unseen = predict_hazard(model, 99, "New Team")

    assert 0 < solid_3 < fragile_1 < 0.05
    assert solid_3 < fragile_2 < 0.05
    assert unseen == pytest.approx(model.pooled_hazard_per_lap)
    # Sparse evidence must not turn a few DNFs into a near-certain per-lap failure.
    assert fragile_1 < 5 * model.pooled_hazard_per_lap


def test_live_overrides_follow_current_team_and_keep_unseen_driver_safe():
    model, _ = calibrate_reliability(_records(), pooled_fallback=0.0018)
    state = {"drivers": [
        {"driver_number": 1, "team_name": "Fragile"},
        {"driver_number": 3, "team_name": "Solid"},
        {"driver_number": 99, "team_name": "New Team"},
    ]}
    overrides = reliability_overrides_from_state(state, model)
    assert overrides[1] > overrides[3]
    assert overrides[99] == pytest.approx(model.pooled_hazard_per_lap)


def test_driver_specific_hazard_changes_simulated_dnf_probability():
    state = {
        "session_key": 9,
        "current_lap": 5,
        "drivers": [
            {"driver_number": 1, "acronym": "A", "position": 1, "gap_to_leader_s": 0.0,
             "recent_laps_s": [90.0, 90.0, 90.0, 90.0], "last_lap_s": 90.0,
             "compound": "MEDIUM", "tyre_age": 5, "pit_stops": 0},
            {"driver_number": 2, "acronym": "B", "position": 2, "gap_to_leader_s": 2.0,
             "recent_laps_s": [90.1, 90.1, 90.1, 90.1], "last_lap_s": 90.1,
             "compound": "MEDIUM", "tyre_age": 5, "pit_stops": 0},
        ],
    }
    report = predict_from_state(
        state,
        total_laps=25,
        config=SimulationConfig(samples=12_000, seed=11, dnf_hazard_per_lap=0.001,
                                safety_car_hazard_per_lap=0.0),
        dnf_hazard_overrides={1: 0.010, 2: 0.0001},
    )
    by_driver = {row["driver_number"]: row for row in report["predictions"]}
    assert by_driver[1]["dnf_probability"] > 0.15
    assert by_driver[2]["dnf_probability"] < 0.01
    assert by_driver[1]["dnf_probability"] > 10 * by_driver[2]["dnf_probability"]
    assert np.isfinite(by_driver[1]["expected_position"])
    assert report["audit"]["dnf_hazards_per_lap"] == {"1": 0.01, "2": 0.0001}
