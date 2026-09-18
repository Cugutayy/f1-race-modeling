import numpy as np
import pytest

from f1_research.strategy import (
    DriverInput,
    SimulationConfig,
    _sample_first_event_lap,
    simulate,
)


def test_first_safety_car_lap_matches_discrete_hazard_probability():
    samples = 120_000
    laps = 8
    hazard = 0.08
    event_lap = _sample_first_event_lap(
        np.random.default_rng(12345),
        samples,
        laps,
        hazard,
    )
    expected_any = 1 - (1 - hazard) ** laps
    assert float(np.mean(event_lap > 0)) == pytest.approx(expected_any, abs=0.004)
    assert float(np.mean(event_lap == 1)) == pytest.approx(hazard, abs=0.003)
    assert event_lap.min() == 0
    assert event_lap.max() <= laps


def test_zero_safety_car_hazard_never_schedules_an_event():
    event_lap = _sample_first_event_lap(np.random.default_rng(7), 1000, 20, 0.0)
    assert np.count_nonzero(event_lap) == 0



def test_dnf_event_time_matches_discrete_hazard_and_classification_threshold():
    hazard = 0.10
    laps_remaining = 7
    drivers = [
        DriverInput(
            1, "A", 1, 0.0, 90.0, 0.0, 0.0, 0, "MEDIUM",
            dnf_hazard_per_lap=hazard,
        ),
        DriverInput(
            2, "B", 2, 1.0, 90.0, 0.0, 0.0, 0, "MEDIUM",
            dnf_hazard_per_lap=0.0,
        ),
    ]
    config = SimulationConfig(
        samples=30_000,
        seed=2026,
        lap_noise_s=0.0,
        safety_car_hazard_per_lap=0.0,
        dnf_hazard_per_lap=0.0,
        traffic_penalty_mean_s=0.0,
        traffic_penalty_sd_s=0.0,
    )
    results, audit = simulate(
        drivers,
        laps_remaining,
        config=config,
        completed_laps_at_start=50,
    )
    first = next(row for row in results if row.driver_number == 1)
    expected_dnf = 1 - (1 - hazard) ** laps_remaining
    assert first.dnf_probability == pytest.approx(expected_dnf, abs=0.008)
    # Winner completes 57 laps -> floor(0.9 * 57) = 51. Only a failure in
    # the first future interval leaves this car on 50 laps and unclassified.
    assert first.classified_probability == pytest.approx(1 - hazard, abs=0.008)
    assert first.expected_completed_laps < 57.0
    assert 1.0 <= first.expected_retirement_lap <= laps_remaining
    assert audit["classification_threshold_laps"] == {"min": 51, "max": 51}
    assert "first failure interval" in audit["dnf_event_time_model"]


def test_no_dnf_keeps_every_sample_classified():
    drivers = [
        DriverInput(1, "A", 1, 0.0, 90.0, 0.0, 0.0, 0, "MEDIUM", dnf_hazard_per_lap=0.0),
        DriverInput(2, "B", 2, 2.0, 90.1, 0.0, 0.0, 0, "MEDIUM", dnf_hazard_per_lap=0.0),
    ]
    config = SimulationConfig(
        samples=1000,
        seed=4,
        lap_noise_s=0.0,
        safety_car_hazard_per_lap=0.0,
        dnf_hazard_per_lap=0.0,
        traffic_penalty_mean_s=0.0,
        traffic_penalty_sd_s=0.0,
    )
    results, _ = simulate(drivers, 3, config=config, completed_laps_at_start=20)
    assert all(row.dnf_probability == 0.0 for row in results)
    assert all(row.classified_probability == 1.0 for row in results)
    assert all(row.expected_completed_laps == 23.0 for row in results)
