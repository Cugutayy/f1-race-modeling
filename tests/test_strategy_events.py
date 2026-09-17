import numpy as np
import pytest

from f1_research.strategy import _sample_first_event_lap


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
