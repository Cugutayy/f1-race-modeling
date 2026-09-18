import numpy as np
import pytest

from f1_research.race_classification import classify_completed_laps, minimum_classified_laps


def test_fia_ninety_percent_threshold_is_rounded_down():
    np.testing.assert_array_equal(
        minimum_classified_laps(np.array([57, 58, 10, 1, 0])),
        np.array([51, 52, 9, 0, 0]),
    )


def test_completed_laps_dominate_crossing_time_and_same_lap_uses_line_order():
    laps = np.array([
        [57, 56, 51, 50],
        [10, 10, 9, 8],
    ])
    times = np.array([
        [100.0, 90.0, 80.0, 70.0],
        [100.0, 99.0, 200.0, 50.0],
    ])
    ranks, classified, thresholds = classify_completed_laps(laps, times)
    np.testing.assert_array_equal(ranks[0], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(ranks[1], np.array([2, 1, 3, 4]))
    np.testing.assert_array_equal(thresholds, np.array([51, 9]))
    np.testing.assert_array_equal(classified[0], np.array([True, True, True, False]))
    np.testing.assert_array_equal(classified[1], np.array([True, True, True, False]))


def test_classification_rejects_fractional_laps_and_nonfinite_times():
    with pytest.raises(ValueError, match="whole numbers"):
        classify_completed_laps(np.array([[10.5, 10.0]]), np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="finite"):
        classify_completed_laps(np.array([[10, 9]]), np.array([[1.0, np.nan]]))
