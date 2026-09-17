import pandas as pd
import pytest

from f1_research.asof_contract import assert_target_unavailable, enforce_asof


def test_asof_accepts_information_available_at_cutoff():
    frame = pd.DataFrame({"quali_position": [1], "quali_known_at": ["2026-03-07T07:59:00Z"]})
    assert len(enforce_asof(
        frame, cutoff_at="2026-03-07T08:00:00Z",
        timestamp_columns={"quali_position": "quali_known_at"},
    )) == 1


def test_asof_rejects_future_information():
    frame = pd.DataFrame({"weather": [22.0], "weather_known_at": ["2026-03-07T08:01:00Z"]})
    with pytest.raises(ValueError, match="leaks 1"):
        enforce_asof(frame, cutoff_at="2026-03-07T08:00:00Z",
                     timestamp_columns={"weather": "weather_known_at"})


def test_asof_rejects_unknown_publication_time():
    frame = pd.DataFrame({"grid": [1], "grid_known_at": [None]})
    with pytest.raises(ValueError, match="unknown information timestamps"):
        enforce_asof(frame, cutoff_at="2026-03-07T08:00:00Z",
                     timestamp_columns={"grid": "grid_known_at"})


def test_inference_target_outcomes_are_forbidden():
    with pytest.raises(ValueError, match="target-race outcome"):
        assert_target_unavailable(pd.DataFrame({"driver": ["VER"], "finish_position": [1]}))
