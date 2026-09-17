from datetime import UTC, datetime, timedelta

import pytest

from f1_research.data_truth import (
    assert_trusted_live_state,
    parse_provider_timestamp,
    validate_simulation_observations,
)


def _state(now: datetime):
    return {
        "updated_at": now.isoformat(),
        "latest_provider_event_at": now.isoformat(),
        "drivers": [
            {
                "driver_number": 1,
                "position": 1,
                "gap_to_leader_s": 0.0,
                "recent_laps_s": [90.1, 90.0, 89.9],
                "compound": "MEDIUM",
                "tyre_age": 9,
            },
            {
                "driver_number": 2,
                "position": 2,
                "gap_to_leader_s": 2.4,
                "recent_laps_s": [90.4, 90.2, 90.3],
                "compound": "HARD",
                "tyre_age": 14,
            },
        ],
    }


def test_invalid_provider_timestamp_is_not_coerced_to_now():
    assert parse_provider_timestamp("not-a-timestamp") is None
    assert parse_provider_timestamp(None) is None
    assert parse_provider_timestamp("2026-09-17T12:00:00Z") is not None


def test_trusted_live_state_accepts_complete_fresh_observations():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    audit = assert_trusted_live_state(_state(now), max_age_s=20, now=now + timedelta(seconds=4))
    assert audit.status == "trusted_live"
    assert audit.state_age_s == pytest.approx(4.0)
    assert audit.provider_event_age_s == pytest.approx(4.0)
    assert audit.missing_fields == {}


def test_trusted_live_state_requires_explicit_provider_event_time():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["latest_provider_event_at"] = None
    with pytest.raises(ValueError, match="provider event timestamp"):
        assert_trusted_live_state(state, max_age_s=20, now=now)


def test_trusted_live_state_rejects_stale_provider_event_even_if_received_recently():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["latest_provider_event_at"] = (now - timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="provider event age"):
        assert_trusted_live_state(state, max_age_s=20, now=now)


def test_trusted_live_state_rejects_stale_receive_time_even_with_fresh_provider_event():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["updated_at"] = (now - timedelta(minutes=2)).isoformat()
    with pytest.raises(ValueError, match="receive/update age"):
        assert_trusted_live_state(state, max_age_s=20, now=now)


def test_trusted_live_state_rejects_implausibly_future_provider_time():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["latest_provider_event_at"] = (now + timedelta(seconds=30)).isoformat()
    with pytest.raises(ValueError, match="in the future"):
        assert_trusted_live_state(state, max_age_s=20, future_tolerance_s=5, now=now)


def test_missing_gap_compound_tyre_age_or_pace_is_not_silently_imputed():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["drivers"][1]["gap_to_leader_s"] = None
    state["drivers"][1]["compound"] = None
    state["drivers"][1]["tyre_age"] = None
    state["drivers"][1]["recent_laps_s"] = []
    missing = validate_simulation_observations(state)
    assert set(missing["2"]) == {
        "compound", "gap_to_leader_s", "pace_observation", "tyre_age"
    }
    with pytest.raises(ValueError, match="incomplete"):
        assert_trusted_live_state(state, max_age_s=20, now=now)


def test_leader_gap_may_be_inferred_as_zero_but_nonleader_gap_may_not():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["drivers"][0]["gap_to_leader_s"] = None
    assert validate_simulation_observations(state) == {}


def test_duplicate_or_nonconsecutive_positions_fail_truth_validation():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    duplicate = _state(now)
    duplicate["drivers"][1]["position"] = 1
    missing = validate_simulation_observations(duplicate)
    assert "duplicate_positions" in missing["state"]

    nonconsecutive = _state(now)
    nonconsecutive["drivers"][1]["position"] = 3
    missing = validate_simulation_observations(nonconsecutive)
    assert "non_consecutive_positions" in missing["state"]


def test_gap_order_must_match_race_position_order():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["drivers"].append({
        "driver_number": 3,
        "position": 3,
        "gap_to_leader_s": 1.0,
        "recent_laps_s": [90.5, 90.4, 90.6],
        "compound": "SOFT",
        "tyre_age": 7,
    })
    missing = validate_simulation_observations(state)
    assert "non_monotonic_gaps" in missing["state"]
