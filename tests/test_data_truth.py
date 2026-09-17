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
    assert audit.provider_event_age_s == pytest.approx(4.0)
    assert audit.missing_fields == {}


def test_trusted_live_state_rejects_stale_provider_event_even_if_received_recently():
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    state = _state(now)
    state["updated_at"] = (now + timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="stale"):
        assert_trusted_live_state(state, max_age_s=20, now=now + timedelta(minutes=5))


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
