from datetime import UTC, datetime, timedelta

import pytest

from f1_research.data_truth import assert_trusted_live_state, validate_simulation_observations
from f1_research.live_state import RaceStateStore
from f1_research.simulation_scope import ensure_strategy_eligible, split_simulation_scope


def _driver(number, position, gap, *, laps_behind=None):
    row = {
        "driver_number": number,
        "acronym": f"D{number}",
        "position": position,
        "gap_to_leader_s": gap,
        "laps_behind": laps_behind,
        "recent_laps_s": [89.4, 89.6, 89.5],
        "last_lap_s": 89.5,
        "compound": "MEDIUM",
        "tyre_age": 8,
        "pit_stops": 0,
    }
    return row


def _snapshot():
    now = datetime.now(UTC)
    return {
        "updated_at": (now - timedelta(seconds=1)).isoformat(),
        "latest_provider_event_at": (now - timedelta(seconds=1)).isoformat(),
        "session_key": 9693,
        "current_lap": 30,
        "drivers": [
            _driver(1, 1, 0.0),
            _driver(4, 2, 3.2),
            {
                "driver_number": 14,
                "acronym": "D14",
                "position": 3,
                "gap_to_leader_s": None,
                "laps_behind": 1,
                "gap_to_leader_raw": "+1 LAP",
                # Exact-time pace/tyre fields are intentionally absent: this row is
                # classification evidence, not a seconds-based simulation input.
            },
        ],
    }


def test_openf1_lap_deficit_is_preserved_as_discrete_unit_then_cleared_by_seconds():
    store = RaceStateStore(session_key=9693)
    first = {
        "session_key": 9693,
        "driver_number": 14,
        "date": "2025-03-16T06:10:00Z",
        "gap_to_leader": "+1 LAP",
        "interval": "+1 LAP",
    }
    assert store.ingest("intervals", first) is True
    driver = store.state.driver(14)
    assert driver.gap_to_leader_s is None
    assert driver.laps_behind == 1
    assert driver.gap_to_leader_raw == "+1 LAP"
    assert driver.interval_s is None
    assert driver.interval_laps_behind == 1

    second = {
        "session_key": 9693,
        "driver_number": 14,
        "date": "2025-03-16T06:11:00Z",
        "gap_to_leader": 12.5,
        "interval": 1.2,
    }
    assert store.ingest("intervals", second) is True
    driver = store.state.driver(14)
    assert driver.gap_to_leader_s == pytest.approx(12.5)
    assert driver.laps_behind is None
    assert driver.interval_s == pytest.approx(1.2)
    assert driver.interval_laps_behind is None


def test_multiple_lap_deficit_is_not_parsed_as_seconds():
    store = RaceStateStore(session_key=9693)
    assert store.ingest("intervals", {
        "driver_number": 30,
        "date": "2025-03-16T06:10:00Z",
        "gap_to_leader": "+12 LAPS",
        "interval": None,
    })
    driver = store.state.driver(30)
    assert driver.gap_to_leader_s is None
    assert driver.laps_behind == 12


def test_string_false_is_not_truthy_for_provider_boolean_fields():
    store = RaceStateStore(session_key=9693)
    assert store.ingest("weather", {
        "date": "2025-03-16T06:10:00Z",
        "rainfall": "false",
        "air_temperature": 21.0,
    })
    assert store.state.weather.rainfall is False

    assert store.ingest("car_data", {
        "driver_number": 1,
        "date": "2025-03-16T06:10:01Z",
        "brake": "false",
        "speed": 280,
    })
    assert store.state.driver(1).brake is False


def test_truth_gate_accepts_lap_down_classification_without_fake_pace_or_tyre_inputs():
    snapshot = _snapshot()
    assert validate_simulation_observations(snapshot) == {}
    audit = assert_trusted_live_state(snapshot, max_age_s=20.0)
    assert audit.observed_drivers == 3
    assert audit.simulation_eligible_drivers == 2
    assert audit.classification_only_drivers == 1


def test_conflicting_seconds_and_lap_deficit_fails_closed():
    snapshot = _snapshot()
    snapshot["drivers"][2]["gap_to_leader_s"] = 90.0
    missing = validate_simulation_observations(snapshot)
    assert "conflicting_gap_representations" in missing["14"]


def test_scope_excludes_lap_down_driver_without_renumbering_or_mutating_source():
    snapshot = _snapshot()
    scoped, classification_only = split_simulation_scope(snapshot)
    assert [row["driver_number"] for row in scoped["drivers"]] == [1, 4]
    assert [row["position"] for row in scoped["drivers"]] == [1, 2]
    assert classification_only == [{
        "driver_number": 14,
        "acronym": "D14",
        "full_name": None,
        "position": 3,
        "laps_behind": 1,
        "gap_to_leader_raw": "+1 LAP",
        "reason": "provider supplied lap deficit without exact seconds gap",
        "prediction_status": "classification_only",
    }]
    assert snapshot["drivers"][2]["laps_behind"] == 1
    assert scoped["simulation_scope"]["invented_lap_deficit_seconds"] is False


def test_strategy_refuses_lap_down_driver_until_seconds_gap_exists():
    _scoped, classification_only = split_simulation_scope(_snapshot())
    ensure_strategy_eligible(1, classification_only)
    with pytest.raises(ValueError, match="classification-only"):
        ensure_strategy_eligible(14, classification_only)


def test_scope_refuses_interleaved_lap_deficit_instead_of_silently_renumbering():
    snapshot = _snapshot()
    snapshot["drivers"][1]["laps_behind"] = 1
    snapshot["drivers"][1]["gap_to_leader_s"] = None
    snapshot["drivers"][1]["gap_to_leader_raw"] = "+1 LAP"
    snapshot["drivers"][2] = _driver(14, 3, 8.0)
    with pytest.raises(ValueError, match="interleaved"):
        split_simulation_scope(snapshot)
