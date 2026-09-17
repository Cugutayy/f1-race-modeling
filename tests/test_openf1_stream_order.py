from datetime import UTC, datetime, timedelta

from f1_research.live_state import RaceStateStore


def test_same_document_older_openf1_revision_cannot_overwrite_newer_state():
    store = RaceStateStore(123)
    received = datetime(2026, 9, 17, 10, tzinfo=UTC)
    event_time = "2026-09-17T09:59:58+00:00"

    assert store.ingest("v1/laps", {
        "session_key": 123,
        "driver_number": 1,
        "lap_number": 10,
        "date_start": event_time,
        "lap_duration": 90.25,
        "_key": "lap-10-driver-1",
        "_id": 101,
    }, received)
    assert store.ingest("v1/laps", {
        "session_key": 123,
        "driver_number": 1,
        "lap_number": 10,
        "date_start": event_time,
        "lap_duration": 89.95,
        "duration_sector_3": 28.1,
        "_key": "lap-10-driver-1",
        "_id": 103,
    }, received)
    assert not store.ingest("v1/laps", {
        "session_key": 123,
        "driver_number": 1,
        "lap_number": 10,
        "date_start": event_time,
        "lap_duration": 91.50,
        "_key": "lap-10-driver-1",
        "_id": 102,
    }, received)

    state = store.snapshot(received)
    driver = state["drivers"][0]
    assert driver["last_lap_s"] == 89.95
    assert driver["sector_3_s"] == 28.1
    assert driver["recent_laps_s"] == [89.95]
    assert state["received_messages"] == 2
    assert state["rejected_provider_order_messages"] == 1


def test_different_openf1_keys_are_not_dropped_only_because_ids_arrive_out_of_order():
    store = RaceStateStore(123)
    received = datetime(2026, 9, 17, 10, tzinfo=UTC)
    assert store.ingest("position", {
        "driver_number": 1,
        "position": 1,
        "date": "2026-09-17T10:00:00+00:00",
        "_key": "position-driver-1",
        "_id": 500,
    }, received)
    assert store.ingest("position", {
        "driver_number": 2,
        "position": 2,
        "date": "2026-09-17T10:00:00+00:00",
        "_key": "position-driver-2",
        "_id": 499,
    }, received)
    state = store.snapshot(received)
    assert [row["position"] for row in state["drivers"]] == [1, 2]
    assert state["rejected_provider_order_messages"] == 0


def test_explicit_malformed_provider_timestamp_is_quarantined_not_replaced_by_receive_time():
    store = RaceStateStore(123)
    received = datetime(2026, 9, 17, 10, tzinfo=UTC)
    assert not store.ingest("position", {
        "driver_number": 1,
        "position": 1,
        "date": "not-a-date",
    }, received)
    state = store.snapshot(received)
    assert state["received_messages"] == 0
    assert state["rejected_invalid_timestamp_messages"] == 1
    assert state["latest_provider_event_at"] is None
    assert state["drivers"] == []


def test_receive_time_fallback_is_allowed_only_when_provider_timestamp_is_absent():
    store = RaceStateStore(123)
    received = datetime(2026, 9, 17, 10, tzinfo=UTC)
    assert store.ingest("drivers", {
        "driver_number": 1,
        "name_acronym": "AAA",
    }, received)
    state = store.snapshot(received + timedelta(seconds=1))
    assert state["received_messages"] == 1
    assert state["latest_provider_event_at"] is None
    assert state["drivers"][0]["last_seen_at"] == received.isoformat()


def test_latest_provider_event_time_tracks_event_time_not_download_time():
    store = RaceStateStore(123)
    received = datetime(2026, 9, 17, 10, tzinfo=UTC)
    event_time = datetime(2026, 9, 17, 9, 55, tzinfo=UTC)
    assert store.ingest("position", {
        "driver_number": 1,
        "position": 1,
        "date": event_time.isoformat(),
    }, received)
    state = store.snapshot(received)
    assert state["updated_at"] == received.isoformat()
    assert state["latest_provider_event_at"] == event_time.isoformat()
