from f1_research.replay import ReplayEvent, replay


def test_replay_orders_events_and_rejects_stale_provider_revision():
    events = [
        ReplayEvent("position", {
            "_id": 2, "_key": "p1", "date": "2026-03-08T05:00:02Z",
            "session_key": 1, "driver_number": 1, "position": 1,
        }, "2026-03-08T05:00:02Z"),
        ReplayEvent("position", {
            "_id": 1, "_key": "p1", "date": "2026-03-08T05:00:03Z",
            "session_key": 1, "driver_number": 1, "position": 2,
        }, "2026-03-08T05:00:03Z"),
    ]
    result = replay(events, session_key=1)
    assert result["event_count"] == 2
    assert result["accepted_count"] == 1
    assert result["final_state"]["drivers"][0]["position"] == 1
    assert result["final_state"]["rejected_provider_order_messages"] == 1


def test_replay_is_content_deterministic():
    event = ReplayEvent("position", {
        "date": "2026-03-08T05:00:00Z", "session_key": 1,
        "driver_number": 1, "position": 1,
    }, "2026-03-08T05:00:00Z")
    assert replay([event])["source_sha256"] == replay([event])["source_sha256"]
