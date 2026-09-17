from f1_research.monitoring import snapshot


def test_monitoring_snapshot_preserves_rejection_counters():
    result = snapshot({
        "session_key": 7, "current_lap": 12, "drivers": [{}, {}],
        "received_messages": 100, "rejected_stale_messages": 2,
        "rejected_provider_order_messages": 3,
        "rejected_invalid_timestamp_messages": 4,
    }, state_age_s=1.5, provider_age_s=2.0)
    assert result["driver_count"] == 2
    assert result["received_messages"] == 100
    assert result["rejected_provider_order_messages"] == 3
