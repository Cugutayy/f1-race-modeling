from f1_research.live_quality import classify


def test_live_quality_states():
    assert classify(state_age_s=1, provider_age_s=1, connection_state="connected", max_age_s=20).status == "LIVE"
    assert classify(state_age_s=30, provider_age_s=1, connection_state="connected", max_age_s=20).status == "STALE"
    assert classify(state_age_s=1, provider_age_s=1, connection_state="disconnected", max_age_s=20).status == "DISCONNECTED"
    assert classify(state_age_s=None, provider_age_s=1, connection_state="connected", max_age_s=20).status == "DEGRADED"
