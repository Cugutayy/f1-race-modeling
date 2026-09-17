"""Small dependency-free operational metrics snapshot for production health."""
from __future__ import annotations

from typing import Any


def snapshot(state: dict[str, Any], *, state_age_s: float | None,
             provider_age_s: float | None, prediction_latency_ms: float | None = None) -> dict[str, Any]:
    drivers = state.get("drivers")
    driver_count = len(drivers) if isinstance(drivers, list) else 0
    return {
        "schema_version": 1,
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "driver_count": driver_count,
        "state_age_s": state_age_s,
        "provider_event_age_s": provider_age_s,
        "prediction_latency_ms": prediction_latency_ms,
        "received_messages": int(state.get("received_messages") or 0),
        "rejected_stale_messages": int(state.get("rejected_stale_messages") or 0),
        "rejected_provider_order_messages": int(state.get("rejected_provider_order_messages") or 0),
        "rejected_invalid_timestamp_messages": int(state.get("rejected_invalid_timestamp_messages") or 0),
    }
