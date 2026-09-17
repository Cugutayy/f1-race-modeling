"""Operational quality classification for live provider state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityState:
    status: str
    state_age_s: float | None
    provider_age_s: float | None
    connection_state: str | None
    reasons: tuple[str, ...]


def classify(*, state_age_s: float | None, provider_age_s: float | None,
             connection_state: str | None, max_age_s: float) -> QualityState:
    reasons = []
    if connection_state not in {None, "connected"}:
        reasons.append(f"transport:{connection_state}")
    if state_age_s is None:
        reasons.append("state_timestamp_missing")
    elif state_age_s > max_age_s:
        reasons.append("state_stale")
    if provider_age_s is None:
        reasons.append("provider_timestamp_missing")
    elif provider_age_s > max_age_s:
        reasons.append("provider_stale")
    if connection_state == "disconnected":
        status = "DISCONNECTED"
    elif any(reason.endswith("_stale") for reason in reasons):
        status = "STALE"
    elif reasons:
        status = "DEGRADED"
    else:
        status = "LIVE"
    return QualityState(status, state_age_s, provider_age_s, connection_state, tuple(reasons))
