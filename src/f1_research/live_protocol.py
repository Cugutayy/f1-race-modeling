"""Versioned live event envelopes for WebSocket/SSE transports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class LiveEnvelope:
    schema_version: int
    sequence: int
    event_type: str
    provider_time: str | None
    generated_at: str
    state_sha256: str
    payload: dict[str, Any]


def state_sha256(state: dict[str, Any]) -> str:
    raw = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def envelope(*, sequence: int, event_type: str, state: dict[str, Any],
             payload: dict[str, Any], provider_time: str | None = None,
             generated_at: str | None = None) -> LiveEnvelope:
    if sequence < 1:
        raise ValueError("sequence must be positive")
    if not event_type:
        raise ValueError("event_type is required")
    return LiveEnvelope(
        schema_version=1, sequence=sequence, event_type=event_type,
        provider_time=provider_time,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        state_sha256=state_sha256(state), payload=payload,
    )


def encode(item: LiveEnvelope) -> str:
    return json.dumps(asdict(item), separators=(",", ":"), allow_nan=False)
