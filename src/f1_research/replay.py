"""Deterministic replay of captured provider observations through the live state engine."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .live_state import RaceStateStore, parse_provider_timestamp


@dataclass(frozen=True)
class ReplayEvent:
    topic: str
    payload: dict[str, Any]
    received_at: str


def _event_time(event: ReplayEvent) -> datetime:
    raw = event.payload.get("date") or event.payload.get("date_start") or event.received_at
    parsed = parse_provider_timestamp(raw)
    if parsed is None:
        raise ValueError(f"replay event has invalid timestamp: {raw!r}")
    return parsed


def load_jsonl(path: Path) -> list[ReplayEvent]:
    events = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("topic"), str) or not isinstance(row.get("payload"), dict):
            raise ValueError(f"invalid replay row {number}")
        received = row.get("received_at")
        if parse_provider_timestamp(received) is None:
            raise ValueError(f"invalid replay received_at on row {number}")
        events.append(ReplayEvent(row["topic"], row["payload"], received))
    if not events:
        raise ValueError("replay capture is empty")
    return events


def replay(events: Iterable[ReplayEvent], *, session_key: int | None = None,
           snapshot_every: int = 1) -> dict[str, Any]:
    if snapshot_every < 1:
        raise ValueError("snapshot_every must be >= 1")
    ordered = sorted(list(events), key=lambda event: (_event_time(event), event.topic))
    if not ordered:
        raise ValueError("replay has no events")
    store = RaceStateStore(session_key=session_key)
    snapshots = []
    accepted = 0
    for index, event in enumerate(ordered, 1):
        received = parse_provider_timestamp(event.received_at)
        if received is None:
            raise ValueError("invalid received_at")
        if store.ingest(event.topic, event.payload, received_at=received):
            accepted += 1
        if index % snapshot_every == 0 or index == len(ordered):
            snapshots.append({"sequence": index, "state": store.state.to_dict()})
    canonical = json.dumps(
        [{"topic": e.topic, "payload": e.payload, "received_at": e.received_at} for e in ordered],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "event_count": len(ordered),
        "accepted_count": accepted,
        "rejected_count": len(ordered) - accepted,
        "snapshots": snapshots,
        "final_state": store.state.to_dict(),
    }


def write_replay(path: Path, result: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
