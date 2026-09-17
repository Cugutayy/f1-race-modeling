"""Fail-closed data-truth checks for live F1 predictions.

This module deliberately separates observed provider state from model assumptions.
A prediction is allowed to proceed only when the minimum live observations required
by the race simulator are present and fresh. Research/offline pipelines may choose a
looser policy explicitly, but the production API uses the strict policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


@dataclass(frozen=True)
class LiveTruthAudit:
    status: str
    state_age_s: float | None
    provider_event_age_s: float | None
    max_age_s: float
    observed_drivers: int
    required_drivers: int
    missing_fields: dict[str, list[str]]


def parse_provider_timestamp(value: Any) -> datetime | None:
    """Parse a provider timestamp without substituting receive time on failure."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def age_seconds(value: Any, now: datetime | None = None) -> float | None:
    parsed = parse_provider_timestamp(value)
    if parsed is None:
        return None
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return max(0.0, (now - parsed.astimezone(UTC)).total_seconds())


def _finite_positive_laps(row: dict[str, Any]) -> list[float]:
    values = row.get("recent_laps_s") or []
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            output.append(number)
    if output:
        return output
    try:
        last = float(row.get("last_lap_s"))
    except (TypeError, ValueError):
        return []
    return [last] if np.isfinite(last) and last > 0 else []


def validate_simulation_observations(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    """Return missing/invalid observed fields keyed by driver number.

    The simulator needs real race gaps, usable pace observations and tyre state. It is
    safer to refuse a trusted-live probability than to silently manufacture a gap or
    assume MEDIUM/age zero for an unknown tyre.
    """
    rows = snapshot.get("drivers")
    if not isinstance(rows, list):
        return {"state": ["drivers"]}

    missing: dict[str, list[str]] = {}
    positioned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        position = row.get("position")
        try:
            position_int = int(position)
        except (TypeError, ValueError):
            continue
        if position_int < 1:
            continue
        positioned.append((position_int, row))

    for position, row in positioned:
        key = str(row.get("driver_number") or f"position_{position}")
        fields: list[str] = []
        try:
            number = int(row.get("driver_number"))
            if number <= 0:
                raise ValueError
        except (TypeError, ValueError):
            fields.append("driver_number")

        if position > 1:
            gap = row.get("gap_to_leader_s")
            try:
                gap_f = float(gap)
            except (TypeError, ValueError):
                gap_f = float("nan")
            if not np.isfinite(gap_f) or gap_f < 0:
                fields.append("gap_to_leader_s")

        if not _finite_positive_laps(row):
            fields.append("pace_observation")

        compound = str(row.get("compound") or "").upper().strip()
        if compound not in SUPPORTED_COMPOUNDS:
            fields.append("compound")

        tyre_age = row.get("tyre_age")
        try:
            tyre_age_f = float(tyre_age)
        except (TypeError, ValueError):
            tyre_age_f = float("nan")
        if not np.isfinite(tyre_age_f) or tyre_age_f < 0 or tyre_age_f % 1:
            fields.append("tyre_age")

        if fields:
            missing[key] = sorted(set(fields))

    if len(positioned) < 2:
        missing.setdefault("state", []).append("at_least_two_positioned_drivers")
    return missing


def assert_trusted_live_state(
    snapshot: dict[str, Any],
    *,
    max_age_s: float = 20.0,
    now: datetime | None = None,
) -> LiveTruthAudit:
    """Validate freshness and minimum observations for a trusted-live prediction."""
    if not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be positive and finite")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    state_age = age_seconds(snapshot.get("updated_at"), now)
    provider_age = age_seconds(snapshot.get("latest_provider_event_at"), now)

    effective_age = provider_age if provider_age is not None else state_age
    if effective_age is None:
        raise ValueError("Live state has no trustworthy timestamp")
    if effective_age > max_age_s:
        raise ValueError(
            f"Live state is stale: age={effective_age:.1f}s exceeds max_age={max_age_s:.1f}s"
        )

    missing = validate_simulation_observations(snapshot)
    if missing:
        details = "; ".join(f"{driver}: {','.join(fields)}" for driver, fields in sorted(missing.items()))
        raise ValueError(f"Live state is incomplete for trusted simulation: {details}")

    rows = snapshot.get("drivers", [])
    positioned = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("position"), int)
        and row.get("position", 0) > 0
    )
    return LiveTruthAudit(
        status="trusted_live",
        state_age_s=state_age,
        provider_event_age_s=provider_age,
        max_age_s=float(max_age_s),
        observed_drivers=positioned,
        required_drivers=positioned,
        missing_fields={},
    )


def audit_payload(audit: LiveTruthAudit) -> dict[str, Any]:
    return asdict(audit)
