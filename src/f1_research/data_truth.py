"""Fail-closed data-truth checks for live F1 predictions.

This module deliberately separates observed provider state from model assumptions.
A prediction is allowed to proceed only when the minimum live observations required
by the race simulator are present and fresh. Research/offline pipelines may choose a
looser policy explicitly, but the production API uses the strict policy.

OpenF1 may represent a lapped car's gap as ``+N LAP(S)`` instead of seconds. Such a
lap deficit is trusted classification evidence, but it is not silently converted into
a time gap. Lap-down cars can therefore remain in the live classification while the
exact-time Monte Carlo operates on the lead-lap subset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
MAX_TYRE_AGE_LAPS = 100
MIN_PLAUSIBLE_LAP_S = 20.0
MAX_PLAUSIBLE_LAP_S = 600.0
DEFAULT_FUTURE_TOLERANCE_S = 5.0


@dataclass(frozen=True)
class LiveTruthAudit:
    status: str
    state_age_s: float
    provider_event_age_s: float
    max_age_s: float
    future_tolerance_s: float
    observed_drivers: int
    required_drivers: int
    missing_fields: dict[str, list[str]]
    simulation_eligible_drivers: int = 0
    classification_only_drivers: int = 0


def parse_provider_timestamp(value: Any) -> datetime | None:
    """Parse a provider timestamp without substituting receive time on failure."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def age_seconds(value: Any, now: datetime | None = None) -> float | None:
    """Descriptive non-negative age; trust decisions use the stricter helper below."""
    parsed = parse_provider_timestamp(value)
    if parsed is None:
        return None
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return max(0.0, (now - parsed.astimezone(UTC)).total_seconds())


def _strict_timestamp_age(
    value: Any,
    *,
    label: str,
    now: datetime,
    future_tolerance_s: float,
) -> float:
    parsed = parse_provider_timestamp(value)
    if parsed is None:
        raise ValueError(f"Live state has no trustworthy {label} timestamp")
    delta = (now - parsed.astimezone(UTC)).total_seconds()
    if delta < -future_tolerance_s:
        raise ValueError(
            f"Live state {label} timestamp is {-delta:.1f}s in the future; "
            f"allowed clock skew is {future_tolerance_s:.1f}s"
        )
    return max(0.0, delta)


def _finite_positive_laps(row: dict[str, Any]) -> list[float]:
    values = (
        row.get("pace_laps_s")
        if "pace_laps_s" in row
        else row.get("recent_laps_s")
    ) or []
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and MIN_PLAUSIBLE_LAP_S <= number <= MAX_PLAUSIBLE_LAP_S:
            output.append(number)
    if output:
        return output
    try:
        last = float(row.get("last_lap_s"))
    except (TypeError, ValueError):
        return []
    return (
        [last]
        if np.isfinite(last) and MIN_PLAUSIBLE_LAP_S <= last <= MAX_PLAUSIBLE_LAP_S
        else []
    )


def _exact_positive_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        return None
    return int(number)


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0:
        return None
    return number


def validate_simulation_observations(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    """Return missing/invalid observed fields keyed by driver number.

    Exact-time simulation requires seconds-based gaps, usable pace observations and tyre
    state. A valid positive ``laps_behind`` is different evidence: it keeps the driver in
    the trusted classification but marks that car classification-only rather than
    manufacturing an exact time gap.
    """
    rows = snapshot.get("drivers")
    if not isinstance(rows, list):
        return {"state": ["drivers"]}

    missing: dict[str, list[str]] = {}
    positioned: list[tuple[int, int | None, dict[str, Any], float | None, int | None]] = []
    driver_numbers: list[int] = []
    positions: list[int] = []
    simulation_eligible = 0

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            missing.setdefault("state", []).append(f"driver_row_{index}_object")
            continue
        position = _exact_positive_int(row.get("position"))
        if position is None:
            if _exact_positive_int(row.get("driver_number")) is not None:
                key = str(int(float(row["driver_number"])))
                missing.setdefault(key, []).append("position")
            continue

        number = _exact_positive_int(row.get("driver_number"))
        key = str(number) if number is not None else f"position_{position}"
        fields: list[str] = []
        if number is None:
            fields.append("driver_number")
        else:
            driver_numbers.append(number)
        positions.append(position)

        gap_value = _finite_nonnegative(row.get("gap_to_leader_s"))
        laps_behind = _exact_positive_int(row.get("laps_behind"))

        if position == 1:
            if laps_behind is not None:
                fields.append("leader_laps_behind")
            if row.get("gap_to_leader_s") is None:
                gap_value = 0.0
            elif gap_value is None:
                fields.append("gap_to_leader_s")
        elif laps_behind is not None:
            if gap_value is not None:
                fields.append("conflicting_gap_representations")
            # Classification is observed, but exact-time simulation deliberately excludes
            # this row until a seconds gap becomes available again.
            gap_value = None
        elif gap_value is None:
            # No lap-deficit evidence exists for this row, so a missing seconds gap is
            # an incomplete exact-time observation rather than a license to fabricate it.
            fields.append("gap_to_leader_s")

        requires_exact_time_inputs = laps_behind is None
        exact_time_eligible = requires_exact_time_inputs and gap_value is not None
        if exact_time_eligible:
            simulation_eligible += 1

        # A non-lapped row belongs to the exact-time model domain even when one required
        # field (for example the seconds gap) is missing. Report all missing inputs so the
        # trusted-live API fails closed with a complete diagnostic. Lap-down rows are the
        # only rows allowed to omit pace/tyre inputs because they are classification-only.
        if requires_exact_time_inputs:
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
            if (
                not np.isfinite(tyre_age_f)
                or tyre_age_f < 0
                or tyre_age_f > MAX_TYRE_AGE_LAPS
                or not tyre_age_f.is_integer()
            ):
                fields.append("tyre_age")

        if fields:
            missing[key] = sorted(set(fields))
        positioned.append((position, number, row, gap_value, laps_behind))

    if len(positioned) < 2:
        missing.setdefault("state", []).append("at_least_two_positioned_drivers")
    if simulation_eligible < 2:
        missing.setdefault("state", []).append("at_least_two_exact_time_drivers")

    if len(driver_numbers) != len(set(driver_numbers)):
        missing.setdefault("state", []).append("duplicate_driver_numbers")
    if len(positions) != len(set(positions)):
        missing.setdefault("state", []).append("duplicate_positions")
    if positions and sorted(positions) != list(range(1, len(positions) + 1)):
        missing.setdefault("state", []).append("non_consecutive_positions")

    # Seconds gaps are monotonic only inside the exact-time/lead-lap representation.
    # Lap-down rows are a different unit and are intentionally excluded here.
    ordered_gaps = [
        (position, gap)
        for position, _number, _row, gap, laps_behind in sorted(positioned, key=lambda item: item[0])
        if laps_behind is None and gap is not None
    ]
    for (prev_position, prev_gap), (position, gap) in zip(ordered_gaps, ordered_gaps[1:]):
        if position > prev_position and gap + 1e-9 < prev_gap:
            missing.setdefault("state", []).append("non_monotonic_gaps")
            break

    if "state" in missing:
        missing["state"] = sorted(set(missing["state"]))
    return missing


def assert_trusted_live_state(
    snapshot: dict[str, Any],
    *,
    max_age_s: float = 20.0,
    future_tolerance_s: float = DEFAULT_FUTURE_TOLERANCE_S,
    now: datetime | None = None,
) -> LiveTruthAudit:
    """Validate explicit provider freshness and minimum observations for live prediction."""
    if not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be positive and finite")
    if not np.isfinite(future_tolerance_s) or future_tolerance_s < 0:
        raise ValueError("future_tolerance_s must be finite and non-negative")
    now = (now or datetime.now(UTC)).astimezone(UTC)

    state_age = _strict_timestamp_age(
        snapshot.get("updated_at"),
        label="receive/update",
        now=now,
        future_tolerance_s=future_tolerance_s,
    )
    provider_age = _strict_timestamp_age(
        snapshot.get("latest_provider_event_at"),
        label="provider event",
        now=now,
        future_tolerance_s=future_tolerance_s,
    )

    if state_age > max_age_s:
        raise ValueError(
            f"Live state is stale: receive/update age={state_age:.1f}s exceeds max_age={max_age_s:.1f}s"
        )
    if provider_age > max_age_s:
        raise ValueError(
            f"Live state is stale: provider event age={provider_age:.1f}s exceeds max_age={max_age_s:.1f}s"
        )

    missing = validate_simulation_observations(snapshot)
    if missing:
        details = "; ".join(f"{driver}: {','.join(fields)}" for driver, fields in sorted(missing.items()))
        raise ValueError(f"Live state is incomplete for trusted simulation: {details}")

    rows = snapshot.get("drivers", [])
    positioned = [
        row
        for row in rows
        if isinstance(row, dict) and _exact_positive_int(row.get("position")) is not None
    ]
    classification_only = sum(
        _exact_positive_int(row.get("laps_behind")) is not None for row in positioned
    )
    eligible = len(positioned) - classification_only
    return LiveTruthAudit(
        status="trusted_live",
        state_age_s=state_age,
        provider_event_age_s=provider_age,
        max_age_s=float(max_age_s),
        future_tolerance_s=float(future_tolerance_s),
        observed_drivers=len(positioned),
        required_drivers=eligible,
        missing_fields={},
        simulation_eligible_drivers=eligible,
        classification_only_drivers=classification_only,
    )


def audit_payload(audit: LiveTruthAudit) -> dict[str, Any]:
    return asdict(audit)
