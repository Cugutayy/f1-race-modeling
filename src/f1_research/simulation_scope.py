"""Define the production simulation scope without inventing lapped-car seconds.

OpenF1 may report ``+N LAP(S)`` instead of a numeric gap. Those rows remain valid live
classification observations, but the current Monte Carlo engine is time-gap based.
This adapter therefore separates exact-time/lead-lap rows from classification-only
lap-down rows. It never converts lap deficits to seconds.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _positive_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 1 or not number.is_integer():
        return None
    return int(number)


def split_simulation_scope(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = snapshot.get("drivers")
    if not isinstance(rows, list):
        raise ValueError("state.drivers must be a list")

    simulation_rows: list[dict[str, Any]] = []
    classification_only: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        laps_behind = _positive_int(row.get("laps_behind"))
        if laps_behind is None:
            simulation_rows.append(deepcopy(row))
            continue
        classification_only.append({
            "driver_number": row.get("driver_number"),
            "acronym": row.get("acronym"),
            "full_name": row.get("full_name"),
            "position": row.get("position"),
            "laps_behind": laps_behind,
            "gap_to_leader_raw": row.get("gap_to_leader_raw"),
            "reason": "provider supplied lap deficit without exact seconds gap",
            "prediction_status": "classification_only",
        })

    if len(simulation_rows) < 2:
        raise ValueError("At least two exact-time drivers are required for simulation")

    # Lap-down classifications occur behind lead-lap cars, so the retained positions
    # should remain 1..N. Refuse a malformed/interleaved snapshot rather than renumber it.
    positions = [row.get("position") for row in simulation_rows]
    if any(not isinstance(value, int) or value < 1 for value in positions):
        raise ValueError("Exact-time simulation rows have invalid positions")
    if sorted(positions) != list(range(1, len(positions) + 1)):
        raise ValueError("Lap-deficit rows are interleaved with exact-time classification")

    scoped = deepcopy(snapshot)
    scoped["drivers"] = simulation_rows
    scoped["simulation_scope"] = {
        "kind": "exact_time_lead_lap_subset",
        "included_driver_numbers": [row.get("driver_number") for row in simulation_rows],
        "classification_only_driver_numbers": [
            row.get("driver_number") for row in classification_only
        ],
        "invented_lap_deficit_seconds": False,
    }
    return scoped, classification_only


def annotate_simulation_report(
    report: dict[str, Any],
    classification_only: list[dict[str, Any]],
) -> dict[str, Any]:
    report["classification_only"] = classification_only
    audit = report.setdefault("audit", {})
    audit["probability_scope"] = (
        "exact_time_lead_lap_conditional" if classification_only else "full_observed_exact_time_field"
    )
    audit["classification_only_driver_numbers"] = [
        row.get("driver_number") for row in classification_only
    ]
    audit["lap_deficit_policy"] = "preserve_discrete_laps; never convert provider lap deficit to seconds"
    audit["invented_lap_deficit_seconds"] = False
    if classification_only:
        audit["probability_scope_warning"] = (
            "Probabilities are conditional on the exact-time lead-lap subset. Lap-down cars "
            "remain in live classification but receive no fabricated win/podium probabilities."
        )
    return report


def ensure_strategy_eligible(
    driver_number: int,
    classification_only: list[dict[str, Any]],
) -> None:
    excluded = {
        int(row["driver_number"])
        for row in classification_only
        if row.get("driver_number") is not None
    }
    if int(driver_number) in excluded:
        raise ValueError(
            f"Driver {driver_number} is lap-down/classification-only; exact-time strategy "
            "simulation requires a provider seconds gap"
        )
