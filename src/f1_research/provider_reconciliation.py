"""Cross-provider numerical reconciliation for completed F1 races.

This module verifies completed-race facts; it does not predict or silently repair
provider disagreements. Jolpica, OpenF1 and FastF1 are normalized into one explicit
schema. Hard facts are compared only when the provider contract actually supplies the
same semantic field. Missing evidence remains explicit and is never converted to zero,
False, or a majority-vote "truth".

Driver identity uses the race number exposed by each source. Names/codes are audit
context only and are never used to force a match when numbers disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import JsonCache
from .openf1_live import OpenF1Client
from .provider_event_identity import build_event_identity, require_event_identity

HARD_FIELDS = ("position", "laps", "result_class", "start_status")
SECONDARY_FIELDS = ("grid_position", "pit_stops", "points")
POSITION_REQUIRED_STATUSES = {"finished", "classified_lapped"}
POSITION_OPTIONAL_STATUSES = {"dnf", "dns", "dsq"}
FIELD_SEMANTICS = {
    "position": (
        "Provider-reported final/classified position. Some public providers leave position "
        "null for DNF/DNS/DSQ rows even when other sources publish a classification order. "
        "That absence is recorded as insufficient evidence, not invented or majority-repaired."
    ),
    "laps": (
        "Jolpica Results.laps; OpenF1 session_result.number_of_laps; FastF1 Results.Laps "
        "when present, otherwise maximum observed FastF1 LapNumber."
    ),
    "status_class": (
        "Provider-specific normalized label: finished, classified_lapped, dnf, dns or dsq. "
        "Retained for audit context; not compared as a cross-provider hard fact because providers "
        "can encode start participation and final classification semantics differently."
    ),
    "result_class": (
        "Cross-provider comparable race-result class derived conservatively from each provider label: "
        "completed, classified_lapped, non_finisher or disqualified. DNS and DNF both map to "
        "non_finisher here; their start-participation distinction is carried separately."
    ),
    "start_status": (
        "Start-participation evidence: started, dns or UNKNOWN. A zero-lap generic Retired/DNF label "
        "from Jolpica/FastF1 is UNKNOWN rather than guessed started or DNS; OpenF1 explicit DNS/DNF "
        "flags can distinguish the states."
    ),
    "grid_position": (
        "Provider starting-grid position: Jolpica Results.grid, OpenF1 starting_grid.position, "
        "and FastF1 Results.GridPosition. Missing rows remain UNKNOWN and are never zero-filled."
    ),
    "pit_stops": (
        "Count of provider pit-stop observations. Missing pit evidence stays UNKNOWN, "
        "never zero. Equal counts do not prove identical pit timing semantics."
    ),
    "points": "Provider-reported race points when available; missing values stay UNKNOWN.",
}


@dataclass(frozen=True)
class ResultRow:
    provider: str
    driver_number: int
    position: int | None
    laps: int | None
    status_class: str | None
    grid_position: int | None = None
    pit_stops: int | None = None
    points: float | None = None
    status_raw: str | None = None
    driver_code: str | None = None
    driver_name: str | None = None


@dataclass(frozen=True)
class Mismatch:
    provider_a: str
    provider_b: str
    driver_number: int | None
    field: str
    value_a: Any
    value_b: Any
    severity: str
    reason: str


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _positive_int(value: Any, *, allow_zero: bool = False) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    lower = 0 if allow_zero else 1
    return int(number) if number >= lower else None


def _finite_float(value: Any) -> float | None:
    return _finite_number(value)


def _clean_text(value: Any) -> str | None:
    """Normalize provider text without pandas NA truth-value coercion."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any, *, field: str) -> bool | None:
    """Accept only explicit boolean encodings; Python truthiness is forbidden."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
    raise ValueError(f"Unsupported explicit boolean for {field}: {value!r}")


def _status_from_text(status: Any, position_text: Any = None) -> str | None:
    raw = (_clean_text(status) or "").upper()
    position_raw = (_clean_text(position_text) or "").upper()
    if "DISQUAL" in raw or position_raw in {"D", "DSQ"}:
        return "dsq"
    if raw in {"DID NOT START", "DNS", "WITHDREW"} or "DID NOT START" in raw:
        return "dns"
    if raw == "FINISHED":
        return "finished"
    if "LAPPED" in raw or (raw.startswith("+") and "LAP" in raw):
        return "classified_lapped"
    if raw:
        return "dnf"
    return None


def _openf1_status(row: dict[str, Any]) -> str | None:
    dsq = _optional_bool(row.get("dsq"), field="openf1.dsq")
    dns = _optional_bool(row.get("dns"), field="openf1.dns")
    dnf = _optional_bool(row.get("dnf"), field="openf1.dnf")
    if dsq:
        return "dsq"
    if dns:
        return "dns"
    if dnf:
        return "dnf"
    gap = (_clean_text(row.get("gap_to_leader")) or "").upper()
    if "LAP" in gap:
        return "classified_lapped"
    if _positive_int(row.get("position")) is not None:
        return "finished"
    return None


def _result_class(row: ResultRow) -> str | None:
    """Map provider labels only to semantics that are comparable across sources."""
    if row.status_class == "finished":
        return "completed"
    if row.status_class == "classified_lapped":
        return "classified_lapped"
    if row.status_class in {"dnf", "dns"}:
        return "non_finisher"
    if row.status_class == "dsq":
        return "disqualified"
    return None


def _start_status(row: ResultRow) -> str | None:
    """Return start evidence without turning a zero-lap generic retirement into a guess."""
    if row.status_class == "dns":
        return "dns"
    if row.status_class in {"finished", "classified_lapped"}:
        return "started"
    if row.status_class == "dnf":
        if row.provider == "OpenF1":
            return "started"
        if row.laps is not None and row.laps > 0:
            return "started"
        return None
    if row.status_class == "dsq" and row.laps is not None and row.laps > 0:
        return "started"
    return None


def _pit_counts_jolpica(payload: dict[str, Any] | None) -> tuple[bool, dict[str, int]]:
    if payload is None:
        return False, {}
    try:
        races = payload["MRData"]["RaceTable"]["Races"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed Jolpica pit-stop payload") from exc
    if not isinstance(races, list) or len(races) > 1:
        raise ValueError("Jolpica pit-stop reconciliation requires zero or one race")
    counts: dict[str, int] = {}
    for stop in (races[0].get("PitStops", []) if races else []):
        if not isinstance(stop, dict):
            raise ValueError("Jolpica pit-stop row must be an object")
        driver_id = _clean_text(stop.get("driverId"))
        if driver_id:
            counts[driver_id] = counts.get(driver_id, 0) + 1
    return True, counts


def normalize_jolpica_results(
    payload: dict[str, Any],
    pit_payload: dict[str, Any] | None = None,
) -> tuple[list[ResultRow], dict[str, Any]]:
    """Normalize one Jolpica race response without guessing missing evidence."""
    try:
        races = payload["MRData"]["RaceTable"]["Races"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed Jolpica results payload") from exc
    if not isinstance(races, list) or len(races) != 1:
        raise ValueError("Jolpica reconciliation requires exactly one race")
    race = races[0]
    raw_results = race.get("Results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("Jolpica race has no Results rows")
    pit_evidence, pit_counts = _pit_counts_jolpica(pit_payload)

    rows: list[ResultRow] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ValueError("Jolpica result row must be an object")
        number = _positive_int(raw.get("number"))
        if number is None:
            raise ValueError("Jolpica result row has no valid race driver number")
        driver = raw.get("Driver") if isinstance(raw.get("Driver"), dict) else {}
        name_parts = [driver.get("givenName"), driver.get("familyName")]
        name = " ".join(str(value).strip() for value in name_parts if _clean_text(value)) or None
        driver_id = _clean_text(driver.get("driverId"))
        rows.append(ResultRow(
            provider="Jolpica",
            driver_number=number,
            position=_positive_int(raw.get("position")),
            laps=_positive_int(raw.get("laps"), allow_zero=True),
            status_class=_status_from_text(raw.get("status"), raw.get("positionText")),
            grid_position=_positive_int(raw.get("grid"), allow_zero=True),
            pit_stops=(pit_counts.get(driver_id, 0) if pit_evidence and driver_id else None),
            points=_finite_float(raw.get("points")),
            status_raw=_clean_text(raw.get("status")),
            driver_code=_clean_text(driver.get("code")),
            driver_name=name,
        ))

    metadata = {
        "provider": "Jolpica",
        "season": _positive_int(race.get("season")),
        "round": _positive_int(race.get("round")),
        "race_name": _clean_text(race.get("raceName")),
        "date": _clean_text(race.get("date")),
        "circuit_id": (
            _clean_text((race.get("Circuit") or {}).get("circuitId"))
            if isinstance(race.get("Circuit"), dict)
            else None
        ),
        "constructor_ids": sorted({
            _clean_text((raw.get("Constructor") or {}).get("constructorId"))
            for raw in raw_results
            if isinstance(raw, dict) and isinstance(raw.get("Constructor"), dict)
        } - {None}),
    }
    return _validated_rows(rows), metadata


def normalize_openf1_results(
    result_rows: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]] | None = None,
    pit_rows: list[dict[str, Any]] | None = None,
    starting_grid_rows: list[dict[str, Any]] | None = None,
) -> list[ResultRow]:
    if not isinstance(result_rows, list) or not result_rows:
        raise ValueError("OpenF1 session_result rows are empty")
    lookup: dict[int, dict[str, Any]] = {}
    for driver in driver_rows or []:
        if not isinstance(driver, dict):
            continue
        number = _positive_int(driver.get("driver_number"))
        if number is not None:
            lookup[number] = driver

    grid_evidence = starting_grid_rows is not None
    grid_lookup: dict[int, int] = {}
    for raw in starting_grid_rows or []:
        if not isinstance(raw, dict):
            raise ValueError("OpenF1 starting-grid row must be an object")
        number = _positive_int(raw.get("driver_number"))
        position = _positive_int(raw.get("position"), allow_zero=True)
        if number is None or position is None:
            raise ValueError("OpenF1 starting-grid row has invalid driver_number/position")
        if number in grid_lookup:
            raise ValueError(f"OpenF1 starting-grid has duplicate driver_number {number}")
        grid_lookup[number] = position

    pit_evidence = pit_rows is not None
    pit_keys: dict[int, set[tuple[Any, Any]]] = {}
    for raw in pit_rows or []:
        if not isinstance(raw, dict):
            raise ValueError("OpenF1 pit row must be an object")
        number = _positive_int(raw.get("driver_number"))
        if number is not None:
            pit_keys.setdefault(number, set()).add((raw.get("lap_number"), raw.get("date")))

    rows: list[ResultRow] = []
    for raw in result_rows:
        if not isinstance(raw, dict):
            raise ValueError("OpenF1 result row must be an object")
        number = _positive_int(raw.get("driver_number"))
        if number is None:
            raise ValueError("OpenF1 result row has no valid driver_number")
        driver = lookup.get(number, {})
        status_class = _openf1_status(raw)
        rows.append(ResultRow(
            provider="OpenF1",
            driver_number=number,
            position=_positive_int(raw.get("position")),
            laps=_positive_int(raw.get("number_of_laps"), allow_zero=True),
            status_class=status_class,
            grid_position=(grid_lookup.get(number) if grid_evidence else None),
            pit_stops=(len(pit_keys.get(number, set())) if pit_evidence else None),
            points=_finite_float(raw.get("points")),
            status_raw=status_class,
            driver_code=_clean_text(driver.get("name_acronym")),
            driver_name=_clean_text(driver.get("full_name")),
        ))
    return _validated_rows(rows)


def normalize_fastf1_results(
    results: pd.DataFrame | list[dict[str, Any]],
    laps: pd.DataFrame | list[dict[str, Any]] | None = None,
) -> list[ResultRow]:
    frame = pd.DataFrame(results).copy()
    if frame.empty:
        raise ValueError("FastF1 results are empty")
    if "DriverNumber" not in frame:
        raise ValueError("FastF1 results lack DriverNumber")

    lap_frame = pd.DataFrame(laps).copy() if laps is not None else pd.DataFrame()
    lap_evidence = not lap_frame.empty and {"DriverNumber", "LapNumber"} <= set(lap_frame)
    pit_evidence = lap_evidence and "PitInTime" in lap_frame
    lap_counts: dict[int, int] = {}
    pit_counts: dict[int, int] = {}
    if lap_evidence:
        lap_frame["DriverNumber"] = pd.to_numeric(lap_frame["DriverNumber"], errors="coerce")
        lap_frame["LapNumber"] = pd.to_numeric(lap_frame["LapNumber"], errors="coerce")
        for number, group in lap_frame.dropna(subset=["DriverNumber", "LapNumber"]).groupby("DriverNumber"):
            driver_number = int(number)
            lap_counts[driver_number] = int(group["LapNumber"].max())
            if pit_evidence:
                pit_counts[driver_number] = int(group["PitInTime"].notna().sum())

    rows: list[ResultRow] = []
    for raw in frame.to_dict("records"):
        number = _positive_int(raw.get("DriverNumber"))
        if number is None:
            raise ValueError("FastF1 result row has no valid DriverNumber")
        status = raw.get("Status")
        status_class = _status_from_text(status, raw.get("ClassifiedPosition"))
        result_laps = _positive_int(raw.get("Laps"), allow_zero=True)
        if result_laps is None:
            result_laps = lap_counts.get(number)
        if result_laps is None and status_class == "dns":
            result_laps = 0

        code = _clean_text(raw.get("Abbreviation")) or _clean_text(raw.get("Driver"))
        name = _clean_text(raw.get("FullName"))
        if name is None:
            first = _clean_text(raw.get("FirstName"))
            last = _clean_text(raw.get("LastName"))
            name = " ".join(value for value in (first, last) if value) or None

        rows.append(ResultRow(
            provider="FastF1",
            driver_number=number,
            position=_positive_int(raw.get("Position")),
            laps=result_laps,
            status_class=status_class,
            grid_position=_positive_int(raw.get("GridPosition"), allow_zero=True),
            pit_stops=(pit_counts.get(number, 0) if pit_evidence else None),
            points=_finite_float(raw.get("Points")),
            status_raw=_clean_text(status),
            driver_code=code,
            driver_name=name,
        ))
    return _validated_rows(rows)


def _validated_rows(rows: Iterable[ResultRow]) -> list[ResultRow]:
    rows = list(rows)
    if len(rows) < 2:
        raise ValueError("A reconciled race needs at least two result rows")
    numbers = [row.driver_number for row in rows]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate driver numbers in provider results")
    return sorted(rows, key=lambda row: (row.position is None, row.position or 10_000, row.driver_number))


def _position_required(row: ResultRow) -> bool:
    return row.status_class in POSITION_REQUIRED_STATUSES


def _provider_integrity(
    provider: str,
    rows: list[ResultRow],
) -> tuple[list[Mismatch], list[dict[str, Any]]]:
    """Return hard integrity failures plus explicit hard-evidence gaps."""
    failures: list[Mismatch] = []
    insufficient: list[dict[str, Any]] = []
    positions = [row.position for row in rows if row.position is not None]

    if len(positions) != len(set(positions)):
        failures.append(Mismatch(
            provider,
            provider,
            None,
            "classification_integrity",
            positions,
            None,
            "hard",
            "provider has duplicate final positions",
        ))
    if positions and sorted(positions) != list(range(1, max(positions) + 1)):
        failures.append(Mismatch(
            provider,
            provider,
            None,
            "classification_integrity",
            sorted(positions),
            None,
            "hard",
            "provider observed positions are not consecutive from 1 through max(position)",
        ))

    for row in rows:
        if row.position is None:
            if _position_required(row):
                failures.append(Mismatch(
                    provider,
                    provider,
                    row.driver_number,
                    "position",
                    None,
                    None,
                    "hard",
                    "provider is missing position for a classified finisher/lapped car",
                ))
            else:
                insufficient.append({
                    "provider": provider,
                    "driver_number": row.driver_number,
                    "field": "position",
                    "status_class": row.status_class,
                    "reason": "provider does not expose a final position for this non-finisher",
                })
        if row.laps is None:
            failures.append(Mismatch(
                provider,
                provider,
                row.driver_number,
                "laps",
                None,
                None,
                "hard",
                "provider is missing completed laps",
            ))
        if row.status_class is None:
            failures.append(Mismatch(
                provider,
                provider,
                row.driver_number,
                "status_class",
                None,
                None,
                "hard",
                "provider is missing normalized classification status",
            ))
    return failures, insufficient


def _row_index(rows: Iterable[ResultRow]) -> dict[int, ResultRow]:
    return {row.driver_number: row for row in rows}


def _values_equal(field: str, value_a: Any, value_b: Any) -> bool:
    if field == "points":
        try:
            return bool(np.isclose(float(value_a), float(value_b), atol=1e-9, rtol=0.0))
        except (TypeError, ValueError):
            return value_a == value_b
    return value_a == value_b


def reconcile_results(provider_rows: dict[str, list[ResultRow]]) -> dict[str, Any]:
    """Compare normalized provider results without repairing disagreements."""
    if len(provider_rows) < 2:
        raise ValueError("At least two providers are required for reconciliation")

    normalized = {name: _validated_rows(rows) for name, rows in provider_rows.items()}
    providers = sorted(normalized)
    mismatches: list[Mismatch] = []
    insufficient_hard: list[dict[str, Any]] = []
    insufficient_secondary: list[dict[str, Any]] = []

    for provider in providers:
        failures, missing = _provider_integrity(provider, normalized[provider])
        mismatches.extend(failures)
        insufficient_hard.extend(missing)

    for i, provider_a in enumerate(providers):
        for provider_b in providers[i + 1:]:
            left = _row_index(normalized[provider_a])
            right = _row_index(normalized[provider_b])
            left_numbers, right_numbers = set(left), set(right)

            for number in sorted(left_numbers - right_numbers):
                mismatches.append(Mismatch(
                    provider_a,
                    provider_b,
                    number,
                    "driver_presence",
                    True,
                    False,
                    "hard",
                    "driver exists only in first provider",
                ))
            for number in sorted(right_numbers - left_numbers):
                mismatches.append(Mismatch(
                    provider_a,
                    provider_b,
                    number,
                    "driver_presence",
                    False,
                    True,
                    "hard",
                    "driver exists only in second provider",
                ))

            for number in sorted(left_numbers & right_numbers):
                a, b = left[number], right[number]

                if a.position is None or b.position is None:
                    insufficient_hard.append({
                        "provider_a": provider_a,
                        "provider_b": provider_b,
                        "driver_number": number,
                        "field": "position",
                        "value_a": a.position,
                        "value_b": b.position,
                        "status_a": a.status_class,
                        "status_b": b.status_class,
                        "reason": "at least one provider does not expose comparable non-finisher position evidence",
                    })
                elif a.position != b.position:
                    mismatches.append(Mismatch(
                        provider_a,
                        provider_b,
                        number,
                        "position",
                        a.position,
                        b.position,
                        "hard",
                        "provider values disagree",
                    ))

                if a.laps != b.laps:
                    mismatches.append(Mismatch(
                        provider_a,
                        provider_b,
                        number,
                        "laps",
                        a.laps,
                        b.laps,
                        "hard",
                        "provider values disagree",
                    ))

                for field, semantic in (("result_class", _result_class), ("start_status", _start_status)):
                    value_a, value_b = semantic(a), semantic(b)
                    if value_a is None or value_b is None:
                        insufficient_hard.append({
                            "provider_a": provider_a,
                            "provider_b": provider_b,
                            "driver_number": number,
                            "field": field,
                            "value_a": value_a,
                            "value_b": value_b,
                            "status_class_a": a.status_class,
                            "status_class_b": b.status_class,
                            "reason": "at least one provider lacks comparable semantic evidence",
                        })
                    elif value_a != value_b:
                        mismatches.append(Mismatch(
                            provider_a,
                            provider_b,
                            number,
                            field,
                            value_a,
                            value_b,
                            "hard",
                            "provider semantic values disagree",
                        ))

                for field in SECONDARY_FIELDS:
                    value_a, value_b = getattr(a, field), getattr(b, field)
                    if value_a is None or value_b is None:
                        insufficient_secondary.append({
                            "provider_a": provider_a,
                            "provider_b": provider_b,
                            "driver_number": number,
                            "field": field,
                            "value_a": value_a,
                            "value_b": value_b,
                        })
                    elif not _values_equal(field, value_a, value_b):
                        mismatches.append(Mismatch(
                            provider_a,
                            provider_b,
                            number,
                            field,
                            value_a,
                            value_b,
                            "warning",
                            "secondary provider values disagree; no truth is elected",
                        ))

                if a.driver_code and b.driver_code and a.driver_code.upper() != b.driver_code.upper():
                    mismatches.append(Mismatch(
                        provider_a,
                        provider_b,
                        number,
                        "driver_code",
                        a.driver_code,
                        b.driver_code,
                        "warning",
                        "audit identity label differs; number match is retained",
                    ))

    hard = [row for row in mismatches if row.severity == "hard"]
    warnings = [row for row in mismatches if row.severity == "warning"]
    normalized_payload: dict[str, list[dict[str, Any]]] = {}
    for provider, rows in normalized.items():
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            payload = asdict(row)
            payload["result_class"] = _result_class(row)
            payload["start_status"] = _start_status(row)
            payload_rows.append(payload)
        normalized_payload[provider] = payload_rows
    verification_status = "FAIL" if hard else ("PASS_WITH_GAPS" if insufficient_hard else "PASS")
    return {
        "schema_version": 4,
        "kind": "cross_provider_completed_race_reconciliation",
        "providers": providers,
        "row_counts": {provider: len(rows) for provider, rows in normalized.items()},
        "passed": not hard,
        "verification_status": verification_status,
        "hard_mismatch_count": len(hard),
        "warning_count": len(warnings),
        "insufficient_hard_count": len(insufficient_hard),
        "insufficient_secondary_count": len(insufficient_secondary),
        "mismatches": [asdict(row) for row in mismatches],
        "insufficient_hard_evidence": insufficient_hard,
        "insufficient_secondary": insufficient_secondary,
        "normalized": normalized_payload,
        "normalized_sha256": {
            provider: _sha256_json(rows)
            for provider, rows in normalized_payload.items()
        },
        "field_semantics": FIELD_SEMANTICS,
        "policy": {
            "identity_key": "race driver number",
            "hard_fields": list(HARD_FIELDS),
            "secondary_fields": list(SECONDARY_FIELDS),
            "raw_status_class_is_audit_only": True,
            "nonfinisher_position_may_be_unknown": True,
            "missing_hard_evidence_is_not_mismatch": True,
            "missing_secondary_is_unknown": True,
            "repair_disagreements": False,
            "majority_vote": False,
        },
    }


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def collect_jolpica_raw(year: int, round_number: int, cache: Path) -> dict[str, Any]:
    client = JsonCache(Path(cache) / "jolpica")
    base = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}"
    results = client.get(f"{base}/results/?limit=100")
    pitstops = client.get(f"{base}/pitstops/?limit=200")
    try:
        total = int(results["MRData"]["total"])
        races = results["MRData"]["RaceTable"]["Races"]
        returned = len(races[0]["Results"]) if len(races) == 1 else 0
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Malformed Jolpica reconciliation response") from exc
    if total != returned:
        raise ValueError(f"Jolpica reconciliation response is incomplete: total={total}, rows={returned}")
    return {"results": results, "pitstops": pitstops, "provenance": client.provenance}


def collect_openf1_raw(session_key: int) -> dict[str, Any]:
    client = OpenF1Client()
    sessions = client.get("sessions", session_key=int(session_key))
    if len(sessions) != 1:
        raise ValueError("OpenF1 reconciliation expected exactly one session")
    if str(sessions[0].get("session_name") or "").lower() != "race":
        raise ValueError("OpenF1 session_key is not a Race session")
    meeting_key = _positive_int(sessions[0].get("meeting_key"))
    if meeting_key is None:
        raise ValueError("OpenF1 Race session has no valid meeting_key")
    meetings = client.get("meetings", meeting_key=meeting_key)
    if len(meetings) != 1:
        raise ValueError("OpenF1 reconciliation expected exactly one meeting")

    optional_collection_errors: dict[str, dict[str, Any]] = {}
    try:
        starting_grid = client.get("starting_grid", session_key=int(session_key))
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status != 404:
            raise
        starting_grid = None
        optional_collection_errors["starting_grid"] = {
            "error_type": type(exc).__name__,
            "http_status": int(status),
            "message": str(exc),
        }

    # Older OpenF1 sessions can omit optional timing collections (notably pit).
    # Absence must remain explicit evidence instead of discarding otherwise valid
    # session/meeting/result identity for the whole provider.
    collections: dict[str, Any] = {}
    for endpoint in ("laps", "pit"):
        try:
            collections[endpoint] = client.get(endpoint, session_key=int(session_key))
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 404:
                raise
            collections[endpoint] = []
            optional_collection_errors[endpoint] = {
                "error_type": type(exc).__name__,
                "http_status": int(status),
                "message": str(exc),
            }

    return {
        "session": sessions[0],
        "meeting": meetings[0],
        "session_result": client.get("session_result", session_key=int(session_key)),
        "drivers": client.get("drivers", session_key=int(session_key)),
        "laps": collections["laps"],
        "pit": collections["pit"],
        "starting_grid": starting_grid,
        "optional_collection_errors": optional_collection_errors,
        "provenance": list(client.provenance),
    }


def collect_fastf1_raw(year: int, round_number: int, cache: Path) -> dict[str, Any]:
    try:
        import fastf1
    except ImportError as exc:
        raise RuntimeError("Install telemetry dependencies to reconcile FastF1") from exc

    cache = Path(cache) / "fastf1"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(int(year), int(round_number), "R")
    session.load(telemetry=False, weather=False, messages=False)
    if session.results is None or session.results.empty:
        raise ValueError("FastF1 reconciliation returned no results")
    if session.laps is None or session.laps.empty:
        raise ValueError("FastF1 reconciliation returned no laps")

    event = session.event
    event_meta = {
        "EventName": str(event.get("EventName")),
        "RoundNumber": int(event.get("RoundNumber")),
        "EventDate": str(event.get("EventDate")),
        "Country": str(event.get("Country")),
        "Location": str(event.get("Location")),
    }
    lap_columns = [
        column
        for column in ("DriverNumber", "LapNumber", "PitInTime")
        if column in session.laps
    ]
    if not {"DriverNumber", "LapNumber"} <= set(lap_columns):
        raise ValueError("FastF1 laps lack DriverNumber/LapNumber")

    return {
        "event": event_meta,
        "results": json.loads(session.results.to_json(orient="records", date_format="iso")),
        "laps": json.loads(session.laps[lap_columns].to_json(orient="records", date_format="iso")),
        "fastf1_version": fastf1.__version__,
    }


def reconcile_completed_race(
    *,
    year: int,
    round_number: int,
    openf1_session_key: int,
    cache: Path,
    output: Path,
) -> dict[str, Any]:
    """Collect three providers, persist raw snapshots/hashes, and report disagreements."""
    output = Path(output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jolpica = collect_jolpica_raw(year, round_number, cache)
    openf1 = collect_openf1_raw(openf1_session_key)
    fastf1 = collect_fastf1_raw(year, round_number, cache)

    _write_json(raw_dir / "jolpica.json", jolpica)
    _write_json(raw_dir / "openf1.json", openf1)
    _write_json(raw_dir / "fastf1.json", fastf1)

    event_identity = build_event_identity(
        year=year,
        round_number=round_number,
        openf1_session_key=openf1_session_key,
        jolpica_raw=jolpica,
        openf1_raw=openf1,
        fastf1_raw=fastf1,
    )
    require_event_identity(event_identity)

    jolpica_rows, jolpica_meta = normalize_jolpica_results(
        jolpica["results"],
        jolpica["pitstops"],
    )
    openf1_rows = normalize_openf1_results(
        openf1["session_result"],
        openf1["drivers"],
        openf1["pit"],
        openf1.get("starting_grid"),
    )
    fastf1_rows = normalize_fastf1_results(fastf1["results"], fastf1["laps"])

    session = openf1["session"]
    if _positive_int(session.get("year")) != int(year):
        raise ValueError(f"OpenF1 session year mismatch: expected {year}, got {session.get('year')}")
    if jolpica_meta.get("round") != int(round_number):
        raise ValueError("Jolpica round metadata mismatch")
    if _positive_int(fastf1["event"].get("RoundNumber")) != int(round_number):
        raise ValueError("FastF1 round metadata mismatch")

    report = reconcile_results({
        "Jolpica": jolpica_rows,
        "OpenF1": openf1_rows,
        "FastF1": fastf1_rows,
    })
    report.update({
        "year": int(year),
        "round": int(round_number),
        "openf1_session_key": int(openf1_session_key),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "event_identity": event_identity,
        "event_metadata": {
            "jolpica": jolpica_meta,
            "openf1_session": session,
            "openf1_meeting": openf1["meeting"],
            "fastf1": fastf1["event"],
        },
        "raw_sha256": {
            "Jolpica": _sha256_json(jolpica),
            "OpenF1": _sha256_json(openf1),
            "FastF1": _sha256_json(fastf1),
        },
        "limitations": [
            "Agreement among public providers does not make them statistically independent sources.",
            "Result reconciliation verifies completed-race facts, not live publication latency.",
            "FastF1 lap count may be derived from maximum observed LapNumber when Results.Laps is absent.",
            "Some providers omit a non-finisher classification position; this stays explicit evidence gap.",
            "Secondary fields are compared only when both providers expose evidence; missing evidence stays unknown.",
            "A provider disagreement is reported and never resolved by majority vote.",
            "OpenF1 session_key is explicit; this tool does not guess which session belongs to a round.",
        ],
    })

    _write_json(output / "reconciliation.json", report)
    pd.DataFrame(report["mismatches"]).to_csv(output / "mismatches.csv", index=False)
    pd.DataFrame(report["insufficient_hard_evidence"]).to_csv(
        output / "insufficient_hard_evidence.csv",
        index=False,
    )
    pd.DataFrame(report["insufficient_secondary"]).to_csv(
        output / "insufficient_secondary.csv",
        index=False,
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile a completed race across three public providers"
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--openf1-session-key", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/reconciliation-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = reconcile_completed_race(
        year=args.year,
        round_number=args.round_number,
        openf1_session_key=args.openf1_session_key,
        cache=args.cache,
        output=args.output,
    )
    print(json.dumps({
        "passed": report["passed"],
        "verification_status": report["verification_status"],
        "hard_mismatch_count": report["hard_mismatch_count"],
        "warning_count": report["warning_count"],
        "insufficient_hard_count": report["insufficient_hard_count"],
        "insufficient_secondary_count": report["insufficient_secondary_count"],
        "output": str(args.output),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
