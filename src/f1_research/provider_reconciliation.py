"""Cross-provider numerical reconciliation for completed F1 races.

The purpose of this module is data verification, not prediction. It compares the
same completed race across Jolpica, OpenF1 and FastF1 after normalizing each source
into a deliberately small canonical result schema. A mismatch is reported; it is
never silently repaired by majority vote.

Driver identity uses the race number exposed by each source (Jolpica ``Results.number``,
OpenF1 ``driver_number`` and FastF1 ``DriverNumber``). Names/codes are retained only
as audit context and are never used to force a match when numbers disagree.
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

CANONICAL_FIELDS = ("position", "laps", "status_class")


@dataclass(frozen=True)
class ResultRow:
    provider: str
    driver_number: int
    position: int | None
    laps: int | None
    status_class: str | None
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


def _positive_int(value: Any, *, allow_zero: bool = False) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    lower = 0 if allow_zero else 1
    if not np.isfinite(number) or number < lower or not number.is_integer():
        return None
    return int(number)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    if bool(row.get("dsq")):
        return "dsq"
    if bool(row.get("dns")):
        return "dns"
    if bool(row.get("dnf")):
        return "dnf"
    gap = (_clean_text(row.get("gap_to_leader")) or "").upper()
    if "LAP" in gap:
        return "classified_lapped"
    if _positive_int(row.get("position")) is not None:
        return "finished"
    return None


def normalize_jolpica_results(payload: dict[str, Any]) -> tuple[list[ResultRow], dict[str, Any]]:
    """Normalize one Jolpica race-results response without guessing missing fields."""
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
        rows.append(ResultRow(
            provider="Jolpica",
            driver_number=number,
            position=_positive_int(raw.get("position")),
            laps=_positive_int(raw.get("laps"), allow_zero=True),
            status_class=_status_from_text(raw.get("status"), raw.get("positionText")),
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
        "circuit_id": _clean_text((race.get("Circuit") or {}).get("circuitId"))
        if isinstance(race.get("Circuit"), dict)
        else None,
    }
    return _validated_rows(rows), metadata


def normalize_openf1_results(
    result_rows: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]] | None = None,
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

    rows: list[ResultRow] = []
    for raw in result_rows:
        if not isinstance(raw, dict):
            raise ValueError("OpenF1 result row must be an object")
        number = _positive_int(raw.get("driver_number"))
        if number is None:
            raise ValueError("OpenF1 result row has no valid driver_number")
        driver = lookup.get(number, {})
        rows.append(ResultRow(
            provider="OpenF1",
            driver_number=number,
            position=_positive_int(raw.get("position")),
            laps=_positive_int(raw.get("number_of_laps"), allow_zero=True),
            status_class=_openf1_status(raw),
            status_raw=(
                "dsq" if raw.get("dsq") else "dns" if raw.get("dns") else "dnf" if raw.get("dnf") else "classified"
            ),
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

    lap_counts: dict[int, int] = {}
    if laps is not None:
        lap_frame = pd.DataFrame(laps).copy()
        if not lap_frame.empty and {"DriverNumber", "LapNumber"} <= set(lap_frame):
            lap_frame["DriverNumber"] = pd.to_numeric(lap_frame["DriverNumber"], errors="coerce")
            lap_frame["LapNumber"] = pd.to_numeric(lap_frame["LapNumber"], errors="coerce")
            for number, group in lap_frame.dropna(subset=["DriverNumber", "LapNumber"]).groupby("DriverNumber"):
                lap_counts[int(number)] = int(group["LapNumber"].max())

    rows: list[ResultRow] = []
    for raw in frame.to_dict("records"):
        number = _positive_int(raw.get("DriverNumber"))
        if number is None:
            raise ValueError("FastF1 result row has no valid DriverNumber")
        position = _positive_int(raw.get("Position"))
        classified = raw.get("ClassifiedPosition")
        status = raw.get("Status")
        status_class = _status_from_text(status, classified)
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
            position=position,
            laps=result_laps,
            status_class=status_class,
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


def _provider_integrity(provider: str, rows: list[ResultRow]) -> list[Mismatch]:
    output: list[Mismatch] = []
    positions = [row.position for row in rows if row.position is not None]
    if len(positions) != len(rows):
        missing = [row.driver_number for row in rows if row.position is None]
        output.append(Mismatch(
            provider, provider, None, "classification_integrity", missing, None, "hard",
            "provider has result rows without final position",
        ))
    if len(positions) != len(set(positions)):
        output.append(Mismatch(
            provider, provider, None, "classification_integrity", positions, None, "hard",
            "provider has duplicate final positions",
        ))
    if len(positions) == len(rows) and sorted(positions) != list(range(1, len(rows) + 1)):
        output.append(Mismatch(
            provider, provider, None, "classification_integrity", sorted(positions), None, "hard",
            "provider classification is not consecutive 1..N",
        ))
    for row in rows:
        for field in CANONICAL_FIELDS:
            if getattr(row, field) is None:
                output.append(Mismatch(
                    provider, provider, row.driver_number, field, None, None, "hard",
                    "provider is missing a required reconciliation field",
                ))
    return output


def _row_index(rows: Iterable[ResultRow]) -> dict[int, ResultRow]:
    return {row.driver_number: row for row in rows}


def reconcile_results(provider_rows: dict[str, list[ResultRow]]) -> dict[str, Any]:
    """Compare normalized provider results without repairing disagreements."""
    if len(provider_rows) < 2:
        raise ValueError("At least two providers are required for reconciliation")
    normalized = {name: _validated_rows(rows) for name, rows in provider_rows.items()}
    providers = sorted(normalized)
    mismatches: list[Mismatch] = []

    for provider in providers:
        mismatches.extend(_provider_integrity(provider, normalized[provider]))

    for i, provider_a in enumerate(providers):
        for provider_b in providers[i + 1:]:
            left = _row_index(normalized[provider_a])
            right = _row_index(normalized[provider_b])
            left_numbers, right_numbers = set(left), set(right)
            for number in sorted(left_numbers - right_numbers):
                mismatches.append(Mismatch(
                    provider_a, provider_b, number, "driver_presence", True, False, "hard",
                    "driver exists only in first provider",
                ))
            for number in sorted(right_numbers - left_numbers):
                mismatches.append(Mismatch(
                    provider_a, provider_b, number, "driver_presence", False, True, "hard",
                    "driver exists only in second provider",
                ))
            for number in sorted(left_numbers & right_numbers):
                a, b = left[number], right[number]
                for field in CANONICAL_FIELDS:
                    value_a, value_b = getattr(a, field), getattr(b, field)
                    if value_a != value_b:
                        mismatches.append(Mismatch(
                            provider_a, provider_b, number, field, value_a, value_b, "hard",
                            "provider values disagree",
                        ))
                if a.driver_code and b.driver_code and a.driver_code.upper() != b.driver_code.upper():
                    mismatches.append(Mismatch(
                        provider_a, provider_b, number, "driver_code", a.driver_code, b.driver_code,
                        "warning", "audit identity label differs; number match is retained",
                    ))

    hard = [row for row in mismatches if row.severity == "hard"]
    warning = [row for row in mismatches if row.severity == "warning"]
    row_counts = {provider: len(rows) for provider, rows in normalized.items()}
    return {
        "schema_version": 1,
        "kind": "cross_provider_completed_race_reconciliation",
        "providers": providers,
        "row_counts": row_counts,
        "passed": not hard,
        "hard_mismatch_count": len(hard),
        "warning_count": len(warning),
        "mismatches": [asdict(row) for row in mismatches],
        "normalized": {
            provider: [asdict(row) for row in rows]
            for provider, rows in normalized.items()
        },
        "policy": {
            "identity_key": "race driver number",
            "hard_fields": list(CANONICAL_FIELDS),
            "repair_disagreements": False,
            "majority_vote": False,
        },
    }


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def collect_jolpica_raw(year: int, round_number: int, cache: Path) -> dict[str, Any]:
    client = JsonCache(Path(cache) / "jolpica")
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/results/?limit=100"
    payload = client.get(url)
    try:
        total = int(payload["MRData"]["total"])
        races = payload["MRData"]["RaceTable"]["Races"]
        returned = len(races[0]["Results"]) if len(races) == 1 else 0
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Malformed Jolpica reconciliation response") from exc
    if total != returned:
        raise ValueError(f"Jolpica reconciliation response is incomplete: total={total}, rows={returned}")
    return payload


def collect_openf1_raw(session_key: int) -> dict[str, Any]:
    client = OpenF1Client()
    sessions = client.get("sessions", session_key=int(session_key))
    if len(sessions) != 1:
        raise ValueError("OpenF1 reconciliation expected exactly one session")
    if str(sessions[0].get("session_name") or "").lower() != "race":
        raise ValueError("OpenF1 session_key is not a Race session")
    return {
        "session": sessions[0],
        "session_result": client.get("session_result", session_key=int(session_key)),
        "drivers": client.get("drivers", session_key=int(session_key)),
        "laps": client.get("laps", session_key=int(session_key)),
        "pit": client.get("pit", session_key=int(session_key)),
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
    }
    lap_columns = [column for column in ("DriverNumber", "LapNumber") if column in session.laps]
    if set(lap_columns) != {"DriverNumber", "LapNumber"}:
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
    """Collect three providers, persist raw snapshots/hashes, and fail visibly on disagreement."""
    output = Path(output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jolpica = collect_jolpica_raw(year, round_number, cache)
    openf1 = collect_openf1_raw(openf1_session_key)
    fastf1 = collect_fastf1_raw(year, round_number, cache)

    _write_json(raw_dir / "jolpica.json", jolpica)
    _write_json(raw_dir / "openf1.json", openf1)
    _write_json(raw_dir / "fastf1.json", fastf1)

    jolpica_rows, jolpica_meta = normalize_jolpica_results(jolpica)
    openf1_rows = normalize_openf1_results(openf1["session_result"], openf1["drivers"])
    fastf1_rows = normalize_fastf1_results(fastf1["results"], fastf1["laps"])

    session = openf1["session"]
    openf1_year = _positive_int(session.get("year"))
    if openf1_year != int(year):
        raise ValueError(f"OpenF1 session year mismatch: expected {year}, got {openf1_year}")
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
        "event_metadata": {
            "jolpica": jolpica_meta,
            "openf1": session,
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
            "A provider disagreement is reported and never resolved by majority vote.",
            "OpenF1 session_key is explicit; this tool does not guess which session belongs to a round.",
        ],
    })
    _write_json(output / "reconciliation.json", report)
    pd.DataFrame(report["mismatches"]).to_csv(output / "mismatches.csv", index=False)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Reconcile a completed race across three public providers")
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
        "hard_mismatch_count": report["hard_mismatch_count"],
        "warning_count": report["warning_count"],
        "output": str(args.output),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
