"""Cross-provider F1 data reconciliation without majority-vote truth invention.

The goal is evidence, not silent repair. Jolpica, OpenF1 and FastF1 are normalized
into a small race-result contract. Fields are compared driver-by-driver. A disagreement
remains unresolved in the report; this module never elects a provider as the truth.

Network collection is intentionally separate from the pure normalization/comparison
functions so the normal test suite stays deterministic and provider outages do not
break ordinary CI.
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

PROVIDERS = ("jolpica", "openf1", "fastf1")
CRITICAL_FIELDS = ("finish_position", "completed_laps")
SECONDARY_FIELDS = ("grid_position", "pit_stops", "dnf", "dns", "dsq", "points")


@dataclass(frozen=True)
class ProviderDriverRecord:
    provider: str
    driver_number: int | None
    driver_id: str | None = None
    team: str | None = None
    finish_position: int | None = None
    grid_position: int | None = None
    completed_laps: int | None = None
    pit_stops: int | None = None
    dnf: bool | None = None
    dns: bool | None = None
    dsq: bool | None = None
    points: float | None = None
    status_raw: str | None = None


@dataclass(frozen=True)
class FieldComparison:
    driver_number: int
    field: str
    severity: str
    status: str
    values: dict[str, Any]
    consensus: Any | None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _integer(value: Any, *, allow_zero: bool = False) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    result = int(number)
    if allow_zero:
        return result if result >= 0 else None
    return result if result >= 1 else None


def _float(value: Any) -> float | None:
    return _finite_number(value)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jolpica_status(status: str | None) -> tuple[bool | None, bool | None, bool | None]:
    """Conservative status mapping; raw text is always retained in the record."""
    text = str(status or "").strip()
    upper = text.upper()
    if not text:
        return None, None, None
    dsq = "DISQUAL" in upper or upper == "DSQ"
    dns = "DID NOT START" in upper or upper == "DNS" or "WITHDREW" in upper
    if dsq or dns:
        return False, dns, dsq
    if text == "Finished" or text.startswith("+"):
        return False, False, False
    return True, False, False


def normalize_jolpica(
    result_payload: dict[str, Any],
    pit_payload: dict[str, Any] | None = None,
) -> list[ProviderDriverRecord]:
    races = result_payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if len(races) != 1:
        raise ValueError(f"Expected exactly one Jolpica race, found {len(races)}")
    race = races[0]
    results = race.get("Results")
    if not isinstance(results, list) or not results:
        raise ValueError("Jolpica race has no Results")

    pit_counts: dict[str, int] = {}
    if pit_payload:
        pit_races = pit_payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if len(pit_races) > 1:
            raise ValueError("Jolpica pit-stop payload contains multiple races")
        for stop in (pit_races[0].get("PitStops", []) if pit_races else []):
            driver_id = str(stop.get("driverId") or "")
            if driver_id:
                pit_counts[driver_id] = pit_counts.get(driver_id, 0) + 1

    output: list[ProviderDriverRecord] = []
    for result in results:
        driver = result.get("Driver") or {}
        constructor = result.get("Constructor") or {}
        driver_id = str(driver.get("driverId") or "") or None
        status_raw = str(result.get("status") or "") or None
        dnf, dns, dsq = _jolpica_status(status_raw)
        number = _integer(driver.get("permanentNumber"))
        output.append(ProviderDriverRecord(
            provider="jolpica",
            driver_number=number,
            driver_id=driver_id,
            team=str(constructor.get("constructorId") or "") or None,
            finish_position=_integer(result.get("position")),
            grid_position=_integer(result.get("grid"), allow_zero=True),
            completed_laps=_integer(result.get("laps"), allow_zero=True),
            pit_stops=pit_counts.get(driver_id, 0) if driver_id is not None else None,
            dnf=dnf,
            dns=dns,
            dsq=dsq,
            points=_float(result.get("points")),
            status_raw=status_raw,
        ))
    return output


def normalize_openf1(
    result_rows: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]] | None = None,
    pit_rows: list[dict[str, Any]] | None = None,
) -> list[ProviderDriverRecord]:
    if not result_rows:
        raise ValueError("OpenF1 session_result is empty")
    driver_lookup: dict[int, dict[str, Any]] = {}
    for row in driver_rows or []:
        number = _integer(row.get("driver_number"))
        if number is not None:
            driver_lookup[number] = row

    pit_keys: dict[int, set[tuple[Any, Any]]] = {}
    for row in pit_rows or []:
        number = _integer(row.get("driver_number"))
        if number is None:
            continue
        key = (row.get("lap_number"), row.get("date"))
        pit_keys.setdefault(number, set()).add(key)

    output: list[ProviderDriverRecord] = []
    for row in result_rows:
        number = _integer(row.get("driver_number"))
        meta = driver_lookup.get(number or -1, {})
        output.append(ProviderDriverRecord(
            provider="openf1",
            driver_number=number,
            driver_id=str(meta.get("name_acronym") or "") or None,
            team=str(meta.get("team_name") or "") or None,
            finish_position=_integer(row.get("position")),
            grid_position=None,
            completed_laps=_integer(row.get("number_of_laps"), allow_zero=True),
            pit_stops=len(pit_keys.get(number, set())) if number is not None else None,
            dnf=bool(row.get("dnf")) if row.get("dnf") is not None else None,
            dns=bool(row.get("dns")) if row.get("dns") is not None else None,
            dsq=bool(row.get("dsq")) if row.get("dsq") is not None else None,
            points=None,
            status_raw=None,
        ))
    return output


def _fastf1_status(status: Any) -> tuple[bool | None, bool | None, bool | None]:
    text = str(status or "").strip()
    upper = text.upper()
    if not text or upper == "NAN":
        return None, None, None
    dsq = "DISQUAL" in upper or upper == "DSQ"
    dns = "DID NOT START" in upper or upper == "DNS" or "WITHDRAW" in upper
    if dsq or dns:
        return False, dns, dsq
    if text == "Finished" or text.startswith("+"):
        return False, False, False
    return True, False, False


def normalize_fastf1(results: pd.DataFrame, laps: pd.DataFrame) -> list[ProviderDriverRecord]:
    if results is None or results.empty:
        raise ValueError("FastF1 results are empty")
    result_frame = results.copy()
    lap_frame = laps.copy() if laps is not None else pd.DataFrame()

    lap_counts: dict[int, int] = {}
    pit_counts: dict[int, int] = {}
    if not lap_frame.empty and "DriverNumber" in lap_frame:
        numbers = pd.to_numeric(lap_frame["DriverNumber"], errors="coerce")
        lap_numbers = pd.to_numeric(lap_frame.get("LapNumber"), errors="coerce")
        for number in sorted(numbers.dropna().unique()):
            mask = numbers.eq(number)
            driver = int(number)
            finite_laps = lap_numbers[mask].dropna()
            lap_counts[driver] = int(finite_laps.max()) if not finite_laps.empty else 0
            if "PitInTime" in lap_frame:
                pit_counts[driver] = int(lap_frame.loc[mask, "PitInTime"].notna().sum())

    output: list[ProviderDriverRecord] = []
    for _, row in result_frame.iterrows():
        number = _integer(row.get("DriverNumber"))
        status_raw = str(row.get("Status") or "") or None
        dnf, dns, dsq = _fastf1_status(status_raw)
        output.append(ProviderDriverRecord(
            provider="fastf1",
            driver_number=number,
            driver_id=str(row.get("Abbreviation") or "") or None,
            team=str(row.get("TeamName") or "") or None,
            finish_position=_integer(row.get("Position")),
            grid_position=_integer(row.get("GridPosition"), allow_zero=True),
            completed_laps=lap_counts.get(number) if number is not None else None,
            pit_stops=pit_counts.get(number, 0) if number is not None else None,
            dnf=dnf,
            dns=dns,
            dsq=dsq,
            points=_float(row.get("Points")),
            status_raw=status_raw,
        ))
    return output


def _value_equal(field: str, left: Any, right: Any) -> bool:
    if field == "points":
        try:
            return bool(np.isclose(float(left), float(right), atol=1e-9, rtol=0.0))
        except (TypeError, ValueError):
            return left == right
    return left == right


def _compare_field(driver_number: int, field: str, records: dict[str, ProviderDriverRecord]) -> FieldComparison:
    values = {
        provider: getattr(record, field)
        for provider, record in records.items()
        if getattr(record, field) is not None
    }
    severity = "critical" if field in CRITICAL_FIELDS else "secondary"
    if len(values) < 2:
        return FieldComparison(driver_number, field, severity, "INSUFFICIENT", values, None)
    observed = list(values.values())
    first = observed[0]
    agree = all(_value_equal(field, first, value) for value in observed[1:])
    return FieldComparison(
        driver_number=driver_number,
        field=field,
        severity=severity,
        status="PASS" if agree else "FAIL",
        values=values,
        consensus=first if agree else None,
    )


def reconcile(records: Iterable[ProviderDriverRecord]) -> dict[str, Any]:
    rows = list(records)
    unknown_identity = [asdict(row) for row in rows if row.driver_number is None]
    by_driver: dict[int, dict[str, ProviderDriverRecord]] = {}
    for row in rows:
        if row.provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {row.provider}")
        if row.driver_number is None:
            continue
        bucket = by_driver.setdefault(row.driver_number, {})
        if row.provider in bucket:
            raise ValueError(f"Duplicate {row.provider} record for driver {row.driver_number}")
        bucket[row.provider] = row

    comparisons: list[FieldComparison] = []
    presence: list[dict[str, Any]] = []
    for number in sorted(by_driver):
        available = by_driver[number]
        missing_providers = [provider for provider in PROVIDERS if provider not in available]
        presence.append({
            "driver_number": number,
            "providers": sorted(available),
            "missing_providers": missing_providers,
            "status": "PASS" if not missing_providers else "FAIL",
        })
        for field in CRITICAL_FIELDS + SECONDARY_FIELDS:
            comparisons.append(_compare_field(number, field, available))

    critical_failures = sum(
        row.status == "FAIL" and row.severity == "critical" for row in comparisons
    ) + sum(item["status"] == "FAIL" for item in presence) + len(unknown_identity)
    secondary_failures = sum(
        row.status == "FAIL" and row.severity == "secondary" for row in comparisons
    )
    insufficient = sum(row.status == "INSUFFICIENT" for row in comparisons)

    normalized = {
        provider: [
            asdict(row)
            for row in sorted(
                (item for item in rows if item.provider == provider),
                key=lambda item: (item.driver_number is None, item.driver_number or 9999),
            )
        ]
        for provider in PROVIDERS
    }
    return {
        "schema_version": 1,
        "evidence_kind": "cross_provider_race_reconciliation",
        "providers": list(PROVIDERS),
        "provider_normalized_sha256": {
            provider: _canonical_hash(normalized[provider]) for provider in PROVIDERS
        },
        "summary": {
            "drivers": len(by_driver),
            "critical_failures": int(critical_failures),
            "secondary_failures": int(secondary_failures),
            "insufficient_comparisons": int(insufficient),
            "unknown_identity_rows": len(unknown_identity),
            "truth_elected_on_disagreement": False,
        },
        "presence": presence,
        "comparisons": [asdict(row) for row in comparisons],
        "unknown_identity": unknown_identity,
        "normalized": normalized,
    }


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _jolpica_single_race(client: JsonCache, year: int, round_number: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}"
    result = client.get(f"{base}/results/?limit=100")
    pit = client.get(f"{base}/pitstops/?limit=200")
    return result, pit


def _jolpica_race_date(result_payload: dict[str, Any]) -> pd.Timestamp:
    races = result_payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if len(races) != 1 or not races[0].get("date"):
        raise ValueError("Cannot resolve Jolpica race date")
    return pd.to_datetime(races[0]["date"], utc=True, errors="raise")


def _resolve_openf1_race_session(client: OpenF1Client, year: int, race_date: pd.Timestamp) -> dict[str, Any]:
    sessions = client.get("sessions", year=year, session_name="Race")
    candidates = []
    for row in sessions:
        observed = pd.to_datetime(row.get("date_start"), utc=True, errors="coerce")
        if pd.isna(observed):
            continue
        distance_days = abs((observed.normalize() - race_date.normalize()).days)
        if distance_days <= 1:
            candidates.append((distance_days, observed, row))
    if not candidates:
        raise ValueError("No OpenF1 Race session matched the Jolpica race date within one day")
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance = candidates[0][0]
    best = [item for item in candidates if item[0] == best_distance]
    if len(best) != 1:
        keys = [item[2].get("session_key") for item in best]
        raise ValueError(f"Ambiguous OpenF1 Race session candidates: {keys}")
    return best[0][2]


def collect_reconciliation(year: int, round_number: int, output: Path, cache: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)

    jolpica_client = JsonCache(cache / "jolpica", delay=0.0)
    jolpica_result, jolpica_pit = _jolpica_single_race(jolpica_client, year, round_number)
    race_date = _jolpica_race_date(jolpica_result)
    _save_json(raw / "jolpica_results.json", jolpica_result)
    _save_json(raw / "jolpica_pitstops.json", jolpica_pit)

    openf1 = OpenF1Client.from_env()
    session = _resolve_openf1_race_session(openf1, year, race_date)
    session_key = int(session["session_key"])
    openf1_results = openf1.get("session_result", session_key=session_key)
    openf1_drivers = openf1.get("drivers", session_key=session_key)
    openf1_pit = openf1.get("pit", session_key=session_key)
    _save_json(raw / "openf1_session.json", session)
    _save_json(raw / "openf1_results.json", openf1_results)
    _save_json(raw / "openf1_drivers.json", openf1_drivers)
    _save_json(raw / "openf1_pit.json", openf1_pit)

    try:
        import fastf1
    except ImportError as exc:
        raise RuntimeError("Install the telemetry extra for three-provider reconciliation") from exc
    fastf1.Cache.enable_cache(str(cache / "fastf1"))
    fast_session = fastf1.get_session(year, round_number, "R")
    fast_session.load(telemetry=False, weather=False, messages=False)
    fast_results = fast_session.results.copy()
    fast_laps = fast_session.laps.copy()
    fast_results.to_csv(raw / "fastf1_results.csv", index=False)
    lap_columns = [
        column for column in ("DriverNumber", "LapNumber", "PitInTime", "PitOutTime", "LapTime")
        if column in fast_laps.columns
    ]
    fast_laps[lap_columns].to_csv(raw / "fastf1_laps.csv", index=False)

    records = [
        *normalize_jolpica(jolpica_result, jolpica_pit),
        *normalize_openf1(openf1_results, openf1_drivers, openf1_pit),
        *normalize_fastf1(fast_results, fast_laps),
    ]
    report = reconcile(records)
    report["run"] = {
        "year": int(year),
        "round": int(round_number),
        "openf1_session_key": session_key,
        "generated_at": datetime.now(UTC).isoformat(),
        "jolpica_requests": jolpica_client.provenance,
        "fastf1_version": getattr(fastf1, "__version__", None),
    }
    _save_json(output / "reconciliation.json", report)

    comparison_rows = []
    for row in report["comparisons"]:
        comparison_rows.append({
            "driver_number": row["driver_number"],
            "field": row["field"],
            "severity": row["severity"],
            "status": row["status"],
            "jolpica": row["values"].get("jolpica"),
            "openf1": row["values"].get("openf1"),
            "fastf1": row["values"].get("fastf1"),
            "consensus": row["consensus"],
        })
    pd.DataFrame(comparison_rows).to_csv(output / "comparisons.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reconcile one F1 race across three public providers")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/provider-reconciliation"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/provider-reconciliation"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on critical disagreement")
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    report = collect_reconciliation(args.year, args.round_number, args.output, args.cache)
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    if args.strict and summary["critical_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
