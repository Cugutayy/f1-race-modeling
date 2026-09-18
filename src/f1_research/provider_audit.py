"""Resilient real-network provider audit.

The pure reconciliation contract lives in :mod:`provider_reconciliation`. This module
adds failure-aware collection for real public providers. A provider outage or a
partially loaded FastF1 session is recorded as evidence; it never crashes before an
audit artifact can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .provider_event_identity import build_event_identity
from .provider_reconciliation import (
    collect_jolpica_raw,
    collect_openf1_raw,
    normalize_fastf1_results,
    normalize_jolpica_results,
    normalize_openf1_results,
    reconcile_results,
)
from .revision import current_git_sha


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def collect_fastf1_partial(year: int, round_number: int, cache: Path) -> dict[str, Any]:
    """Collect as much FastF1 evidence as is actually available."""
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastF1 is not installed") from exc

    cache_dir = Path(cache) / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(int(year), int(round_number), "R")
    errors: dict[str, str] = {}
    try:
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as exc:  # provider/library failure becomes audit evidence
        errors["session_load"] = f"{type(exc).__name__}: {exc}"

    results: list[dict[str, Any]] = []
    result_columns: list[str] = []
    try:
        frame = session.results
        if frame is not None and not frame.empty:
            result_columns = list(frame.columns)
            results = json.loads(frame.to_json(orient="records", date_format="iso"))
        else:
            errors["results"] = "FastF1 result table is empty"
    except Exception as exc:
        errors["results"] = f"{type(exc).__name__}: {exc}"

    laps: list[dict[str, Any]] | None = None
    lap_columns: list[str] = []
    try:
        lap_frame = session.laps
        if lap_frame is not None and not lap_frame.empty:
            lap_columns = [
                column
                for column in ("DriverNumber", "LapNumber", "PitInTime")
                if column in lap_frame.columns
            ]
            if {"DriverNumber", "LapNumber"} <= set(lap_columns):
                laps = json.loads(
                    lap_frame[lap_columns].to_json(orient="records", date_format="iso")
                )
            else:
                errors["laps"] = "FastF1 lap table lacks DriverNumber/LapNumber"
        else:
            errors["laps"] = "FastF1 lap table is empty"
    except Exception as exc:
        errors["laps"] = f"{type(exc).__name__}: {exc}"

    event: dict[str, Any] = {}
    try:
        raw_event = session.event
        event = {
            "EventName": str(raw_event.get("EventName")),
            "RoundNumber": int(raw_event.get("RoundNumber")),
            "EventDate": str(raw_event.get("EventDate")),
            "Country": str(raw_event.get("Country")),
            "Location": str(raw_event.get("Location")),
        }
    except Exception as exc:
        errors["event"] = f"{type(exc).__name__}: {exc}"

    return {
        "provider": "FastF1",
        "fastf1_version": fastf1.__version__,
        "event": event,
        "results": results,
        "results_columns": result_columns,
        "laps": laps,
        "lap_columns": lap_columns,
        "collection_errors": errors,
        "results_available": bool(results),
        "laps_available": laps is not None and len(laps) > 0,
    }


def _failure_report(
    *,
    year: int,
    round_number: int,
    openf1_session_key: int,
    provider_errors: dict[str, str],
    raw_snapshots: dict[str, Any],
    producer_git_sha: str | None = None,
) -> dict[str, Any]:
    producer_git_sha = producer_git_sha or current_git_sha(required=True)
    return {
        "schema_version": 4,
        "artifact_schema_version": 2,
        "reconciliation_schema_version": None,
        "kind": "cross_provider_completed_race_reconciliation",
        "year": int(year),
        "round": int(round_number),
        "openf1_session_key": int(openf1_session_key),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "producer_git_sha": producer_git_sha,
        "passed": False,
        "verification_status": "FAIL",
        "hard_mismatch_count": len(provider_errors),
        "warning_count": 0,
        "insufficient_hard_count": 0,
        "insufficient_secondary_count": 0,
        "provider_errors": provider_errors,
        "mismatches": [],
        "insufficient_hard_evidence": [],
        "insufficient_secondary": [],
        "raw_sha256": {
            name: _json_sha256(value) for name, value in raw_snapshots.items()
        },
        "policy": {
            "provider_failure_is_hard": True,
            "repair_disagreements": False,
            "majority_vote": False,
            "missing_evidence_is_not_zero": True,
        },
    }


def audit_completed_race(
    *,
    year: int,
    round_number: int,
    openf1_session_key: int,
    cache: Path,
    output: Path,
) -> dict[str, Any]:
    """Collect real providers, preserve partial evidence, and always emit an audit report."""
    output = Path(output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    producer_git_sha = current_git_sha(required=True)
    raw: dict[str, Any] = {}
    provider_errors: dict[str, str] = {}

    try:
        raw["Jolpica"] = collect_jolpica_raw(year, round_number, cache)
    except Exception as exc:
        provider_errors["Jolpica"] = f"{type(exc).__name__}: {exc}"

    try:
        raw["OpenF1"] = collect_openf1_raw(openf1_session_key)
    except Exception as exc:
        provider_errors["OpenF1"] = f"{type(exc).__name__}: {exc}"

    try:
        raw["FastF1"] = collect_fastf1_partial(year, round_number, cache)
    except Exception as exc:
        provider_errors["FastF1"] = f"{type(exc).__name__}: {exc}"

    for provider, payload in raw.items():
        _write_json(raw_dir / f"{provider.lower()}.json", payload)

    event_identity = build_event_identity(
        year=year,
        round_number=round_number,
        openf1_session_key=openf1_session_key,
        jolpica_raw=raw.get("Jolpica"),
        openf1_raw=raw.get("OpenF1"),
        fastf1_raw=raw.get("FastF1"),
    )
    if not event_identity["verified"]:
        provider_errors.setdefault(
            "event_identity",
            "failed required checks: " + ", ".join(event_identity["failures"]),
        )

    normalized: dict[str, list[Any]] = {}
    if "Jolpica" in raw:
        try:
            rows, _meta = normalize_jolpica_results(
                raw["Jolpica"]["results"], raw["Jolpica"]["pitstops"]
            )
            normalized["Jolpica"] = rows
        except Exception as exc:
            provider_errors["Jolpica.normalize"] = f"{type(exc).__name__}: {exc}"

    if "OpenF1" in raw:
        try:
            normalized["OpenF1"] = normalize_openf1_results(
                raw["OpenF1"]["session_result"],
                raw["OpenF1"]["drivers"],
                raw["OpenF1"]["pit"],
                raw["OpenF1"].get("starting_grid"),
            )
        except Exception as exc:
            provider_errors["OpenF1.normalize"] = f"{type(exc).__name__}: {exc}"

    if "FastF1" in raw and raw["FastF1"].get("results_available"):
        try:
            normalized["FastF1"] = normalize_fastf1_results(
                raw["FastF1"]["results"], raw["FastF1"].get("laps")
            )
        except Exception as exc:
            provider_errors["FastF1.normalize"] = f"{type(exc).__name__}: {exc}"

    if len(normalized) == 3 and event_identity["verified"]:
        report = reconcile_results(normalized)
        report["artifact_schema_version"] = 2
        report["reconciliation_schema_version"] = report["schema_version"]
        report.update({
            "year": int(year),
            "round": int(round_number),
            "openf1_session_key": int(openf1_session_key),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "producer_git_sha": producer_git_sha,
            "provider_errors": provider_errors,
            "event_identity": event_identity,
            "event_metadata": event_identity["providers"],
            "raw_sha256": {name: _json_sha256(value) for name, value in raw.items()},
        })
        if provider_errors:
            report["passed"] = False
            report["verification_status"] = "FAIL"
            report["hard_mismatch_count"] = int(report["hard_mismatch_count"]) + len(provider_errors)
    else:
        if len(normalized) != 3:
            provider_errors.setdefault(
                "audit",
                f"expected three normalized providers, got {sorted(normalized)}",
            )
        report = _failure_report(
            year=year,
            round_number=round_number,
            openf1_session_key=openf1_session_key,
            provider_errors=provider_errors,
            raw_snapshots=raw,
            producer_git_sha=producer_git_sha,
        )
        report["event_identity"] = event_identity
        report["event_metadata"] = event_identity["providers"]

    report["limitations"] = [
        "Public-provider agreement is not statistical independence or official FIA certification.",
        "FastF1 can expose result metadata even when timing/lap loading fails; partial evidence stays explicit.",
        "A provider-specific absence of non-finisher position is retained as an evidence gap, not imputed.",
        "Missing secondary evidence remains unknown rather than zero/false.",
        "Provider disagreements are never repaired by majority vote.",
        "Result comparison runs only after required same-event metadata checks pass across all three providers.",
        "Location labels are retained but are not a hard identity key because provider locality semantics differ.",
    ]

    _write_json(output / "reconciliation.json", report)
    pd.DataFrame(report.get("mismatches", [])).to_csv(output / "mismatches.csv", index=False)
    pd.DataFrame(report.get("insufficient_hard_evidence", [])).to_csv(
        output / "insufficient_hard_evidence.csv", index=False
    )
    pd.DataFrame(report.get("insufficient_secondary", [])).to_csv(
        output / "insufficient_secondary.csv", index=False
    )
    _write_json(output / "provider_errors.json", provider_errors)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Failure-aware public-provider F1 audit")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--openf1-session-key", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/reconciliation-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = audit_completed_race(
        year=args.year,
        round_number=args.round_number,
        openf1_session_key=args.openf1_session_key,
        cache=args.cache,
        output=args.output,
    )
    print(json.dumps({
        "passed": report["passed"],
        "verification_status": report.get("verification_status"),
        "hard_mismatch_count": report.get("hard_mismatch_count", 0),
        "warning_count": report.get("warning_count", 0),
        "insufficient_hard_count": report.get("insufficient_hard_count", 0),
        "provider_errors": report.get("provider_errors", {}),
        "output": str(args.output),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
