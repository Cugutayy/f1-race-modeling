"""Runtime contract/freshness checks for the public OpenF1 provider.

This is an operational diagnostic, not a model benchmark. It verifies that the
provider endpoints required by the project are reachable and still expose the fields
our adapters expect. Optional car-data inspection also reports observed timestamp
spacing without claiming an SLA.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .openf1_live import OpenF1Client

CONTRACTS: dict[str, tuple[str, ...]] = {
    "sessions": ("session_key", "session_name", "date_start"),
    "drivers": ("driver_number", "name_acronym", "team_name"),
    "position": ("driver_number", "position", "date"),
    "intervals": ("driver_number", "date"),
    "laps": ("driver_number", "lap_number", "date_start"),
    "stints": ("driver_number", "stint_number", "lap_start", "compound"),
    "pit": ("driver_number", "lap_number"),
    "weather": ("date", "air_temperature", "track_temperature"),
    "race_control": ("date", "category"),
}


def _timestamp(row: dict[str, Any]) -> pd.Timestamp | None:
    for field in ("date", "date_start", "date_end"):
        value = pd.to_datetime(row.get(field), utc=True, errors="coerce")
        if not pd.isna(value):
            return value
    return None


def _age_s(timestamp: pd.Timestamp | None, now: datetime) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (pd.Timestamp(now) - timestamp).total_seconds())


def _endpoint_report(
    endpoint: str,
    rows: list[dict[str, Any]],
    required: tuple[str, ...],
    now: datetime,
) -> dict[str, Any]:
    coverage = {}
    for field in required:
        present = sum(int(field in row and row.get(field) is not None) for row in rows)
        coverage[field] = {
            "present": present,
            "fraction": float(present / len(rows)) if rows else 0.0,
        }
    timestamps = [stamp for row in rows if (stamp := _timestamp(row)) is not None]
    latest = max(timestamps) if timestamps else None
    missing_required = [
        field for field, stats in coverage.items()
        if rows and stats["fraction"] < 0.95
    ]
    return {
        "endpoint": endpoint,
        "rows": len(rows),
        "required_field_coverage": coverage,
        "missing_or_sparse_required_fields": missing_required,
        "latest_provider_timestamp": latest.isoformat() if latest is not None else None,
        "latest_provider_age_s": _age_s(latest, now),
        "ok": bool(rows) and not missing_required,
    }


def _telemetry_report(rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    dates = pd.to_datetime(
        [row.get("date") for row in rows if row.get("date") is not None],
        utc=True,
        errors="coerce",
    )
    dates = pd.Series(dates).dropna().sort_values()
    cadence_ms = None
    p90_ms = None
    if len(dates) >= 3:
        tail = dates.iloc[-500:]
        delta_ms = tail.diff().dt.total_seconds().dropna().to_numpy(dtype=float) * 1000.0
        positive = delta_ms[np.isfinite(delta_ms) & (delta_ms > 0)]
        if len(positive):
            cadence_ms = float(np.median(positive))
            p90_ms = float(np.quantile(positive, 0.90))
    latest = dates.iloc[-1] if len(dates) else None
    fields = ("speed", "throttle", "brake", "rpm", "n_gear", "drs")
    coverage = {
        field: float(sum(row.get(field) is not None for row in rows) / len(rows)) if rows else 0.0
        for field in fields
    }
    return {
        "rows": len(rows),
        "latest_provider_timestamp": latest.isoformat() if latest is not None else None,
        "latest_provider_age_s": _age_s(latest, now),
        "median_observed_spacing_ms": cadence_ms,
        "p90_observed_spacing_ms": p90_ms,
        "field_coverage": coverage,
        "note": "Observed spacing from returned samples; not a provider SLA or guaranteed live latency.",
    }


def run_doctor(
    client: OpenF1Client,
    session_key: int | str = "latest",
    *,
    include_telemetry: bool = False,
    telemetry_driver: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": 1,
        "provider": "OpenF1",
        "requested_session_key": session_key,
        "checked_at": now.isoformat(),
        "authenticated": bool(client.token),
        "endpoints": {},
        "errors": [],
    }
    resolved_session: int | str = session_key
    first_driver: int | None = telemetry_driver

    for endpoint, required in CONTRACTS.items():
        try:
            rows = client.get(endpoint, session_key=resolved_session)
            endpoint_report = _endpoint_report(endpoint, rows, required, now)
            report["endpoints"][endpoint] = endpoint_report
            if endpoint == "sessions" and rows:
                key = rows[-1].get("session_key")
                if isinstance(key, (int, float)) and float(key).is_integer():
                    resolved_session = int(key)
                    report["resolved_session_key"] = int(key)
            if endpoint == "drivers" and first_driver is None:
                for row in rows:
                    try:
                        first_driver = int(row.get("driver_number"))
                        break
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:  # provider/network errors are diagnostic output
            report["errors"].append({"endpoint": endpoint, "error": str(exc)})
            report["endpoints"][endpoint] = {"endpoint": endpoint, "ok": False, "error": str(exc)}

    if include_telemetry and first_driver is not None:
        try:
            telemetry = client.get(
                "car_data",
                session_key=resolved_session,
                driver_number=first_driver,
            )
            report["telemetry_driver"] = first_driver
            report["telemetry"] = _telemetry_report(telemetry, now)
        except Exception as exc:
            report["errors"].append({"endpoint": "car_data", "error": str(exc)})
            report["telemetry"] = {"rows": 0, "error": str(exc)}

    critical = ["sessions", "drivers", "position", "laps"]
    report["ok"] = not report["errors"] and all(
        bool(report["endpoints"].get(endpoint, {}).get("ok")) for endpoint in critical
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-key", default="latest")
    parser.add_argument("--include-telemetry", action="store_true")
    parser.add_argument("--telemetry-driver", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    session_key: int | str = int(args.session_key) if str(args.session_key).isdigit() else args.session_key
    report = run_doctor(
        OpenF1Client.from_env(),
        session_key,
        include_telemetry=args.include_telemetry,
        telemetry_driver=args.telemetry_driver,
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
