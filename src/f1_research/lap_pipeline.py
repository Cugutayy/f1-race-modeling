"""Collect OpenF1 race-state features and train a reproducible next-lap artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .lap_intelligence import (
    FEATURES,
    LapModelSpec,
    benchmark_lap_models,
    build_lap_dataset,
    build_lap_estimator,
)
from .openf1_live import OpenF1Client

ENDPOINTS = ("laps", "stints", "weather", "pit", "race_control")


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def select_completed_races(client: OpenF1Client, year: int, count: int,
                           now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    sessions = client.get("sessions", year=year, session_name="Race")
    completed = []
    for session in sessions:
        end = pd.to_datetime(session.get("date_end"), utc=True, errors="coerce")
        if pd.isna(end) or end >= pd.Timestamp(now):
            continue
        if session.get("is_cancelled", False):
            continue
        completed.append(session)
    completed.sort(key=lambda row: pd.Timestamp(row["date_start"]))
    if len(completed) < count:
        raise ValueError(f"Only {len(completed)} completed {year} races available; requested {count}")
    return completed[-count:]


def collect_session(client: OpenF1Client, session: dict[str, Any], output: Path,
                    *, latency_s: float = 1.0, refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    session_key = int(session["session_key"])
    raw_dir = Path(output) / "raw" / str(session_key)
    sources: dict[str, Any] = {}
    data: dict[str, list[dict[str, Any]]] = {}
    for endpoint in ENDPOINTS:
        path = raw_dir / f"{endpoint}.json"
        if path.exists() and not refresh:
            raw = path.read_bytes()
            rows = json.loads(raw)
            if not isinstance(rows, list):
                raise ValueError(f"Invalid cached {endpoint} for session {session_key}")
            sources[endpoint] = {"path": str(path), "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest(), "cached": True}
        else:
            rows = client.get(endpoint, session_key=session_key)
            sources[endpoint] = {**_write_json(path, rows), "cached": False,
                                 "retrieved_at": datetime.now(UTC).isoformat()}
        data[endpoint] = rows
    dataset = build_lap_dataset(
        data["laps"], stint_rows=data["stints"], weather_rows=data["weather"],
        pit_rows=data["pit"], race_control_rows=data["race_control"], latency_s=latency_s,
    )
    if dataset.empty:
        raise ValueError(f"Session {session_key} produced no next-lap training rows")
    dataset_path = Path(output) / "datasets" / f"{session_key}.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)
    manifest = {
        "session_key": session_key,
        "meeting_key": session.get("meeting_key"),
        "country_name": session.get("country_name"),
        "location": session.get("location"),
        "date_start": session.get("date_start"),
        "date_end": session.get("date_end"),
        "latency_assumption_s": latency_s,
        "rows": len(dataset),
        "valid_targets": int(dataset.target_valid.sum()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "sources": sources,
        "limitations": [
            "OpenF1 date_start is approximate and target availability adds an explicit simulated latency.",
            "Historical stint rows have no publication timestamp, so compound/tyre-age are retrospective features.",
            "Public timing data is not equivalent to team telemetry or tyre/fuel/setup data.",
        ],
    }
    _write_json(Path(output) / "manifests" / f"{session_key}.json", manifest)
    return dataset, manifest


def collect_recent(year: int, count: int, output: Path, *, latency_s: float = 1.0,
                   refresh: bool = False) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    client = OpenF1Client.from_env()
    sessions = select_completed_races(client, year, count)
    datasets, manifests = [], []
    for session in sessions:
        dataset, manifest = collect_session(client, session, output, latency_s=latency_s, refresh=refresh)
        datasets.append(dataset)
        manifests.append(manifest)
    return datasets, manifests


def _available_specs(include_foundation: bool) -> tuple[LapModelSpec, ...]:
    specs = [
        LapModelSpec("hist_gradient_boosting", {}),
        LapModelSpec("extra_trees", {}),
    ]
    if include_foundation:
        specs.append(LapModelSpec("tabicl_v2", {}))
    return tuple(specs)


def train_selected(datasets: list[pd.DataFrame], selected_name: str,
                   specs: tuple[LapModelSpec, ...]) -> Any:
    spec = next((candidate for candidate in specs if candidate.name == selected_name), None)
    if spec is None:
        raise ValueError(f"Selected model spec not found: {selected_name}")
    train = pd.concat(datasets[:-1], ignore_index=True)
    train = train[train.target_valid].copy()
    if train.empty:
        raise ValueError("No valid rows for final next-lap artifact")
    return build_lap_estimator(spec).fit(train[FEATURES], train.target_s)


def run(year: int, count: int, output: Path, *, latency_s: float = 1.0,
        include_foundation: bool = False, refresh: bool = False) -> dict[str, Any]:
    if count < 3:
        raise ValueError("At least three completed races are required")
    output = Path(output)
    datasets, manifests = collect_recent(year, count, output, latency_s=latency_s, refresh=refresh)
    specs = _available_specs(include_foundation)
    metrics, audit = benchmark_lap_models(datasets, specs=specs)
    selected = str(audit["selected_model"])
    artifact = train_selected(datasets, selected, specs)
    artifact_path = output / "next_lap_model.joblib"
    joblib.dump({
        "schema_version": 1,
        "task": "next_lap_duration",
        "model_name": selected,
        "features": FEATURES,
        "trained_through_session": int(datasets[-2].session_key.iloc[0]),
        "sealed_test_session": int(datasets[-1].session_key.iloc[0]),
        "pipeline": artifact,
    }, artifact_path)
    result = {
        "schema_version": 1,
        "task": "next_lap_duration",
        "year": year,
        "sessions": [manifest["session_key"] for manifest in manifests],
        "selected_model": selected,
        "summary": metrics.to_dict("records"),
        "audit": audit,
        "artifact": {"path": str(artifact_path),
                     "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest()},
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "lap_model_report.json", result)
    metrics.to_csv(output / "lap_model_summary.csv", index=False)
    return result


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("task") != "next_lap_duration":
        raise ValueError("Not a next-lap model artifact")
    if artifact.get("features") != FEATURES:
        raise ValueError("Next-lap feature schema mismatch")
    if not hasattr(artifact.get("pipeline"), "predict"):
        raise ValueError("Next-lap artifact has no predictive pipeline")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race-count", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("reports/local/lap-intelligence"))
    parser.add_argument("--latency-s", type=float, default=1.0)
    parser.add_argument("--foundation", action="store_true", help="Also evaluate local TabICLv2")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.year, args.race_count, args.output, latency_s=args.latency_s,
                 include_foundation=args.foundation, refresh=args.refresh)
    print(json.dumps({"selected_model": report["selected_model"],
                      "sessions": report["sessions"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
