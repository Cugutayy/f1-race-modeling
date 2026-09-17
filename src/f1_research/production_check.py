"""One fail-closed production readiness gate for deploy/release automation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model_registry import load_manifest
from .replay import load_jsonl, replay


def _check_data_truth(path: Path) -> tuple[bool, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    events = report.get("events")
    if report.get("matrix_schema_version") != 1 or not isinstance(events, list) or not events:
        return False, "invalid/empty data-truth matrix"
    if len(events) < 12:
        return False, f"only {len(events)} real audited events; require at least 12"
    failures = [e for e in events if e.get("verification_status") == "FAIL"]
    if failures:
        return False, f"{len(failures)} audited events failed"
    return True, f"{len(events)} audited events; no FAIL status"


def _check_benchmark(path: Path) -> tuple[bool, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("data_kind") != "historical":
        return False, "benchmark is not historical"
    if int(report.get("test_events", 0)) < 5:
        return False, "fewer than 5 sealed/held-out events"
    if not report.get("predictions") or not report.get("metrics"):
        return False, "benchmark has no predictions/metrics"
    return True, f"{report['test_events']} held-out events"


def run_checks(*, data_truth: Path, benchmark: Path, manifest: Path, model: Path, replay_capture: Path | None = None) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for name, fn, path in (
        ("data_truth", _check_data_truth, data_truth),
        ("benchmark", _check_benchmark, benchmark),
    ):
        try:
            passed, detail = fn(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        checks[name] = {"passed": passed, "detail": detail}
    try:
        loaded = load_manifest(manifest, model_path=model)
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if benchmark_payload.get("run_id") != loaded.benchmark_run_id:
            raise ValueError("model manifest benchmark run does not match benchmark artifact")
        checks["model_integrity"] = {"passed": True, "detail": loaded.model_id}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        checks["model_integrity"] = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    if replay_capture is None:
        checks["replay"] = {"passed": False, "detail": "real replay capture is required"}
    else:
        try:
            result = replay(load_jsonl(replay_capture), snapshot_every=100)
            checks["replay"] = {"passed": result["accepted_count"] > 0, "detail": f"{result['accepted_count']}/{result['event_count']} accepted"}
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            checks["replay"] = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    passed = all(item["passed"] for item in checks.values())
    return {"schema_version": 1, "production_ready": passed, "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-truth", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_checks(data_truth=args.data_truth, benchmark=args.benchmark,
                        manifest=args.manifest, model=args.model, replay_capture=args.replay_capture)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["production_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
