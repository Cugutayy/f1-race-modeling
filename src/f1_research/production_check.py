"""One fail-closed production readiness gate for deploy/release automation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model_registry import load_manifest, sha256_file
from .replay import load_jsonl, replay
from .revision import validate_git_sha


def _check_data_truth(
    path: Path,
    *,
    expected_git_sha: str | None = None,
) -> tuple[bool, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    events = report.get("events")
    if report.get("matrix_schema_version") != 1 or not isinstance(events, list) or not events:
        return False, "invalid/empty data-truth matrix"
    if len(events) < 12:
        return False, f"only {len(events)} real audited events; require at least 12"
    try:
        producer_git_sha = validate_git_sha(
            report.get("producer_git_sha"),
            field="data_truth.producer_git_sha",
        )
    except ValueError as exc:
        return False, str(exc)
    if expected_git_sha is not None:
        expected = validate_git_sha(expected_git_sha, field="expected_git_sha")
        if producer_git_sha != expected:
            return False, (
                f"data-truth evidence revision {producer_git_sha} does not match "
                f"release revision {expected}"
            )
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            return False, f"data-truth event {index} is not an object"
        if event.get("passed") is not True:
            return False, f"data-truth event {index} is not explicitly passing"
        if event.get("verification_status") not in {"PASS", "PASS_WITH_GAPS"}:
            return False, f"data-truth event {index} has invalid passing status"
        if event.get("event_identity_verified") is not True:
            return False, f"data-truth event {index} lacks verified event identity"
        if event.get("reconciliation_performed") is not True:
            return False, f"data-truth event {index} was not reconciled"
        if int(event.get("hard_mismatch_count", -1)) != 0:
            return False, f"data-truth event {index} contains hard mismatches"
        if int(event.get("insufficient_hard_count", -1)) != 0:
            return False, f"data-truth event {index} lacks complete hard-field evidence"
        if int(event.get("provider_error_count", -1)) != 0:
            return False, f"data-truth event {index} contains provider errors"
        source_sha = str(event.get("source_sha256") or "").lower()
        if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
            return False, f"data-truth event {index} lacks a valid source SHA-256"
        if event.get("producer_git_sha") != producer_git_sha:
            return False, f"data-truth event {index} producer revision disagrees with matrix"
    return True, f"{len(events)} audited events; reconciled with zero hard/provider failures"


def _check_benchmark(path: Path) -> tuple[bool, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("data_kind") != "historical":
        return False, "benchmark is not historical"
    if int(report.get("test_events", 0)) < 5:
        return False, "fewer than 5 sealed/held-out events"
    if not report.get("predictions") or not report.get("metrics"):
        return False, "benchmark has no predictions/metrics"
    if report.get("schema_version") != 2:
        return False, "benchmark must use sealed-evidence schema v2"
    if not report.get("winner_calibration"):
        return False, "benchmark lacks winner calibration/ECE evidence"
    audit = report.get("audit")
    if not isinstance(audit, dict) or audit.get("test_updates_model") is not False:
        return False, "benchmark does not prove sealed-test isolation"
    split = audit.get("split")
    if not isinstance(split, dict):
        return False, "benchmark split audit is missing"
    names = ("fit", "tuning", "calibration", "test")
    blocks = {name: set(split.get(name) or []) for name in names}
    if any(blocks[a] & blocks[b] for i, a in enumerate(names) for b in names[i + 1:]):
        return False, "benchmark split blocks overlap"
    if set(report.get("predictions", [{}])[0].keys()).isdisjoint({"event_id"}):
        return False, "benchmark predictions lack event identity"
    predicted_events = {str(row.get("event_id")) for row in report["predictions"]}
    if predicted_events != {str(value) for value in blocks["test"]}:
        return False, "benchmark predictions are not exactly the sealed test block"
    return True, f"{report['test_events']} sealed held-out events"


def run_checks(
    *,
    data_truth: Path,
    benchmark: Path,
    manifest: Path,
    model: Path,
    calibration: Path,
    feature_schema: Path,
    training_data: Path,
    replay_capture: Path | None = None,
    expected_git_sha: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    try:
        passed, detail = _check_data_truth(
            data_truth,
            expected_git_sha=expected_git_sha,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    checks["data_truth"] = {"passed": passed, "detail": detail}
    try:
        passed, detail = _check_benchmark(benchmark)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    checks["benchmark"] = {"passed": passed, "detail": detail}
    try:
        loaded = load_manifest(manifest, model_path=model)
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if benchmark_payload.get("run_id") != loaded.benchmark_run_id:
            raise ValueError("model manifest benchmark run does not match benchmark artifact")
        if expected_git_sha is not None:
            expected = validate_git_sha(expected_git_sha, field="expected_git_sha")
            if loaded.git_sha != expected:
                raise ValueError(
                    f"model manifest revision {loaded.git_sha} does not match release revision {expected}"
                )
        if sha256_file(calibration) != loaded.calibration_sha256:
            raise ValueError("calibration artifact SHA-256 does not match manifest")
        if sha256_file(feature_schema) != loaded.feature_schema_sha256:
            raise ValueError("feature schema SHA-256 does not match manifest")
        if sha256_file(training_data) != loaded.training_data_sha256:
            raise ValueError("training data SHA-256 does not match manifest")
        checks["model_integrity"] = {"passed": True, "detail": loaded.model_id}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        checks["model_integrity"] = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    if replay_capture is None:
        checks["replay"] = {"passed": False, "detail": "real replay capture is required"}
    else:
        try:
            result = replay(load_jsonl(replay_capture), snapshot_every=100)
            required_topics = {"position", "laps", "weather", "race_control"}
            missing_topics = sorted(required_topics - set(result["topic_counts"]))
            replay_passed = result["accepted_count"] > 0 and not missing_topics
            detail = f"{result['accepted_count']}/{result['event_count']} accepted"
            if missing_topics:
                detail += f"; missing topics: {', '.join(missing_topics)}"
            checks["replay"] = {"passed": replay_passed, "detail": detail}
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
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--replay-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-git-sha")
    args = parser.parse_args(argv)
    result = run_checks(
        data_truth=args.data_truth,
        benchmark=args.benchmark,
        manifest=args.manifest,
        model=args.model,
        calibration=args.calibration,
        feature_schema=args.feature_schema,
        training_data=args.training_data,
        replay_capture=args.replay_capture,
        expected_git_sha=args.expected_git_sha,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["production_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
