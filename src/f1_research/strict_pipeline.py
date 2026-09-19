"""Train the evidence-grade next-lap artifact without retrospective stint joins."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .lap_intelligence import LapModelSpec
from .lap_pipeline import ENDPOINTS, collect_recent
from .lap_strict import STRICT_FEATURES, fit_strict_mixture
from .revision import current_git_sha
from .strategy_calibration import calibrate_strategy_priors, save_strategy_priors


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validated_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return digest


def _source_evidence(manifests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not manifests:
        raise ValueError("Strict release requires source manifests")
    sessions: list[dict[str, Any]] = []
    content_index: list[dict[str, Any]] = []
    seen: set[int] = set()
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError("Strict source manifest must be an object")
        session_key = int(manifest.get("session_key") or 0)
        if session_key <= 0 or session_key in seen:
            raise ValueError(f"Invalid or duplicate source session_key: {session_key}")
        seen.add(session_key)
        dataset_sha256 = _validated_sha256(
            manifest.get("dataset_sha256"),
            field=f"session {session_key} dataset_sha256",
        )
        rows = manifest.get("rows")
        valid_targets = manifest.get("valid_targets")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"session {session_key} has invalid row count")
        if not isinstance(valid_targets, int) or not 0 <= valid_targets <= rows:
            raise ValueError(f"session {session_key} has invalid valid-target count")
        raw_sources = manifest.get("sources")
        if not isinstance(raw_sources, dict):
            raise ValueError(f"session {session_key} has no source map")
        missing = set(ENDPOINTS) - set(raw_sources)
        if missing:
            raise ValueError(
                f"session {session_key} is missing source evidence: {sorted(missing)}"
            )

        sources: dict[str, dict[str, Any]] = {}
        content_sources: dict[str, dict[str, Any]] = {}
        for endpoint in ENDPOINTS:
            source = raw_sources.get(endpoint)
            if not isinstance(source, dict):
                raise ValueError(f"session {session_key} source {endpoint} is not an object")
            digest = _validated_sha256(
                source.get("sha256"),
                field=f"session {session_key} source {endpoint} sha256",
            )
            byte_count = source.get("bytes")
            if not isinstance(byte_count, int) or byte_count < 0:
                raise ValueError(f"session {session_key} source {endpoint} has invalid byte count")
            retrieved_at = source.get("retrieved_at")
            sources[endpoint] = {
                "sha256": digest,
                "bytes": byte_count,
                "retrieved_at": retrieved_at if isinstance(retrieved_at, str) else None,
                "cached": bool(source.get("cached", False)),
            }
            content_sources[endpoint] = {
                "sha256": digest,
                "bytes": byte_count,
            }

        content = {
            "session_key": session_key,
            "dataset_sha256": dataset_sha256,
            "rows": rows,
            "valid_targets": valid_targets,
            "sources": content_sources,
        }
        source_content_sha256 = _canonical_sha256(content)
        sessions.append({
            "session_key": session_key,
            "meeting_key": manifest.get("meeting_key"),
            "country_name": manifest.get("country_name"),
            "location": manifest.get("location"),
            "date_start": manifest.get("date_start"),
            "date_end": manifest.get("date_end"),
            "dataset_sha256": dataset_sha256,
            "rows": rows,
            "valid_targets": valid_targets,
            "sources": sources,
            "source_content_sha256": source_content_sha256,
        })
        content_index.append({
            "session_key": session_key,
            "source_content_sha256": source_content_sha256,
        })
    return sessions, _canonical_sha256(content_index)


def _specs(include_foundation: bool) -> tuple[LapModelSpec, ...]:
    values = [
        LapModelSpec("hist_gradient_boosting", {}),
        LapModelSpec("extra_trees", {}),
    ]
    if include_foundation:
        values.append(LapModelSpec("tabicl_v2", {}))
    return tuple(values)


def run(
    year: int,
    count: int,
    output: Path,
    *,
    latency_s: float = 1.0,
    include_foundation: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    if count < 6:
        raise ValueError(
            "At least six completed races are required: fit, tuning, "
            "three-event conformal calibration and sealed test"
        )
    output = Path(output)
    datasets, manifests = collect_recent(
        year,
        count,
        output,
        latency_s=latency_s,
        refresh=refresh,
    )
    source_sessions, source_evidence_sha256 = _source_evidence(manifests)
    artifact, metrics, audit = fit_strict_mixture(
        datasets,
        specs=_specs(include_foundation),
        alpha=0.10,
    )
    artifact_path = output / "next_lap_strict.joblib"
    joblib.dump(artifact, artifact_path)
    metrics_path = output / "strict_summary.csv"
    metrics.to_csv(metrics_path, index=False)

    session_keys = [int(item["session_key"]) for item in manifests]
    priors, prior_audit = calibrate_strategy_priors(datasets, output / "raw", session_keys)
    priors_path = output / "strategy_priors.json"
    priors_payload = save_strategy_priors(priors, prior_audit, priors_path)
    release_manifest_path = output / "strict_release_manifest.json"

    result = {
        "schema_version": 4,
        "task": "next_lap_strict_mixture",
        "artifact_schema_version": artifact["schema_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "year": year,
        "sessions": session_keys,
        "selected_regressor": artifact["selected_regressor"],
        "pace_prediction_mode": artifact["pace_prediction_mode"],
        "baseline_guard": artifact["baseline_guard"],
        "feature_policy": "strict_asof_only",
        "validation_scope": "retrospective_historical_posthoc_diagnostic",
        "prospective_validation": False,
        "features": STRICT_FEATURES,
        "metrics": metrics.to_dict("records"),
        "audit": audit,
        "strategy_priors": priors_payload,
        "artifacts": {
            "next_lap_strict": {
                "path": str(artifact_path),
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            },
            "strategy_priors": {
                "path": str(priors_path),
                "sha256": hashlib.sha256(priors_path.read_bytes()).hexdigest(),
            },
        },
        "release_manifest": str(release_manifest_path),
        "limitations": [
            "Historical OpenF1 is a retrospective source snapshot; provider publication latency is simulated.",
            "Compound, tyre age and stint number are excluded because historical stint publication time is unavailable.",
            "Public timing data is not equivalent to team fuel, setup, tyre-temperature or full sensor data.",
            "Foundation models are challengers and are selected only when validation MAE wins.",
            "Strategy priors are pooled public-data estimates, not team-specific engineering forecasts.",
        ],
    }
    report_path = output / "strict_report.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    calibration_sessions = artifact.get("calibration_sessions")
    conformal_radii = artifact.get("conformal_radii_s")
    if not isinstance(calibration_sessions, list) or len(calibration_sessions) < 2:
        raise ValueError("Strict artifact is missing multi-event calibration evidence")
    if not isinstance(conformal_radii, dict) or set(conformal_radii) != {
        "0.50",
        "0.80",
        "0.90",
        "0.95",
    }:
        raise ValueError("Strict artifact is missing required conformal coverage levels")

    release_manifest = {
        "schema_version": 1,
        "evidence_kind": "strict_live_pace_release",
        "git_sha": current_git_sha(required=True),
        "created_at": result["created_at"],
        "year": year,
        "sessions": session_keys,
        "task": result["task"],
        "selected_regressor": artifact["selected_regressor"],
        "artifact_schema_version": artifact["schema_version"],
        "pace_prediction_mode": artifact["pace_prediction_mode"],
        "baseline_guard": artifact["baseline_guard"],
        "feature_policy": result["feature_policy"],
        "validation_scope": result["validation_scope"],
        "prospective_validation": result["prospective_validation"],
        "feature_schema_sha256": _canonical_sha256(STRICT_FEATURES),
        "model_sha256": result["artifacts"]["next_lap_strict"]["sha256"],
        "strategy_priors_sha256": result["artifacts"]["strategy_priors"]["sha256"],
        "strict_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_evidence_sha256": source_evidence_sha256,
        "source_sessions": source_sessions,
        "calibration_sessions": [int(value) for value in calibration_sessions],
        "sealed_test_session": int(artifact["sealed_test_session"]),
        "conformal_nominal_coverage": artifact.get("conformal_nominal_coverage"),
        "conformal_radii_s": conformal_radii,
        "retrospective_stint_features_used": bool(
            artifact.get("retrospective_stint_features_used", True)
        ),
    }
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race-count", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("reports/local/lap-strict"))
    parser.add_argument("--latency-s", type=float, default=1.0)
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        args.year,
        args.race_count,
        args.output,
        latency_s=args.latency_s,
        include_foundation=args.foundation,
        refresh=args.refresh,
    )
    print(json.dumps({
        "selected_regressor": report["selected_regressor"],
        "sessions": report["sessions"],
        "metrics": report["metrics"],
        "artifacts": report["artifacts"],
        "release_manifest": report["release_manifest"],
        "strategy_priors": report["strategy_priors"]["priors"],
    }, indent=2))


if __name__ == "__main__":
    main()
