"""Build a reproducible release bundle from sealed benchmark evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .benchmark_evidence import save_uncertainty
from .data import validate
from .evaluation_v2 import benchmark_v2, save_v2_report
from .features import FEATURES, build_features
from .model_registry import ModelManifest, sha256_file, write_manifest
from .modern_models import CandidateSpec, fit_selected


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _validated_provenance(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Release provenance must be a JSON object")
    provider = value.get("provider")
    years = value.get("years")
    requests = value.get("requests")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Release provenance is missing provider")
    if not isinstance(years, list) or not years or any(
        not isinstance(year, int) or year < 1950 or year > 2100 for year in years
    ):
        raise ValueError("Release provenance is missing valid source years")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Release provenance is missing provider request evidence")
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"Release provenance request {index} is not an object")
        url = request.get("url")
        retrieved_at = request.get("retrieved_at")
        digest = request.get("sha256")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"Release provenance request {index} has invalid URL")
        if not isinstance(retrieved_at, str) or not retrieved_at.strip():
            raise ValueError(f"Release provenance request {index} has no retrieval time")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"Release provenance request {index} has invalid SHA-256")
    return dict(value)


def _input_provenance(path: Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Sealed release requires source provenance sidecar: {sidecar}"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance = _validated_provenance(payload)
    provenance["source_csv_name"] = path.name
    provenance["source_csv_sha256"] = sha256_file(path)
    provenance["provenance_sidecar_sha256"] = sha256_file(sidecar)
    return provenance


def _provenance_digest(provenance: dict[str, Any]) -> str:
    canonical = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_release(
    frame: pd.DataFrame,
    output: Path,
    *,
    provenance: dict[str, Any],
    test_events: int = 12,
    tuning_events: int = 6,
    calibration_events: int = 4,
    min_fit_events: int = 20,
    modern_names: tuple[str, ...] = ("hist_gradient_boosting", "extra_trees"),
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clean = validate(frame)
    source_provenance = _validated_provenance(provenance)
    metrics, predictions, audit = benchmark_v2(
        clean,
        test_events=test_events,
        tuning_events=tuning_events,
        calibration_events=calibration_events,
        min_fit_events=min_fit_events,
        modern_names=modern_names,
    )
    benchmark_dir = output / "benchmark"
    report = save_v2_report(
        clean,
        metrics,
        predictions,
        audit,
        benchmark_dir,
        provenance=source_provenance,
    )
    uncertainty_dir = benchmark_dir / "evidence"
    uncertainty = save_uncertainty(metrics, uncertainty_dir, baseline="qualifying_order", samples=10000, seed=42)

    selected = audit["selected_modern"]
    spec = CandidateSpec(selected["name"], selected["params"])
    features = build_features(clean)
    # Keep release model bytes identical in training scope to the model evaluated on the sealed test.
    # Calibration events tune probability temperature only; fitting on them here would create an
    # unevaluated model artifact and falsely attach sealed-test evidence to different model bytes.
    train_ids = audit["split"]["fit"] + audit["split"]["tuning"]
    train = features[features.event_id.isin(train_ids)].copy()
    model = fit_selected(train, spec)
    model_path = output / "model.joblib"
    joblib.dump(model, model_path)

    feature_schema_path = output / "feature_schema.json"
    feature_schema_path.write_text(
        json.dumps(FEATURES, separators=(",", ":"), sort_keys=False),
        encoding="utf-8",
    )
    training_data_path = output / "training_features.csv"
    training_snapshot = train.sort_values(["date", "event_id", "driver"])
    training_data_path.write_text(training_snapshot.to_csv(index=False), encoding="utf-8")
    feature_schema_sha = sha256_file(feature_schema_path)
    training_data_sha = sha256_file(training_data_path)
    calibration_payload = {
        "schema_version": 1,
        "temperatures": audit["temperatures"],
        "selected_model": f"modern::{spec.name}",
    }
    calibration_path = output / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    calibration_sha = sha256_file(calibration_path)
    trained_until = pd.to_datetime(train["date"], utc=True).max().isoformat()
    manifest = ModelManifest(
        schema_version=1,
        model_id=f"modern::{spec.name}",
        model_sha256=sha256_file(model_path),
        feature_schema_sha256=feature_schema_sha,
        training_data_sha256=training_data_sha,
        calibration_sha256=calibration_sha,
        trained_until=trained_until,
        benchmark_run_id=report["run_id"],
        git_sha=_git_sha(),
    )
    manifest_path = output / "model_manifest.json"
    write_manifest(manifest_path, manifest)
    release = {
        "schema_version": 1,
        "benchmark": str(benchmark_dir / "report.json"),
        "model": str(model_path),
        "manifest": str(manifest_path),
        "calibration": str(calibration_path),
        "feature_schema": str(feature_schema_path),
        "training_data": str(training_data_path),
        "model_id": manifest.model_id,
        "sealed_test_events": report["test_events"],
        "model_training_blocks": ["fit", "tuning"],
        "calibration_used_for_model_fit": False,
        "run_id": report["run_id"],
        "uncertainty": str(uncertainty_dir / "uncertainty.json"),
        "uncertainty_unit": uncertainty["unit"],
        "uncertainty_bootstrap_samples": uncertainty["bootstrap_samples"],
        "source_provenance_sha256": _provenance_digest(source_provenance),
        "source_provider": source_provenance["provider"],
        "source_years": source_provenance["years"],
    }
    (output / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    return release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build sealed benchmark + model release bundle")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-events", type=int, default=12)
    parser.add_argument("--tuning-events", type=int, default=6)
    parser.add_argument("--calibration-events", type=int, default=4)
    parser.add_argument("--min-fit-events", type=int, default=20)
    parser.add_argument("--modern-models", default="hist_gradient_boosting,extra_trees")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input)
    provenance = _input_provenance(args.input)
    models = tuple(x.strip() for x in args.modern_models.split(",") if x.strip())
    release = build_release(
        frame,
        args.output,
        provenance=provenance,
        test_events=args.test_events,
        tuning_events=args.tuning_events, calibration_events=args.calibration_events,
        min_fit_events=args.min_fit_events, modern_names=models,
    )
    print(json.dumps(release, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
