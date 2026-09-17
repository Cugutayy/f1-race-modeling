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
from .lap_pipeline import collect_recent
from .lap_strict import STRICT_FEATURES, fit_strict_mixture
from .strategy_calibration import calibrate_strategy_priors, save_strategy_priors


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
    if count < 4:
        raise ValueError("At least four completed races are required")
    output = Path(output)
    datasets, manifests = collect_recent(
        year,
        count,
        output,
        latency_s=latency_s,
        refresh=refresh,
    )
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

    result = {
        "schema_version": 3,
        "task": "next_lap_strict_mixture",
        "created_at": datetime.now(UTC).isoformat(),
        "year": year,
        "sessions": session_keys,
        "selected_regressor": artifact["selected_regressor"],
        "feature_policy": "strict_asof_only",
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
        "strategy_priors": report["strategy_priors"]["priors"],
    }, indent=2))


if __name__ == "__main__":
    main()
