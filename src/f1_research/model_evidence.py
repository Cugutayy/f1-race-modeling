"""Build a compact, deployable model-evidence artifact from sealed benchmark output.

The live product must never hard-code benchmark claims. This module reads the
versioned benchmark report/selection/uncertainty files and emits a small JSON payload
that can be mounted read-only beside the API. Missing optional uncertainty evidence is
reported as unavailable rather than inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CORE_MODELS = (
    "rank_ensemble",
    "modern::catboost",
    "qualifying_order",
    "ridge_rank",
    "plackett_luce_mle",
)
CORE_METRICS = (
    "position_mae",
    "winner_log_loss",
    "winner_brier",
    "winner_accuracy",
    "podium_recall",
    "spearman_rank",
    "kendall_rank",
    "ndcg",
)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _clean_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _summary_by_model(report: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    rows = report.get("summary")
    if not isinstance(rows, list):
        raise ValueError("Benchmark report is missing summary rows")
    output: dict[str, dict[str, float | int | None]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        model = row["model"]
        output[model] = {
            metric: _clean_number(row.get(metric))
            for metric in CORE_METRICS
        }
    return output


def _paired_uncertainty(uncertainty: dict[str, Any] | None) -> dict[str, Any] | None:
    if uncertainty is None:
        return None
    baseline = uncertainty.get("baseline")
    paired = uncertainty.get("paired_vs_baseline")
    if not isinstance(baseline, str) or not isinstance(paired, dict):
        raise ValueError("Malformed benchmark uncertainty artifact")
    keep_metrics = {"winner_log_loss", "position_mae"}
    compact: dict[str, Any] = {}
    for model, metric_map in paired.items():
        if not isinstance(model, str) or not isinstance(metric_map, dict):
            continue
        selected: dict[str, Any] = {}
        for metric in keep_metrics:
            stats = metric_map.get(metric)
            if not isinstance(stats, dict):
                continue
            interval = stats.get("interval_95")
            selected[metric] = {
                "mean_difference_model_minus_baseline": _clean_number(
                    stats.get("mean_difference_model_minus_baseline")
                ),
                "interval_95": (
                    [_clean_number(item) for item in interval]
                    if isinstance(interval, list) and len(interval) == 2
                    else None
                ),
                "events": _clean_number(stats.get("events")),
                "model_better_events": _clean_number(stats.get("model_better_events")),
                "baseline_better_events": _clean_number(stats.get("baseline_better_events")),
                "ties": _clean_number(stats.get("ties")),
                "bootstrap_fraction_favorable": _clean_number(
                    stats.get("bootstrap_fraction_favorable")
                ),
                "direction": stats.get("direction"),
            }
        if selected:
            compact[model] = selected
    return {
        "baseline": baseline,
        "bootstrap_samples": _clean_number(uncertainty.get("bootstrap_samples")),
        "unit": uncertainty.get("unit"),
        "paired_vs_baseline": compact,
        "interpretation": uncertainty.get("interpretation"),
    }


def build_model_evidence(
    report: dict[str, Any],
    selection: dict[str, Any],
    uncertainty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(report.get("schema_version", 0)) < 2:
        raise ValueError("Unsupported benchmark report schema")
    split = selection.get("split")
    if not isinstance(split, dict) or not isinstance(split.get("test"), list):
        raise ValueError("Selection artifact is missing the sealed test split")

    all_models = _summary_by_model(report)
    preferred = [model for model in CORE_MODELS if model in all_models]
    for model in sorted(all_models):
        if model not in preferred:
            preferred.append(model)

    provenance = report.get("provenance") if isinstance(report.get("provenance"), dict) else {}
    years = provenance.get("years") if isinstance(provenance.get("years"), list) else None
    selected_modern = selection.get("selected_modern")
    if not isinstance(selected_modern, dict):
        selected_modern = None
    weights = selection.get("ensemble_weights")
    if not isinstance(weights, dict):
        weights = None

    return {
        "schema_version": 1,
        "evidence_kind": "retrospective_sealed_historical_benchmark",
        "benchmark_run_id": report.get("run_id"),
        "provider": provenance.get("provider"),
        "years": years,
        "protocol": selection.get("protocol"),
        "sealed_test_events": len(split["test"]),
        "sealed_test_event_ids": split["test"],
        "selected_modern": selected_modern,
        "ensemble_weights": {
            str(key): _clean_number(value) for key, value in weights.items()
        } if weights is not None else None,
        "winner_calibration": report.get("winner_calibration") if isinstance(report.get("winner_calibration"), list) else None,
        "models": [
            {"model": model, **all_models[model]}
            for model in preferred
        ],
        "uncertainty": _paired_uncertainty(uncertainty),
        "limitations": [
            "Retrospective sealed historical evidence is not a prospective performance guarantee.",
            "The sealed test block was not used for model, hyperparameter, ensemble-weight or temperature selection.",
            "Public F1 data omits private team telemetry, fuel load, setup and internal tyre-state variables.",
            "Event-bootstrap intervals describe uncertainty across the held-out races and are not an iid significance test.",
        ],
    }


def generate_model_evidence(benchmark_dir: Path, output: Path) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    report = _read_json(benchmark_dir / "report.json")
    selection = _read_json(benchmark_dir / "selection.json")
    uncertainty = _read_json(
        benchmark_dir / "evidence" / "uncertainty.json",
        required=False,
    )
    assert report is not None and selection is not None
    payload = build_model_evidence(report, selection, uncertainty)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = generate_model_evidence(args.benchmark_dir, args.output)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
