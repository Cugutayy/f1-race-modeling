"""Descriptive event-level uncertainty for sealed benchmark outputs.

The benchmark unit is a race, not an individual driver row. Resampling whole events
preserves the paired comparison between models that saw the same race. Intervals are
descriptive bootstrap intervals; they are not a claim of independence or a formal
prospective significance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_DIRECTION = {
    "position_mae": "lower",
    "winner_log_loss": "lower",
    "winner_brier": "lower",
    "winner_accuracy": "higher",
    "podium_recall": "higher",
}


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indices].mean(axis=1)


def benchmark_uncertainty(
    metrics: pd.DataFrame,
    *,
    baseline: str = "qualifying_order",
    samples: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    required = {"event_id", "model"} | set(METRIC_DIRECTION)
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    duplicates = metrics.duplicated(["event_id", "model"])
    if duplicates.any():
        raise ValueError("Expected exactly one metric row per event/model")
    if baseline not in set(metrics["model"]):
        raise ValueError(f"Baseline model is absent: {baseline}")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    model_intervals: dict[str, dict] = {}
    paired: dict[str, dict] = {}

    for model, group in metrics.groupby("model", sort=True):
        model_intervals[model] = {}
        for metric, direction in METRIC_DIRECTION.items():
            values = group[metric].to_numpy(dtype=float)
            draws = _bootstrap_mean(values, rng, samples)
            model_intervals[model][metric] = {
                "mean": float(values.mean()),
                "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "events": int(len(values)),
                "direction": direction,
            }

    baseline_frame = metrics[metrics.model == baseline].set_index("event_id")
    for model in sorted(set(metrics.model) - {baseline}):
        challenger = metrics[metrics.model == model].set_index("event_id")
        shared = baseline_frame.index.intersection(challenger.index)
        if len(shared) == 0:
            continue
        paired[model] = {}
        for metric, direction in METRIC_DIRECTION.items():
            differences = (
                challenger.loc[shared, metric].to_numpy(dtype=float)
                - baseline_frame.loc[shared, metric].to_numpy(dtype=float)
            )
            draws = _bootstrap_mean(differences, rng, samples)
            if direction == "lower":
                better = differences < 0
                worse = differences > 0
                favorable_draw = draws < 0
            else:
                better = differences > 0
                worse = differences < 0
                favorable_draw = draws > 0
            tied = np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)
            stats = {
                "mean_difference_model_minus_baseline": float(differences.mean()),
                "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "events": int(len(differences)),
                "model_better_events": int(np.count_nonzero(better)),
                "baseline_better_events": int(np.count_nonzero(worse)),
                "ties": int(np.count_nonzero(tied)),
                "bootstrap_fraction_favorable": float(np.mean(favorable_draw)),
                "direction": direction,
            }
            paired[model][metric] = stats
            rows.append({"model": model, "baseline": baseline, "metric": metric, **stats})

    report = {
        "schema_version": 1,
        "unit": "whole race event",
        "bootstrap_samples": samples,
        "seed": seed,
        "baseline": baseline,
        "model_intervals": model_intervals,
        "paired_vs_baseline": paired,
        "interpretation": (
            "Paired differences are model minus baseline on the same held-out races. "
            "Negative favors the model for lower-is-better metrics; positive favors the model "
            "for higher-is-better metrics. Intervals are descriptive event bootstrap intervals."
        ),
    }
    return pd.DataFrame(rows), report


def save_uncertainty(
    metrics: pd.DataFrame,
    output: Path,
    *,
    baseline: str = "qualifying_order",
    samples: int = 10000,
    seed: int = 42,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comparisons, report = benchmark_uncertainty(
        metrics,
        baseline=baseline,
        samples=samples,
        seed=seed,
    )
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    (output / "uncertainty.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="qualifying_order")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    metrics = pd.read_csv(args.metrics)
    report = save_uncertainty(
        metrics,
        args.output,
        baseline=args.baseline,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(report["paired_vs_baseline"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
