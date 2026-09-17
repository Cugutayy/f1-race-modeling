"""Event-level uncertainty for sealed benchmark outputs.

The benchmark unit is a race, not an individual driver row. Ordinary event bootstrap
preserves paired model comparison; circular moving-block bootstrap adds a sensitivity
check for short-range chronological dependence. Paired sign-flip randomization provides
a distribution-free paired null test. None of these diagnostics are used to select or
refit a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistical_diagnostics import moving_block_bootstrap_mean, paired_randomization_test

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


def _event_order(metrics: pd.DataFrame) -> list[str]:
    columns = ["event_id"] + (["date"] if "date" in metrics.columns else [])
    events = metrics[columns].drop_duplicates("event_id")
    if "date" in events:
        events = events.assign(_date=pd.to_datetime(events["date"], errors="coerce", utc=True))
        events = events.sort_values(["_date", "event_id"], na_position="last")
    else:
        events = events.sort_values("event_id")
    return events["event_id"].astype(str).tolist()


def benchmark_uncertainty(
    metrics: pd.DataFrame,
    *,
    baseline: str = "qualifying_order",
    samples: int = 10000,
    seed: int = 42,
    block_length: int = 3,
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

    frame = metrics.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    event_order = _event_order(frame)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    model_intervals: dict[str, dict] = {}
    paired: dict[str, dict] = {}

    for model, group in frame.groupby("model", sort=True):
        group = group.set_index("event_id").loc[[event for event in event_order if event in set(group.event_id)]]
        model_intervals[model] = {}
        for metric, direction in METRIC_DIRECTION.items():
            values = group[metric].to_numpy(dtype=float)
            draws = _bootstrap_mean(values, rng, samples)
            local_block = min(block_length, len(values))
            block_draws = (
                moving_block_bootstrap_mean(
                    values,
                    samples=samples,
                    block_length=local_block,
                    seed=seed + 101,
                )
                if len(values) >= 2
                else draws
            )
            model_intervals[model][metric] = {
                "mean": float(values.mean()),
                "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "moving_block_interval_95": np.quantile(block_draws, [0.025, 0.975]).tolist(),
                "events": int(len(values)),
                "direction": direction,
            }

    baseline_frame = frame[frame.model == baseline].set_index("event_id")
    for model in sorted(set(frame.model) - {baseline}):
        challenger = frame[frame.model == model].set_index("event_id")
        shared = [event for event in event_order if event in baseline_frame.index and event in challenger.index]
        if not shared:
            continue
        paired[model] = {}
        for metric, direction in METRIC_DIRECTION.items():
            differences = (
                challenger.loc[shared, metric].to_numpy(dtype=float)
                - baseline_frame.loc[shared, metric].to_numpy(dtype=float)
            )
            draws = _bootstrap_mean(differences, rng, samples)
            local_block = min(block_length, len(differences))
            block_draws = moving_block_bootstrap_mean(
                differences,
                samples=samples,
                block_length=local_block,
                seed=seed + 211,
            )
            randomization = paired_randomization_test(
                differences,
                direction=direction,
                samples=max(samples, 1000),
                seed=seed + 307,
            )
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
                "moving_block_interval_95": np.quantile(block_draws, [0.025, 0.975]).tolist(),
                "moving_block_length": int(local_block),
                "events": int(len(differences)),
                "model_better_events": int(np.count_nonzero(better)),
                "baseline_better_events": int(np.count_nonzero(worse)),
                "ties": int(np.count_nonzero(tied)),
                "bootstrap_fraction_favorable": float(np.mean(favorable_draw)),
                "paired_randomization": randomization,
                "direction": direction,
            }
            paired[model][metric] = stats
            rows.append({
                "model": model,
                "baseline": baseline,
                "metric": metric,
                **{key: value for key, value in stats.items() if key != "paired_randomization"},
                "randomization_p_two_sided": randomization["p_two_sided"],
                "randomization_p_one_sided_favorable": randomization["p_one_sided_favorable"],
            })

    report = {
        "schema_version": 2,
        "unit": "whole race event",
        "bootstrap_samples": samples,
        "moving_block_length_requested": block_length,
        "seed": seed,
        "baseline": baseline,
        "model_intervals": model_intervals,
        "paired_vs_baseline": paired,
        "interpretation": (
            "Paired differences are model minus baseline on the same sealed races. Negative favors the "
            "model for lower-is-better metrics; positive favors it for higher-is-better metrics. Ordinary "
            "event bootstrap is descriptive. Moving-block intervals are a sensitivity check for adjacent "
            "race dependence. Sign-flip p-values test a paired zero-effect null and are not model-selection inputs."
        ),
        "limitations": [
            "A short sealed test block limits precision regardless of the statistical method.",
            "Moving-block bootstrap addresses only local serial dependence and does not prove stationarity.",
            "Randomization inference assumes paired effect signs are exchangeable under the null.",
        ],
    }
    return pd.DataFrame(rows), report


def save_uncertainty(
    metrics: pd.DataFrame,
    output: Path,
    *,
    baseline: str = "qualifying_order",
    samples: int = 10000,
    seed: int = 42,
    block_length: int = 3,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comparisons, report = benchmark_uncertainty(
        metrics,
        baseline=baseline,
        samples=samples,
        seed=seed,
        block_length=block_length,
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
    parser.add_argument("--block-length", type=int, default=3)
    args = parser.parse_args(argv)
    metrics = pd.read_csv(args.metrics)
    report = save_uncertainty(
        metrics,
        args.output,
        baseline=args.baseline,
        samples=args.samples,
        seed=args.seed,
        block_length=args.block_length,
    )
    print(json.dumps(report["paired_vs_baseline"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
