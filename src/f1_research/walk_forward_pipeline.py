"""Run real-provider strict walk-forward backtests and persist auditable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lap_intelligence import LapModelSpec
from .lap_pipeline import collect_recent
from .walk_forward import WalkForwardConfig, walk_forward_strict


def model_specs(*, include_modern: bool, include_foundation: bool) -> tuple[LapModelSpec, ...]:
    specs = [
        LapModelSpec("hist_gradient_boosting", {}),
        LapModelSpec("extra_trees", {}),
    ]
    if include_modern:
        specs.extend(
            [
                LapModelSpec("xgboost", {}),
                LapModelSpec("lightgbm", {}),
                LapModelSpec("catboost", {}),
            ]
        )
    if include_foundation:
        specs.append(LapModelSpec("tabicl_v2", {}))
    return tuple(specs)


def run(
    *,
    year: int,
    race_count: int,
    output: Path,
    latency_s: float = 1.0,
    refresh: bool = False,
    include_modern: bool = False,
    include_foundation: bool = False,
    max_events: int | None = None,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets, manifests = collect_recent(
        year,
        race_count,
        output / "source",
        latency_s=latency_s,
        refresh=refresh,
    )
    folds, predictions, audit = walk_forward_strict(
        datasets,
        specs=model_specs(
            include_modern=include_modern,
            include_foundation=include_foundation,
        ),
        config=WalkForwardConfig(max_events=max_events),
    )
    folds.to_csv(output / "walk_forward_folds.csv", index=False)
    predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
    report = {
        "schema_version": 1,
        "task": "strict_next_green_lap_walk_forward",
        "year": year,
        "race_count": race_count,
        "provider": "OpenF1",
        "simulated_provider_latency_s": latency_s,
        "source_sessions": [
            {
                "session_key": int(item["session_key"]),
                "country_name": item.get("country_name"),
                "location": item.get("location"),
                "date_start": item.get("date_start"),
                "date_end": item.get("date_end"),
                "dataset_sha256": item.get("dataset_sha256"),
            }
            for item in manifests
        ],
        "models": [spec.name for spec in model_specs(
            include_modern=include_modern,
            include_foundation=include_foundation,
        )],
        "audit": audit,
        "outputs": {
            "folds": "walk_forward_folds.csv",
            "predictions": "walk_forward_predictions.csv",
        },
    }
    (output / "walk_forward_report.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race-count", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/local/walk-forward"))
    parser.add_argument("--latency-s", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--modern", action="store_true")
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args(argv)
    report = run(
        year=args.year,
        race_count=args.race_count,
        output=args.output,
        latency_s=args.latency_s,
        refresh=args.refresh,
        include_modern=args.modern,
        include_foundation=args.foundation,
        max_events=args.max_events,
    )
    print(json.dumps(report["audit"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
