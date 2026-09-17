"""Explicit research commands; errors never produce invented entry lists."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import collect
from .demo import synthetic_history
from .evaluation import backtest, predict_entries, save_report


def _provenance(path: Path) -> dict:
    source = path.with_suffix(".provenance.json")
    return json.loads(source.read_text(encoding="utf-8")) if source.exists() else {
        "provider": "user-supplied CSV; source provenance not verified"}


def _simulation_config(priors: Path | None, samples: int):
    if priors is None:
        from .strategy import SimulationConfig

        return SimulationConfig(samples=samples), {"source": "built_in_defaults"}
    from .strategy_calibration import load_simulation_config

    return load_simulation_config(priors, samples=samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description="F1 event-cutoff and live race research")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Clearly synthetic offline software demonstration")
    demo.add_argument("--output", type=Path, default=Path("reports/demo"))

    fetch = commands.add_parser("collect")
    fetch.add_argument("--years", type=int, nargs="+", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--cache", type=Path, default=Path("data/raw"))
    fetch.add_argument("--offline", action="store_true")

    evaluate = commands.add_parser("backtest", help="Original expanding-window benchmark")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--min-history", type=int, default=12)
    evaluate.add_argument("--calibration-events", type=int, default=4)
    evaluate.add_argument("--data-kind", choices=("historical", "synthetic"), default="historical")

    v2 = commands.add_parser("benchmark-v2", help="Disjoint fit/tune/calibrate/sealed-test benchmark")
    v2.add_argument("--input", type=Path, required=True)
    v2.add_argument("--output", type=Path, required=True)
    v2.add_argument("--test-events", type=int, default=12)
    v2.add_argument("--tuning-events", type=int, default=6)
    v2.add_argument("--calibration-events", type=int, default=4)
    v2.add_argument("--min-fit-events", type=int, default=20)
    v2.add_argument("--models", nargs="+", default=["hist_gradient_boosting", "extra_trees"],
                    choices=["hist_gradient_boosting", "extra_trees", "xgboost", "lightgbm",
                             "catboost", "tabicl_v2"])
    v2.add_argument("--max-specs-per-model", type=int, default=4)
    v2.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip rank-ensemble tuning. Useful for expensive single-model challenger runs.",
    )

    predict = commands.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True, help="Historical results CSV")
    predict.add_argument("--entries", type=Path, required=True, help="One qualifying event CSV")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--calibration-events", type=int, default=4)

    live = commands.add_parser("live-predict", help="Monte Carlo forecast from a captured state.json")
    live.add_argument("--state", type=Path, required=True)
    live.add_argument("--total-laps", type=int, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--samples", type=int, default=20_000)
    live.add_argument("--priors", type=Path, help="Optional calibrated strategy_priors.json")

    pit = commands.add_parser("pit-window", help="Compare counterfactual pit windows for one live driver")
    pit.add_argument("--state", type=Path, required=True)
    pit.add_argument("--total-laps", type=int, required=True)
    pit.add_argument("--driver-number", type=int, required=True)
    pit.add_argument("--output", type=Path, required=True)
    pit.add_argument("--samples", type=int, default=20_000)
    pit.add_argument("--priors", type=Path, help="Optional calibrated strategy_priors.json")

    args = parser.parse_args(argv)
    if args.command == "demo":
        frame = synthetic_history()
        metrics, predictions = backtest(frame, min_history=10)
        report = save_report(frame, metrics, predictions, args.output, data_kind="synthetic",
                             provenance={"generator": "synthetic-v1", "seed": 42})
        print(f"SYNTHETIC demo {report['run_id']} -> {args.output}")
    elif args.command == "collect":
        frame = collect(args.years, args.cache, args.offline)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        args.output.with_suffix(".provenance.json").write_text(
            json.dumps(frame.attrs["provenance"], indent=2), encoding="utf-8")
        print(f"Collected {len(frame)} rows / {frame.event_id.nunique()} events -> {args.output}")
    elif args.command == "backtest":
        frame = pd.read_csv(args.input)
        metrics, predictions = backtest(frame, args.min_history, args.calibration_events)
        report = save_report(frame, metrics, predictions, args.output,
                             data_kind=args.data_kind, provenance=_provenance(args.input))
        print(f"Run {report['run_id']}: {report['test_events']} held-out events -> {args.output}")
    elif args.command == "benchmark-v2":
        from .evaluation_v2 import benchmark_v2, save_v2_report

        frame = pd.read_csv(args.input)
        metrics, predictions, audit = benchmark_v2(
            frame, test_events=args.test_events, tuning_events=args.tuning_events,
            calibration_events=args.calibration_events, min_fit_events=args.min_fit_events,
            modern_names=tuple(args.models), max_specs_per_model=args.max_specs_per_model,
            include_ensemble=not args.no_ensemble,
        )
        report = save_v2_report(frame, metrics, predictions, audit, args.output,
                                provenance=_provenance(args.input))
        print(f"V2 sealed benchmark {report['run_id']} -> {args.output}")
    elif args.command == "predict":
        predictions, audit = predict_entries(pd.read_csv(args.input), pd.read_csv(args.entries),
                                            args.calibration_events)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": 1, "series": "f1", **audit,
                                          "predictions": predictions.to_dict("records")},
                                         indent=2, allow_nan=False), encoding="utf-8")
        print(f"Predictions -> {args.output}")
    elif args.command == "live-predict":
        from .strategy import predict_from_state

        state = json.loads(args.state.read_text(encoding="utf-8"))
        config, prior_audit = _simulation_config(args.priors, args.samples)
        result = predict_from_state(state, args.total_laps, config=config)
        result["strategy_prior_source"] = prior_audit
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Live simulation -> {args.output}")
    else:
        from .strategy import compare_pit_windows

        state = json.loads(args.state.read_text(encoding="utf-8"))
        config, prior_audit = _simulation_config(args.priors, args.samples)
        result = compare_pit_windows(state, args.total_laps, args.driver_number, config=config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "strategy_prior_source": prior_audit,
            "scenarios": result,
        }, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Pit-window scenarios -> {args.output}")


if __name__ == "__main__":
    main()
