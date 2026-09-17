"""Explicit research commands; errors never produce invented entry lists."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import collect
from .demo import synthetic_history
from .evaluation import backtest, predict_entries, save_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="F1 event-cutoff research")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Clearly synthetic offline software demonstration")
    demo.add_argument("--output", type=Path, default=Path("reports/demo"))
    fetch = commands.add_parser("collect")
    fetch.add_argument("--years", type=int, nargs="+", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--cache", type=Path, default=Path("data/raw"))
    fetch.add_argument("--offline", action="store_true")
    evaluate = commands.add_parser("backtest")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--min-history", type=int, default=12)
    evaluate.add_argument("--calibration-events", type=int, default=4)
    evaluate.add_argument("--data-kind", choices=("historical", "synthetic"), default="historical")
    predict = commands.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True, help="Historical results CSV")
    predict.add_argument("--entries", type=Path, required=True, help="One qualifying event CSV")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--calibration-events", type=int, default=4)
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
        source = args.input.with_suffix(".provenance.json")
        provenance = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {
            "provider": "user-supplied CSV; source provenance not verified"}
        metrics, predictions = backtest(frame, args.min_history, args.calibration_events)
        report = save_report(frame, metrics, predictions, args.output,
                             data_kind=args.data_kind, provenance=provenance)
        print(f"Run {report['run_id']}: {report['test_events']} held-out events -> {args.output}")
    else:
        predictions, audit = predict_entries(pd.read_csv(args.input), pd.read_csv(args.entries),
                                            args.calibration_events)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": 1, "series": "f1", **audit,
                                          "predictions": predictions.to_dict("records")},
                                         indent=2, allow_nan=False), encoding="utf-8")
        print(f"Predictions -> {args.output}")


if __name__ == "__main__":
    main()
