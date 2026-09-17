"""Persist winner-probability calibration diagnostics from sealed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .statistical_diagnostics import winner_calibration_diagnostics


def save_calibration_evidence(
    predictions: pd.DataFrame,
    output: Path,
    *,
    bins: int = 10,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = winner_calibration_diagnostics(predictions, bins=bins)
    (output / "winner_calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    reliability_rows = []
    adaptive_rows = []
    for model, payload in report["models"].items():
        reliability_rows.extend({"model": model, **row} for row in payload["reliability_bins"])
        adaptive_rows.extend({"model": model, **row} for row in payload["adaptive_bins"])
    pd.DataFrame(reliability_rows).to_csv(output / "reliability_bins.csv", index=False)
    pd.DataFrame(adaptive_rows).to_csv(output / "adaptive_bins.csv", index=False)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)
    predictions = pd.read_csv(args.predictions)
    report = save_calibration_evidence(predictions, args.output, bins=args.bins)
    compact = {
        model: {
            "events": values["events"],
            "ece_equal_width": values["ece_equal_width"],
            "ece_adaptive": values["ece_adaptive"],
            "calibration_intercept": values["calibration_logistic"]["intercept"],
            "calibration_slope": values["calibration_logistic"]["slope"],
            "top_choice": values["top_choice"],
        }
        for model, values in report["models"].items()
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
