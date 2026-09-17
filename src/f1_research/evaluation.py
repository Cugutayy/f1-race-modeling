"""Whole-event expanding-window evaluation and reproducible evidence artifacts."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate
from .features import FEATURES, build_features
from .model import METRICS, distribution, fit_models, score_event, scores_for

ORIGIN = "post-qualifying retrospective event-cutoff; qualifying order, not final race grid"
LIMITATIONS = [
    "Retrospective source snapshots lack historical publication/revision timestamps; not a strict as-published backtest.",
    "Result entrants define historical fields; withdrawals before race and later disqualifications may change membership.",
    "No target-race weather, race telemetry, final grid penalties or sprint points are predictive inputs.",
    "Qualifying gap uses Q1 where supplied by collector; wet/evolving Q1 conditions remain a confounder.",
    "Rank regression parameterizes a Plackett-Luce distribution; it is not a maximum-likelihood PL estimator.",
    "Temperature uses a small earlier holdout; calibrated is a procedure, not a claim of demonstrated reliability.",
    "Podium/top10/intervals use 4096 fixed-seed PL simulations; expected rank and winner probabilities are analytic.",
    "Uniform rank metrics use alphabetical tie order and are not meaningful as a ranking benchmark.",
    "Race bootstrap is descriptive and ignores serial dependence; no significance or prospective guarantee is claimed.",
    "Hyperparameters are fixed; tuning after seeing this report requires a new untouched holdout.",
]


def backtest(frame, min_history=12, calibration_events=4):
    if min_history < calibration_events + 3:
        raise ValueError("min_history must be at least calibration_events + 3")
    features = build_features(validate(frame))
    events = features[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    metrics, predictions = [], []
    for event_id, date in events.itertuples(index=False, name=None):
        history = features[features["date"] < date]
        if history["event_id"].nunique() < min_history:
            continue
        event = features[features["event_id"] == event_id].copy()
        fitted, audit = fit_models(history, calibration_events)
        for name, (learner, temperature) in fitted.items():
            values = scores_for(name, event, learner)
            measured, ranks = score_event(event, values, temperature)
            probs = distribution(values, temperature)
            metrics.append({"event_id": event_id, "date": str(date), "model": name,
                            "temperature": temperature, **audit, **measured})
            for i, (_, row) in enumerate(event.iterrows()):
                predictions.append({
                    "event_id": event_id, "date": str(date), "driver": row["driver"],
                    "team": row["team"], "model": name,
                    "actual_position": int(row["finish_position"]),
                    "predicted_position": int(ranks[i]),
                    **{key: float(value[i]) for key, value in probs.items()},
                })
    if not metrics:
        raise ValueError("No test events: provide more history or reduce min_history")
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def save_report(frame, metrics, predictions, output, *, data_kind="historical", provenance=None):
    if data_kind not in ("historical", "synthetic"):
        raise ValueError("data_kind must be historical or synthetic")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby("model", sort=True)[list(METRICS)].mean().reset_index()
    rng = np.random.default_rng(42)
    intervals = {}
    for name, group in metrics.groupby("model", sort=True):
        values = group[list(METRICS)].to_numpy()
        bootstrap = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
        intervals[name] = {key: np.quantile(bootstrap[:, i], [0.025, 0.975]).tolist()
                           for i, key in enumerate(METRICS)}
    # Paired differences preserve which baseline and challenger saw the same race.
    wide = metrics.pivot(index="event_id", columns="model", values="winner_log_loss")
    paired = {}
    for name in wide.columns:
        if name == "qualifying_order":
            continue
        differences = (wide[name] - wide["qualifying_order"]).to_numpy()
        draws = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(axis=1)
        paired[name] = {"mean": float(differences.mean()),
                        "interval_95": np.quantile(draws, [0.025, 0.975]).tolist()}
    canonical = validate(frame).to_csv(index=False)
    data_hash = hashlib.sha256(canonical.encode()).hexdigest()
    prediction_hash = hashlib.sha256(predictions.to_csv(index=False).encode()).hexdigest()
    reliability = []
    edges = np.linspace(0, 1, 11)
    for name, group in predictions.groupby("model", sort=True):
        bins = np.minimum(np.searchsorted(edges, group.win_probability, side="right") - 1, 9)
        for index in sorted(set(bins)):
            selected = group.iloc[np.flatnonzero(bins == index)]
            reliability.append({"model": name, "lower": float(edges[index]),
                                "upper": float(edges[index + 1]), "count": len(selected),
                                "mean_probability": float(selected.win_probability.mean()),
                                "observed_win_rate": float(selected.actual_position.eq(1).mean())})
    report = {
        "schema_version": 1, "series": "f1", "data_kind": data_kind,
        "run_id": hashlib.sha256((data_hash + prediction_hash).encode()).hexdigest()[:16],
        "forecast_origin": ORIGIN, "summary": summary.to_dict("records"),
        "metrics": metrics.to_dict("records"), "predictions": predictions.to_dict("records"),
        "provenance": {**(provenance or {}), "data_sha256": data_hash,
                       "prediction_sha256": prediction_hash},
        "limitations": LIMITATIONS, "features": FEATURES, "seed": 42,
        "events": int(frame["event_id"].nunique()),
        "test_events": int(metrics["event_id"].nunique()),
        "bootstrap_95pct": intervals, "paired_logloss_vs_qualifying": paired,
        "reliability_bins": reliability,
        "runtime": {"python": platform.python_version(),
                    "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
                    "source_sha256": hashlib.sha256(b"".join(
                        p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
                    )).hexdigest()},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "reliability.csv", index=False)
    (output / "REPORT.md").write_text(
        "# F1 chronological research report\n\n"
        f"Run `{report['run_id']}`; data: **{data_kind}**; {report['test_events']} held-out events.\n\n"
        f"Forecast origin: {ORIGIN}.\n\n"
        "Each event is held out as a whole. Training and temperature selection use strictly earlier events. "
        "The last calibration block is not used to fit the score learner. Baselines use that same calibration block.\n\n"
        "```text\n" + summary.round(4).to_string(index=False) + "\n```\n\n"
        "MAE, winner log loss and multiclass Brier sum: lower is better. Accuracy and podium recall: higher. "
        "Race bootstrap and paired log-loss differences are in report.json. Negative paired differences favor "
        "the named model over qualifying order. These are descriptive uncertainty estimates.\n\n"
        "## Limitations\n\n" + "\n".join(f"- {item}" for item in LIMITATIONS) + "\n",
        encoding="utf-8",
    )
    return report


def predict_entries(history, entries, calibration_events=4):
    history = validate(history)
    entries = entries.copy()
    required = {"event_id", "date", "year", "driver", "team", "circuit",
                "quali_position", "quali_seconds"}
    if required - set(entries) or len(entries) < 2:
        raise ValueError("Provide a real qualifying entry list with all required columns")
    entries["date"] = pd.to_datetime(entries["date"], utc=True, errors="raise")
    if entries["date"].isna().any() or any(entries[c].nunique() != 1 for c in (
            "date", "event_id", "year", "circuit")) or entries["driver"].duplicated().any():
        raise ValueError("Entries must describe one event with unique drivers")
    for col in ("event_id", "driver", "team", "circuit"):
        if entries[col].isna().any() or entries[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing entry identifier: {col}")
    for col in ("quali_position", "quali_seconds"):
        entries[col] = pd.to_numeric(entries[col], errors="raise")
        if np.isinf(entries[col]).any() or (entries[col].dropna() <= 0).any():
            raise ValueError(f"Invalid entry feature: {col}")
    cutoff = entries["date"].iloc[0]
    history = history[(history["date"] < cutoff) & ~history["event_id"].isin(entries["event_id"])]
    for col in ("finish_position", "points", "dnf"):
        entries[col] = np.nan
    features = build_features(pd.concat([history, entries], ignore_index=True))
    fitted, audit = fit_models(features[features["date"] < cutoff], calibration_events)
    event = features[features["event_id"].isin(entries["event_id"])]
    output = []
    for name, (learner, temperature) in fitted.items():
        scores = scores_for(name, event, learner)
        result = event[["event_id", "driver", "team"]].copy()
        result["model"] = name
        result["predicted_position"] = pd.Series(scores, index=result.index).rank(method="first").astype(int)
        for key, values in distribution(scores, temperature).items():
            result[key] = values
        output.append(result)
    return pd.concat(output, ignore_index=True), {**audit, "forecast_origin": ORIGIN}
