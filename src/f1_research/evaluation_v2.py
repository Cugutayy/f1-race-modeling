"""V2 benchmark: disjoint fit, tuning, calibration and sealed chronological test blocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import validate
from .features import FEATURES, build_features
from .model import distribution, estimator, probability, score_event
from .modern_models import candidate_specs, fit_selected, tune_forward_events
from .plackett_luce import PlackettLuceRanker

TEMPERATURES = np.geomspace(0.02, 3.0, 45)


def _baseline_scores(name: str, event: pd.DataFrame) -> np.ndarray:
    if name == "qualifying_order":
        return event.quali_position.fillna(len(event) + 1).to_numpy(dtype=float) / len(event)
    if name == "recent_form":
        return event.driver_form.to_numpy(dtype=float)
    raise ValueError(name)


def _select_temperature(events: list[pd.DataFrame], score_fn) -> float:
    losses = []
    for temperature in TEMPERATURES:
        event_losses = []
        for event in events:
            scores = np.asarray(score_fn(event), dtype=float)
            p = probability(scores, temperature)
            winner = event.finish_position.to_numpy(dtype=float) == 1
            event_losses.append(-np.log(max(float(p[winner][0]), 1e-15)))
        losses.append(float(np.mean(event_losses)))
    return float(TEMPERATURES[int(np.argmin(losses))])


def _tune_pl(frame: pd.DataFrame, tuning_events: int, min_fit_events: int,
             l2_values=(0.5, 2.0, 5.0, 15.0, 50.0)) -> tuple[float, pd.DataFrame]:
    events = frame[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    tune_ids = events.iloc[-tuning_events:].event_id.tolist()
    rows = []
    for l2 in l2_values:
        losses = []
        error = None
        try:
            for event_id in tune_ids:
                target = frame[frame.event_id == event_id]
                train = frame[frame.date < target.date.min()]
                if train.event_id.nunique() < min_fit_events:
                    continue
                model = PlackettLuceRanker(l2=l2).fit(train)
                losses.append(score_event(target, model.predict(target), 1.0)[0]["position_mae"])
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        rows.append({"model": "plackett_luce", "l2": l2, "events": len(losses),
                     "mean_position_mae": float(np.mean(losses)) if losses and not error else np.inf,
                     "error": error})
    table = pd.DataFrame(rows).sort_values(["mean_position_mae", "l2"])
    valid = table[np.isfinite(table.mean_position_mae)]
    if valid.empty:
        raise ValueError("No Plackett-Luce candidate completed tuning")
    return float(valid.iloc[0].l2), table.reset_index(drop=True)


def _split_blocks(features: pd.DataFrame, test_events: int, tuning_events: int,
                  calibration_events: int, min_fit_events: int) -> dict[str, list[str]]:
    events = features[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    required = min_fit_events + tuning_events + calibration_events + test_events
    if len(events) < required:
        raise ValueError(f"Need at least {required} events, received {len(events)}")
    ids = events.event_id.tolist()
    test = ids[-test_events:]
    calibration = ids[-test_events - calibration_events:-test_events]
    tuning = ids[-test_events - calibration_events - tuning_events:-test_events - calibration_events]
    fit = ids[: -test_events - calibration_events - tuning_events]
    if len(fit) < min_fit_events:
        raise AssertionError("split produced insufficient fit history")
    return {"fit": fit, "tuning": tuning, "calibration": calibration, "test": test}


def benchmark_v2(frame: pd.DataFrame, *, test_events: int = 12, tuning_events: int = 6,
                 calibration_events: int = 4, min_fit_events: int = 20,
                 modern_names: tuple[str, ...] = ("hist_gradient_boosting", "extra_trees"),
                 max_specs_per_model: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = build_features(validate(frame))
    split = _split_blocks(features, test_events, tuning_events, calibration_events, min_fit_events)
    pre_cal_ids = split["fit"] + split["tuning"]
    pre_cal = features[features.event_id.isin(pre_cal_ids)].copy()
    calibration = features[features.event_id.isin(split["calibration"])].copy()
    test = features[features.event_id.isin(split["test"])].copy()

    modern_specs = candidate_specs(modern_names)
    modern_best, modern_table = tune_forward_events(
        pre_cal, specs=modern_specs, tuning_events=tuning_events,
        min_fit_events=min_fit_events, max_specs_per_model=max_specs_per_model)
    modern_model = fit_selected(pre_cal, modern_best)

    pl_l2, pl_table = _tune_pl(pre_cal, tuning_events, min_fit_events)
    pl_model = PlackettLuceRanker(l2=pl_l2).fit(pre_cal)

    ridge = estimator("ridge_rank").fit(
        pre_cal[FEATURES],
        (pre_cal.finish_position - 1)
        / (pre_cal.groupby("event_id").driver.transform("size") - 1).clip(lower=1),
    )

    score_functions = {
        "qualifying_order": lambda event: _baseline_scores("qualifying_order", event),
        "recent_form": lambda event: _baseline_scores("recent_form", event),
        "ridge_rank": lambda event: ridge.predict(event[FEATURES]),
        f"modern::{modern_best.name}": lambda event: modern_model.predict(event[FEATURES]),
        "plackett_luce_mle": lambda event: pl_model.predict(event),
    }
    calibration_events_frames = [group for _, group in calibration.groupby("event_id", sort=True)]
    temperatures = {
        name: _select_temperature(calibration_events_frames, fn)
        for name, fn in score_functions.items()
    }

    metrics, predictions = [], []
    for event_id, event in test.groupby("event_id", sort=True):
        for name, score_fn in score_functions.items():
            scores = np.asarray(score_fn(event), dtype=float)
            measured, ranks = score_event(event, scores, temperatures[name])
            probs = distribution(scores, temperatures[name], samples=8192, seed=42)
            metrics.append({"event_id": event_id, "date": str(event.date.iloc[0]), "model": name,
                            "temperature": temperatures[name], **measured})
            for i, (_, row) in enumerate(event.iterrows()):
                predictions.append({
                    "event_id": event_id, "date": str(row.date), "driver": row.driver,
                    "team": row.team, "model": name, "actual_position": int(row.finish_position),
                    "predicted_position": int(ranks[i]),
                    **{key: float(value[i]) for key, value in probs.items()},
                })

    audit = {
        "protocol": "disjoint chronological fit -> tuning -> calibration -> sealed test",
        "split": split,
        "selected_modern": asdict(modern_best),
        "modern_tuning": modern_table.replace({np.inf: None}).to_dict("records"),
        "selected_pl_l2": pl_l2,
        "pl_tuning": pl_table.replace({np.inf: None}).to_dict("records"),
        "temperatures": temperatures,
        "pl_optimization": pl_model.optimization_,
        "feature_schema": FEATURES,
        "test_updates_model": False,
    }
    return pd.DataFrame(metrics), pd.DataFrame(predictions), audit


def save_v2_report(frame: pd.DataFrame, metrics: pd.DataFrame, predictions: pd.DataFrame,
                   audit: dict[str, Any], output: Path, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby("model", sort=True)[
        ["position_mae", "winner_log_loss", "winner_brier", "winner_accuracy", "podium_recall"]
    ].mean().reset_index()
    canonical = validate(frame).to_csv(index=False)
    report = {
        "schema_version": 2, "series": "f1", "data_kind": "historical",
        "run_id": hashlib.sha256((canonical + predictions.to_csv(index=False)).encode()).hexdigest()[:16],
        "summary": summary.to_dict("records"), "metrics": metrics.to_dict("records"),
        "predictions": predictions.to_dict("records"), "audit": audit,
        "provenance": provenance or {"provider": "user-supplied; verify source manifest"},
        "limitations": [
            "This benchmark uses retrospective source snapshots unless provenance proves as-published availability.",
            "The sealed test block is not used for model, hyperparameter or temperature selection.",
            "Public data does not expose full team telemetry, fuel load, setup or tyre internal temperatures.",
            "A foundation model is a challenger, not automatically preferred over simpler baselines.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "selection.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return report
