"""V2 benchmark: disjoint fit, tuning, calibration and sealed chronological test blocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import survivorship_audit, validate
from .features import FEATURES, build_features
from .group_rankers import NativeGroupRanker, RankingSpec
from .model import distribution, estimator, probability, score_event
from .modern_models import candidate_specs, fit_selected, tune_forward_events
from .plackett_luce import PlackettLuceRanker
from .rank_ensemble import blend_scores, tune_rank_ensemble

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


def _tune_pl(
    frame: pd.DataFrame,
    tuning_events: int,
    min_fit_events: int,
    l2_values=(0.5, 2.0, 5.0, 15.0, 50.0),
) -> tuple[float, pd.DataFrame]:
    """Tune PL regularization on winner log loss from walk-forward tuning events."""
    events = frame[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    tune_ids = events.iloc[-tuning_events:].event_id.tolist()
    rows = []
    for l2 in l2_values:
        rank_losses = []
        predicted: dict[str, np.ndarray] = {}
        tuning_frames: dict[str, pd.DataFrame] = {}
        error = None
        try:
            for event_id in tune_ids:
                target = frame[frame.event_id == event_id]
                train = frame[frame.date < target.date.min()]
                if train.event_id.nunique() < min_fit_events:
                    continue
                model = PlackettLuceRanker(l2=l2).fit(train)
                raw_scores = np.asarray(model.predict(target), dtype=float)
                measured, _ = score_event(target, raw_scores, 1.0)
                rank_losses.append(float(measured["position_mae"]))
                predicted[str(event_id)] = raw_scores
                tuning_frames[str(event_id)] = target
            if not predicted:
                raise ValueError("PL candidate had no eligible tuning events")
            ordered_frames = [
                tuning_frames[str(event_id)]
                for event_id in tune_ids
                if str(event_id) in predicted
            ]
            temperature = _select_temperature(
                ordered_frames,
                lambda event: predicted[str(event.event_id.iloc[0])],
            )
            probability_losses = []
            for event in ordered_frames:
                scores = predicted[str(event.event_id.iloc[0])]
                measured, _ = score_event(event, scores, temperature)
                probability_losses.append(float(measured["winner_log_loss"]))
            rows.append({
                "model": "plackett_luce",
                "l2": l2,
                "events": len(rank_losses),
                "mean_winner_log_loss": float(np.mean(probability_losses)),
                "mean_position_mae": float(np.mean(rank_losses)),
                "tuning_temperature": temperature,
                "error": None,
            })
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
            rows.append({
                "model": "plackett_luce",
                "l2": l2,
                "events": len(rank_losses),
                "mean_winner_log_loss": np.inf,
                "mean_position_mae": float(np.mean(rank_losses)) if rank_losses else np.inf,
                "tuning_temperature": np.nan,
                "error": error,
            })
    table = pd.DataFrame(rows).sort_values(
        ["mean_winner_log_loss", "mean_position_mae", "l2"],
        na_position="last",
    )
    valid = table[
        np.isfinite(table.mean_winner_log_loss) & np.isfinite(table.mean_position_mae)
    ]
    if valid.empty:
        raise ValueError("No Plackett-Luce candidate completed tuning")
    return float(valid.iloc[0].l2), table.reset_index(drop=True)


def _split_blocks(
    features: pd.DataFrame,
    test_events: int,
    tuning_events: int,
    calibration_events: int,
    min_fit_events: int,
) -> dict[str, list[str]]:
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


def _tune_ensemble(
    pre_cal: pd.DataFrame,
    modern_spec,
    pl_l2: float,
    tuning_events: int,
    min_fit_events: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Walk-forward component predictions; weights see only the designated tuning block."""
    events = pre_cal[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    tune_ids = events.iloc[-tuning_events:].event_id.tolist()
    predictions: list[tuple[pd.DataFrame, dict[str, np.ndarray]]] = []
    for event_id in tune_ids:
        target = pre_cal[pre_cal.event_id == event_id]
        train = pre_cal[pre_cal.date < target.date.min()]
        if train.event_id.nunique() < min_fit_events:
            continue
        modern = fit_selected(train, modern_spec)
        pl = PlackettLuceRanker(l2=pl_l2).fit(train)
        predictions.append((target, {
            "modern": np.asarray(modern.predict(target[FEATURES]), dtype=float),
            "pl": np.asarray(pl.predict(target), dtype=float),
            "qualifying": _baseline_scores("qualifying_order", target),
        }))
    if not predictions:
        raise ValueError("No eligible tuning events for rank ensemble")
    return tune_rank_ensemble(predictions, step=0.25)


def benchmark_v2(
    frame: pd.DataFrame,
    *,
    test_events: int = 12,
    tuning_events: int = 6,
    calibration_events: int = 4,
    min_fit_events: int = 20,
    modern_names: tuple[str, ...] = ("hist_gradient_boosting", "extra_trees"),
    max_specs_per_model: int = 4,
    modern_selection_metric: str = "winner_log_loss",
    include_ensemble: bool = True,
    native_ranker_names: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    survivorship = survivorship_audit(frame)
    features = build_features(validate(frame))
    split = _split_blocks(features, test_events, tuning_events, calibration_events, min_fit_events)
    pre_cal_ids = split["fit"] + split["tuning"]
    pre_cal = features[features.event_id.isin(pre_cal_ids)].copy()
    calibration = features[features.event_id.isin(split["calibration"])].copy()
    test = features[features.event_id.isin(split["test"])].copy()

    modern_specs = candidate_specs(modern_names)
    modern_best, modern_table = tune_forward_events(
        pre_cal,
        specs=modern_specs,
        tuning_events=tuning_events,
        min_fit_events=min_fit_events,
        max_specs_per_model=max_specs_per_model,
        selection_metric=modern_selection_metric,
    )
    modern_model = fit_selected(pre_cal, modern_best)

    pl_l2, pl_table = _tune_pl(pre_cal, tuning_events, min_fit_events)
    pl_model = PlackettLuceRanker(l2=pl_l2).fit(pre_cal)

    native_rankers = {
        name: NativeGroupRanker(RankingSpec(name, {})).fit(pre_cal)
        for name in native_ranker_names
    }

    ridge = estimator("ridge_rank").fit(
        pre_cal[FEATURES],
        (pre_cal.finish_position - 1)
        / (pre_cal.groupby("event_id").driver.transform("size") - 1).clip(lower=1),
    )

    ensemble_weights: dict[str, float] | None = None
    ensemble_table = pd.DataFrame()
    if include_ensemble:
        ensemble_weights, ensemble_table = _tune_ensemble(
            pre_cal,
            modern_best,
            pl_l2,
            tuning_events,
            min_fit_events,
        )

    def ensemble_scores(event: pd.DataFrame) -> np.ndarray:
        if ensemble_weights is None:
            raise ValueError("Ensemble is disabled")
        return blend_scores({
            "modern": np.asarray(modern_model.predict(event[FEATURES]), dtype=float),
            "pl": np.asarray(pl_model.predict(event), dtype=float),
            "qualifying": _baseline_scores("qualifying_order", event),
        }, ensemble_weights)

    score_functions = {
        "qualifying_order": lambda event: _baseline_scores("qualifying_order", event),
        "recent_form": lambda event: _baseline_scores("recent_form", event),
        "ridge_rank": lambda event: ridge.predict(event[FEATURES]),
        f"modern::{modern_best.name}": lambda event: modern_model.predict(event[FEATURES]),
        "plackett_luce_mle": lambda event: pl_model.predict(event),
    }
    if include_ensemble:
        score_functions["rank_ensemble"] = ensemble_scores
    for name, ranker in native_rankers.items():
        score_functions[f"native::{name}"] = lambda event, ranker=ranker: ranker.predict(event)

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
            metrics.append({
                "event_id": event_id,
                "date": str(event.date.iloc[0]),
                "model": name,
                "temperature": temperatures[name],
                **measured,
            })
            for i, (_, row) in enumerate(event.iterrows()):
                predictions.append({
                    "event_id": event_id,
                    "date": str(row.date),
                    "driver": row.driver,
                    "team": row.team,
                    "model": name,
                    "actual_position": int(row.finish_position),
                    "predicted_position": int(ranks[i]),
                    **{key: float(value[i]) for key, value in probs.items()},
                })

    audit = {
        "protocol": "disjoint chronological fit -> tuning -> calibration -> sealed test",
        "split": split,
        "selected_modern": asdict(modern_best),
        "modern_selection_metric": modern_selection_metric,
        "modern_tuning_temperature_is_final": False,
        "modern_tuning": modern_table.replace({np.inf: None, -np.inf: None}).to_dict("records"),
        "selected_pl_l2": pl_l2,
        "pl_selection_metric": "winner_log_loss",
        "pl_tuning_temperature_is_final": False,
        "pl_tuning": pl_table.replace({np.inf: None, -np.inf: None}).to_dict("records"),
        "ensemble_enabled": include_ensemble,
        "ensemble_weights": ensemble_weights,
        "ensemble_selection_metric": "winner_log_loss" if include_ensemble else None,
        "ensemble_tuning_temperature_is_final": False if include_ensemble else None,
        "ensemble_tuning": (
            ensemble_table.replace({np.inf: None, -np.inf: None}).to_dict("records")
            if include_ensemble
            else []
        ),
        "temperatures": temperatures,
        "pl_optimization": pl_model.optimization_,
        "feature_schema": FEATURES,
        "survivorship": survivorship,
        "native_rankers": list(native_ranker_names),
        "native_ranker_protocol": (
            "fixed-hyperparameter group-aware ranking challengers fit only on pre-calibration history; "
            "probability temperature estimated on the disjoint calibration block"
        ),
        "test_updates_model": False,
        "selection_note": (
            "Tuning-block temperatures and ensemble weights compare candidates only; all final model "
            "temperatures are re-estimated on the disjoint calibration block."
        ),
    }
    return pd.DataFrame(metrics), pd.DataFrame(predictions), audit


def _winner_calibration(predictions: pd.DataFrame, bins: int = 10) -> list[dict[str, Any]]:
    """Reliability/ECE for mutually-exclusive race winner probabilities."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    required = {"model", "actual_position", "win_probability"}
    if not required.issubset(predictions.columns):
        raise ValueError("winner calibration requires model, actual_position and win_probability")
    output = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for model_name, group in predictions.groupby("model", sort=True):
        probability_values = group["win_probability"].to_numpy(dtype=float)
        outcomes = (group["actual_position"].to_numpy(dtype=int) == 1).astype(float)
        if np.any(~np.isfinite(probability_values)) or np.any((probability_values < 0) | (probability_values > 1)):
            raise ValueError(f"invalid winner probabilities for {model_name}")
        assignments = np.minimum(np.digitize(probability_values, edges[1:-1], right=False), bins - 1)
        ece = 0.0
        reliability = []
        for index in range(bins):
            mask = assignments == index
            if not mask.any():
                continue
            mean_probability = float(probability_values[mask].mean())
            observed_rate = float(outcomes[mask].mean())
            weight = float(mask.mean())
            ece += weight * abs(mean_probability - observed_rate)
            reliability.append({
                "bin": index,
                "count": int(mask.sum()),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            })
        output.append({"model": str(model_name), "winner_ece": float(ece), "reliability": reliability})
    return output


def save_v2_report(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: dict[str, Any],
    output: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby("model", sort=True)[
        ["position_mae", "winner_log_loss", "winner_brier", "winner_accuracy", "podium_recall", "spearman_rank", "kendall_rank", "ndcg"]
    ].mean().reset_index()
    canonical = validate(frame).to_csv(index=False)
    report = {
        "schema_version": 2,
        "series": "f1",
        "data_kind": "historical",
        "test_events": len(set(audit["split"]["test"])),
        "run_id": hashlib.sha256((canonical + predictions.to_csv(index=False)).encode()).hexdigest()[:16],
        "summary": summary.to_dict("records"),
        "winner_calibration": _winner_calibration(predictions),
        "metrics": metrics.to_dict("records"),
        "predictions": predictions.to_dict("records"),
        "audit": audit,
        "provenance": provenance or {"provider": "user-supplied; verify source manifest"},
        "limitations": [
            "This benchmark uses retrospective source snapshots unless provenance proves as-published availability.",
            "The sealed test block is not used for model, hyperparameter, ensemble-weight or temperature selection.",
            "Tuning-block log loss is used for model selection and is not reported as sealed performance evidence.",
            "Public data does not expose full team telemetry, fuel load, setup or tyre internal temperatures.",
            "A foundation model or ensemble is a challenger, not automatically preferred over simpler baselines.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "selection.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return report
