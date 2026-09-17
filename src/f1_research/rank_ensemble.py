"""Leakage-safe rank-score blending for race-distribution challengers."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TEMPERATURES = np.geomspace(0.02, 3.0, 45)


def normalized_event_scores(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary score scales to stable [0,1] within-event ranks."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Ensemble component requires at least two finite scores")
    ranks = np.empty(len(scores), dtype=float)
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(len(scores), dtype=float)
    return ranks / max(1, len(scores) - 1)


def blend_scores(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not components:
        raise ValueError("No ensemble components")
    if set(components) != set(weights):
        raise ValueError("Ensemble weights must match component names")
    total_weight = float(sum(weights.values()))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Ensemble weights must have positive finite mass")
    output = None
    for name in sorted(components):
        weight = float(weights[name]) / total_weight
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Ensemble weights must be finite and non-negative")
        normalized = normalized_event_scores(components[name])
        output = weight * normalized if output is None else output + weight * normalized
    return np.asarray(output, dtype=float)


def simplex_weights(names: tuple[str, ...], step: float = 0.25) -> list[dict[str, float]]:
    if len(names) < 2:
        raise ValueError("At least two ensemble components are required")
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("step must be in (0,1]")
    units_float = 1.0 / step
    units = int(round(units_float))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one exactly")
    output = []
    for allocation in product(range(units + 1), repeat=len(names)):
        if sum(allocation) != units:
            continue
        # Require at least two active components; a one-hot candidate adds no value
        # over the already-reported individual models.
        if sum(value > 0 for value in allocation) < 2:
            continue
        output.append({name: allocation[i] / units for i, name in enumerate(names)})
    return output


def _winner_log_loss(event: pd.DataFrame, scores: np.ndarray, temperature: float) -> float:
    winner = event.finish_position.to_numpy(dtype=float) == 1
    if winner.sum() != 1:
        raise ValueError("Each tuning event must contain exactly one winner")
    logits = -np.asarray(scores, dtype=float) / float(temperature)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return float(-np.log(max(float(p[winner][0]), 1e-15)))


def _position_mae(event: pd.DataFrame, scores: np.ndarray) -> float:
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return float(np.mean(np.abs(ranks - event.finish_position.to_numpy(dtype=float))))


def tune_rank_ensemble(
    tuning_predictions: list[tuple[pd.DataFrame, dict[str, np.ndarray]]],
    *,
    step: float = 0.25,
    temperatures: np.ndarray = DEFAULT_TEMPERATURES,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune blend weights and a disposable tuning temperature on past events only.

    The selected temperature is diagnostic/model-selection state only. Callers must
    recalibrate the final fitted ensemble on a later disjoint calibration block.
    """
    if not tuning_predictions:
        raise ValueError("No tuning predictions for ensemble")
    component_names = tuple(sorted(tuning_predictions[0][1]))
    if any(tuple(sorted(parts)) != component_names for _, parts in tuning_predictions):
        raise ValueError("All ensemble tuning events must expose the same components")
    rows: list[dict[str, Any]] = []
    for weights in simplex_weights(component_names, step=step):
        blended = [
            (event, blend_scores(parts, weights))
            for event, parts in tuning_predictions
        ]
        temperature_losses = []
        for temperature in temperatures:
            temperature_losses.append(float(np.mean([
                _winner_log_loss(event, scores, float(temperature))
                for event, scores in blended
            ])))
        best_index = int(np.argmin(temperature_losses))
        selected_temperature = float(temperatures[best_index])
        rank_mae = float(np.mean([
            _position_mae(event, scores) for event, scores in blended
        ]))
        rows.append({
            "weights": weights,
            "mean_winner_log_loss": temperature_losses[best_index],
            "mean_position_mae": rank_mae,
            "tuning_temperature": selected_temperature,
            "events": len(blended),
        })
    table = pd.DataFrame(rows).sort_values(
        ["mean_winner_log_loss", "mean_position_mae"],
        kind="stable",
    ).reset_index(drop=True)
    if table.empty:
        raise ValueError("No ensemble weight candidates")
    return dict(table.iloc[0].weights), table
