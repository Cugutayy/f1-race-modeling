"""Statistical diagnostics for sealed F1 benchmark predictions.

These functions are evidence-only: they never refit or select a forecasting model.
Race/event is the dependence unit for paired tests. Driver-level winner calibration is
reported descriptively and explicitly does not treat driver rows as independent trials.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def paired_randomization_test(
    differences: np.ndarray,
    *,
    direction: str,
    samples: int = 20_000,
    seed: int = 42,
    exact_max_events: int = 16,
) -> dict[str, Any]:
    """Paired sign-flip test on whole-race metric differences.

    ``differences`` are model-minus-baseline for the same held-out races. The null is
    exchangeability of the sign of each paired difference. This does not assume driver
    rows are independent.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite one-dimensional array with >=2 events")
    if direction not in {"lower", "higher"}:
        raise ValueError("direction must be 'lower' or 'higher'")
    if samples < 100:
        raise ValueError("samples must be >=100")

    observed = float(values.mean())
    if len(values) <= exact_max_events:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
        draws = (signs * values[None, :]).mean(axis=1)
        method = "exact_sign_flip"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(values)))
        draws = (signs * values[None, :]).mean(axis=1)
        method = "monte_carlo_sign_flip"

    eps = 1e-15
    two_sided = float(np.mean(np.abs(draws) >= abs(observed) - eps))
    if direction == "lower":
        one_sided = float(np.mean(draws <= observed + eps))
        favorable = observed < 0
    else:
        one_sided = float(np.mean(draws >= observed - eps))
        favorable = observed > 0
    return {
        "events": int(len(values)),
        "observed_mean_difference": observed,
        "direction": direction,
        "observed_favors_model": bool(favorable),
        "p_two_sided": two_sided,
        "p_one_sided_favorable": one_sided,
        "method": method,
        "null_draws": int(len(draws)),
    }


def moving_block_bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int = 10_000,
    block_length: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Circular moving-block bootstrap for chronologically ordered event statistics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional array with >=2 events")
    if samples < 100:
        raise ValueError("samples must be >=100")
    if not 1 <= block_length <= len(values):
        raise ValueError("block_length must be between 1 and the event count")

    rng = np.random.default_rng(seed)
    n = len(values)
    blocks_needed = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(samples, blocks_needed))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(samples, -1)[:, :n]
    return values[indices].mean(axis=1)


def _reliability_bins(probability: np.ndarray, outcome: np.ndarray, bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = np.digitize(probability, edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = labels == index
        if not np.any(mask):
            continue
        rows.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()),
            "observed_frequency": float(outcome[mask].mean()),
        })
    return rows


def _adaptive_bins(probability: np.ndarray, outcome: np.ndarray, bins: int) -> list[dict[str, Any]]:
    order = np.argsort(probability, kind="stable")
    chunks = np.array_split(order, min(bins, len(order)))
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        rows.append({
            "bin": int(index),
            "count": int(len(chunk)),
            "min_probability": float(probability[chunk].min()),
            "max_probability": float(probability[chunk].max()),
            "mean_probability": float(probability[chunk].mean()),
            "observed_frequency": float(outcome[chunk].mean()),
        })
    return rows


def _ece(rows: list[dict[str, Any]], total: int) -> tuple[float, float]:
    weighted = []
    errors = []
    for row in rows:
        error = abs(float(row["mean_probability"]) - float(row["observed_frequency"]))
        errors.append(error)
        weighted.append(int(row["count"]) / total * error)
    return float(sum(weighted)), float(max(errors, default=0.0))


def _brier_decomposition(
    probability: np.ndarray,
    outcome: np.ndarray,
    reliability_rows: list[dict[str, Any]],
) -> dict[str, float]:
    n = len(probability)
    base_rate = float(outcome.mean())
    reliability = 0.0
    resolution = 0.0
    for row in reliability_rows:
        weight = int(row["count"]) / n
        p_bar = float(row["mean_probability"])
        y_bar = float(row["observed_frequency"])
        reliability += weight * (p_bar - y_bar) ** 2
        resolution += weight * (y_bar - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((probability - outcome) ** 2))
    return {
        "brier_binary_driver_row": brier,
        "reliability_binned": float(reliability),
        "resolution_binned": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstructed_brier_binned": float(reliability - resolution + uncertainty),
    }


def _calibration_logistic(probability: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))

    def loss(theta: np.ndarray) -> float:
        intercept, slope = theta
        fitted = np.clip(expit(intercept + slope * logit), 1e-12, 1 - 1e-12)
        return float(-np.sum(outcome * np.log(fitted) + (1 - outcome) * np.log(1 - fitted)))

    result = minimize(loss, np.asarray([0.0, 1.0]), method="L-BFGS-B")
    return {
        "intercept": float(result.x[0]) if result.success else None,
        "slope": float(result.x[1]) if result.success else None,
        "converged": bool(result.success),
        "optimizer_message": str(result.message),
        "interpretation": "descriptive one-vs-rest winner calibration; driver rows within a race are dependent",
    }


def winner_calibration_diagnostics(
    predictions: pd.DataFrame,
    *,
    bins: int = 10,
    probability_column: str = "win_probability",
) -> dict[str, Any]:
    """Calibration diagnostics from sealed driver-level winner probabilities."""
    required = {"event_id", "model", "actual_position", probability_column}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if bins < 3:
        raise ValueError("bins must be >=3")

    output: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sealed_winner_calibration_diagnostics",
        "probability_column": probability_column,
        "models": {},
        "limitations": [
            "Driver rows within the same race are dependent; diagnostics are descriptive, not iid inference.",
            "Reliability and Brier decomposition depend on binning.",
            "Calibration slope/intercept are one-vs-rest summaries of a one-winner categorical distribution.",
        ],
    }

    for model, frame in predictions.groupby("model", sort=True):
        frame = frame.copy()
        probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(dtype=float)
        outcome = (pd.to_numeric(frame["actual_position"], errors="coerce").to_numpy(dtype=float) == 1).astype(float)
        if len(probability) == 0 or not np.isfinite(probability).all():
            raise ValueError(f"Model {model} contains non-finite winner probabilities")
        if np.any((probability < -1e-12) | (probability > 1 + 1e-12)):
            raise ValueError(f"Model {model} contains probabilities outside [0, 1]")

        event_sums = frame.assign(_p=probability).groupby("event_id", sort=False)["_p"].sum()
        winner_counts = frame.assign(_y=outcome).groupby("event_id", sort=False)["_y"].sum()
        if not np.allclose(event_sums.to_numpy(dtype=float), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"Model {model} winner probabilities do not sum to one per event")
        if not np.allclose(winner_counts.to_numpy(dtype=float), 1.0, atol=0.0, rtol=0.0):
            raise ValueError(f"Model {model} does not have exactly one winner per event")

        reliability = _reliability_bins(probability, outcome, bins)
        adaptive = _adaptive_bins(probability, outcome, bins)
        ece, mce = _ece(reliability, len(probability))
        adaptive_ece, _ = _ece(adaptive, len(probability))

        top_rows = []
        for event_id, event in frame.assign(_p=probability, _y=outcome).groupby("event_id", sort=False):
            chosen = event.loc[event["_p"].idxmax()]
            top_rows.append({
                "event_id": event_id,
                "confidence": float(chosen["_p"]),
                "correct": float(chosen["_y"]),
            })
        top = pd.DataFrame(top_rows)

        output["models"][str(model)] = {
            "events": int(frame["event_id"].nunique()),
            "driver_rows": int(len(frame)),
            "event_probability_sum_max_abs_error": float(np.max(np.abs(event_sums.to_numpy() - 1.0))),
            "ece_equal_width": ece,
            "mce_equal_width": mce,
            "ece_adaptive": adaptive_ece,
            "reliability_bins": reliability,
            "adaptive_bins": adaptive,
            "brier_decomposition": _brier_decomposition(probability, outcome, reliability),
            "calibration_logistic": _calibration_logistic(probability, outcome),
            "top_choice": {
                "accuracy": float(top["correct"].mean()),
                "mean_confidence": float(top["confidence"].mean()),
                "confidence_minus_accuracy": float(top["confidence"].mean() - top["correct"].mean()),
            },
        }
    return output
