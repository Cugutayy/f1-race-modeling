"""FIA-style completed-lap classification helpers for race Monte Carlo outputs.

These helpers model only the published completed-lap ordering rule. They do not infer
steward decisions, post-race penalties or classifications that require non-public
race-control judgement.
"""

from __future__ import annotations

import numpy as np

FIA_CLASSIFIED_FRACTION = 0.90


def minimum_classified_laps(winner_laps: np.ndarray | int) -> np.ndarray:
    """Return the B2.5.5 90% completed-lap threshold, rounded down."""
    values = np.asarray(winner_laps, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("winner_laps must be finite and non-negative")
    if np.any(values != np.floor(values)):
        raise ValueError("winner_laps must contain whole completed laps")
    return np.floor(FIA_CLASSIFIED_FRACTION * values).astype(int)


def classify_completed_laps(
    completed_laps: np.ndarray,
    last_completed_line_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank by completed laps, then last completed Line crossing time.

    Returns ranks, classified flags and thresholds. Cars below the FIA 90%
    threshold remain ordered for Monte Carlo bookkeeping but callers must not
    present that numeric order as an official classified position.
    """
    laps = np.asarray(completed_laps)
    times = np.asarray(last_completed_line_time_s, dtype=float)
    if laps.ndim != 2 or times.ndim != 2 or laps.shape != times.shape:
        raise ValueError("completed_laps and crossing times must be same-shaped 2D arrays")
    if laps.shape[1] < 2:
        raise ValueError("at least two cars are required for classification")
    if np.issubdtype(laps.dtype, np.floating):
        if np.any(~np.isfinite(laps)) or np.any(laps != np.floor(laps)):
            raise ValueError("completed_laps must contain finite whole numbers")
    laps = laps.astype(int, copy=False)
    if np.any(laps < 0):
        raise ValueError("completed_laps cannot be negative")
    if np.any(~np.isfinite(times)):
        raise ValueError("crossing times must be finite")

    # numpy.lexsort uses the last key as primary: completed laps descending first,
    # then last completed Line crossing time ascending for equal lap counts.
    orders = np.lexsort((times, -laps), axis=1)
    ranks = np.empty_like(orders)
    np.put_along_axis(
        ranks,
        orders,
        np.arange(1, laps.shape[1] + 1, dtype=int)[None, :],
        axis=1,
    )

    winner_laps = np.max(laps, axis=1)
    thresholds = minimum_classified_laps(winner_laps)
    classified = laps >= thresholds[:, None]
    return ranks, classified, thresholds
