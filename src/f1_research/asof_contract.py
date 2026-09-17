"""Fail-closed event-time contracts used before any model sees a feature row."""
from __future__ import annotations

import pandas as pd


def enforce_asof(frame: pd.DataFrame, *, cutoff_at, timestamp_columns: dict[str, str]) -> pd.DataFrame:
    """Reject feature observations whose information timestamp exceeds forecast cutoff.

    timestamp_columns maps feature name -> column containing when that feature became
    observable. It deliberately does not infer publication time from event time.
    """
    if frame.empty:
        raise ValueError("feature frame is empty")
    cutoff = pd.Timestamp(cutoff_at)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_at must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    checked = frame.copy()
    for feature, column in timestamp_columns.items():
        if feature not in checked:
            raise ValueError(f"declared feature is missing: {feature}")
        if column not in checked:
            raise ValueError(f"feature {feature} has no information timestamp column {column}")
        observed = pd.to_datetime(checked[column], utc=True, errors="coerce")
        if observed.isna().any():
            raise ValueError(f"feature {feature} contains unknown information timestamps")
        leaked = observed > cutoff
        if leaked.any():
            raise ValueError(
                f"feature {feature} leaks {int(leaked.sum())} observations after cutoff {cutoff.isoformat()}"
            )
    return checked


def assert_target_unavailable(frame: pd.DataFrame, target_columns=("finish_position", "dnf", "points")) -> None:
    """Inference rows must not carry target-race outcomes, even accidentally."""
    for column in target_columns:
        if column in frame and frame[column].notna().any():
            raise ValueError(f"inference frame contains target-race outcome: {column}")
