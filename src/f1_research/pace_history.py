"""Shared event-time pace-history eligibility policy for historical and live paths."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

MIN_SLOW_MARGIN_S = 10.0
MAX_SLOW_RELATIVE_MARGIN = 0.25
NEUTRALIZED_TRACK_STATES = frozenset({"YELLOW", "VSC", "SC", "RED", "RESTART"})


def rain_state(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number > 0.0


def same_rain_regime(previous: bool | None, current: bool | None) -> bool:
    """Unknown rain state does not manufacture a regime transition."""
    return previous is None or current is None or previous == current


def pace_duration_is_plausible(
    duration_s: float,
    previous_durations_s: Iterable[float],
) -> bool:
    """Reject only extreme *slow* outliers using already-observed clean pace.

    The asymmetric guard is deliberate: neutralization/restart/pit contamination is
    normally much slower than green pace. Faster observations are retained so the
    model can adapt to fuel burn, track evolution and drying conditions.
    """
    try:
        duration = float(duration_s)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(duration) or duration <= 0:
        return False

    history = []
    for value in previous_durations_s:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            history.append(number)
    if not history:
        return True

    reference = float(median(history[-5:]))
    slow_margin = max(MIN_SLOW_MARGIN_S, MAX_SLOW_RELATIVE_MARGIN * reference)
    return duration <= reference + slow_margin


def track_state_is_pace_eligible(track_state: Any) -> bool:
    state = str(track_state or "UNKNOWN").strip().upper()
    return state not in NEUTRALIZED_TRACK_STATES
