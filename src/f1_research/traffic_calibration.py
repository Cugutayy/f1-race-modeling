"""Retrospective close-following prior from public OpenF1 timing.

The simulator needs an explicit cost for running close behind another car, especially
after a pit rejoin. Public timing cannot isolate dirty air, DRS, driver intent, tyre
state and overtaking difficulty causally, so this module estimates only a conservative
*net observed* lap-time delta. It is never used by the strict next-lap benchmark.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CLOSE_WINDOW_S = 1.20
DEFAULT_CLEAR_MIN_S = 3.0
DEFAULT_CLEAR_MAX_S = 8.0
DEFAULT_PENALTY_MEAN_S = 0.12
DEFAULT_PENALTY_SD_S = 0.05
INTERVAL_FRESHNESS_S = 15.0
MIN_BUCKET_LAPS = 3
MIN_CONTRASTS = 6


@dataclass(frozen=True)
class TrafficPriorModel:
    enabled: bool
    close_window_s: float
    clear_min_s: float
    clear_max_s: float
    raw_close_minus_clear_s: float
    penalty_mean_s: float
    penalty_sd_s: float
    driver_session_contrasts: int
    close_laps: int
    clear_laps: int
    source: str = "retrospective_openf1_intervals_and_laps"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Invalid raw provider file: {path}")
    return value


def _driver_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: str(int(value)) if np.isfinite(value) else "")


def _eligible_laps(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session_key", "driver_number", "forecast_at", "target_s", "target_valid",
        "recent_median_5_s", "lap_regime",
    }
    if required - set(frame):
        return pd.DataFrame()
    output = frame.copy()
    output["driver_key"] = _driver_key(output.driver_number)
    output["forecast_at"] = pd.to_datetime(output.forecast_at, utc=True, errors="coerce")
    output["target_s"] = pd.to_numeric(output.target_s, errors="coerce")
    output["recent_median_5_s"] = pd.to_numeric(output.recent_median_5_s, errors="coerce")
    mask = (
        output.target_valid.fillna(False).astype(bool)
        & output.lap_regime.astype(str).eq("green")
        & output.forecast_at.notna()
        & output.driver_key.ne("")
        & np.isfinite(output.target_s)
        & np.isfinite(output.recent_median_5_s)
    )
    if "safety_car_active" in output:
        mask &= pd.to_numeric(output.safety_car_active, errors="coerce").fillna(0).eq(0)
    if "is_pit_out_lap" in output:
        mask &= ~output.is_pit_out_lap.fillna(False).astype(bool)
    output = output.loc[mask].copy()
    if output.empty:
        return output
    output["pace_residual_s"] = output.target_s - output.recent_median_5_s
    return output[output.pace_residual_s.abs().le(3.0)].copy()


def _interval_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["driver_key", "interval_time", "interval_s"])
    frame = pd.DataFrame(rows)
    if "driver_number" not in frame or "date" not in frame or "interval" not in frame:
        return pd.DataFrame(columns=["driver_key", "interval_time", "interval_s"])
    frame["driver_key"] = _driver_key(frame.driver_number)
    frame["interval_time"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    frame["interval_s"] = pd.to_numeric(frame.interval, errors="coerce")
    frame = frame[
        frame.driver_key.ne("")
        & frame.interval_time.notna()
        & np.isfinite(frame.interval_s)
        & frame.interval_s.gt(0)
    ].copy()
    return frame.sort_values(["driver_key", "interval_time"])


def _attach_intervals(laps: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    if laps.empty or intervals.empty:
        return pd.DataFrame()
    pieces = []
    tolerance = pd.Timedelta(seconds=INTERVAL_FRESHNESS_S)
    for driver, left in laps.groupby("driver_key", sort=False):
        right = intervals[intervals.driver_key.eq(driver)]
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("forecast_at"),
            right[["interval_time", "interval_s"]].sort_values("interval_time"),
            left_on="forecast_at",
            right_on="interval_time",
            direction="backward",
            tolerance=tolerance,
        )
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).dropna(subset=["interval_s"])


def calibrate_traffic_prior(
    datasets: list[pd.DataFrame],
    raw_root: Path,
    session_keys: list[int],
    *,
    fallback_mean_s: float = DEFAULT_PENALTY_MEAN_S,
    fallback_sd_s: float = DEFAULT_PENALTY_SD_S,
    close_window_s: float = DEFAULT_CLOSE_WINDOW_S,
) -> tuple[TrafficPriorModel, dict[str, Any]]:
    """Estimate a pooled close-following penalty from within-driver/session contrasts."""
    if not 0 < close_window_s < DEFAULT_CLEAR_MIN_S:
        raise ValueError("close_window_s must be positive and below the clear-air threshold")
    by_session: dict[int, pd.DataFrame] = {}
    for frame in datasets:
        if frame.empty or "session_key" not in frame:
            continue
        keys = pd.to_numeric(frame.session_key, errors="coerce").dropna().unique()
        for key in keys:
            selected = frame[pd.to_numeric(frame.session_key, errors="coerce").eq(key)].copy()
            by_session[int(key)] = selected

    contrasts: list[float] = []
    close_laps = 0
    clear_laps = 0
    joined_laps = 0
    sessions_with_intervals = 0
    for session_key in session_keys:
        frame = by_session.get(int(session_key))
        if frame is None:
            continue
        laps = _eligible_laps(frame)
        intervals = _interval_frame(_read_rows(Path(raw_root) / str(session_key) / "intervals.json"))
        joined = _attach_intervals(laps, intervals)
        if joined.empty:
            continue
        sessions_with_intervals += 1
        joined_laps += len(joined)
        for _, group in joined.groupby("driver_key"):
            close = group[
                group.interval_s.gt(0.05) & group.interval_s.le(close_window_s)
            ].pace_residual_s.to_numpy(dtype=float)
            clear = group[
                group.interval_s.ge(DEFAULT_CLEAR_MIN_S)
                & group.interval_s.le(DEFAULT_CLEAR_MAX_S)
            ].pace_residual_s.to_numpy(dtype=float)
            close_laps += len(close)
            clear_laps += len(clear)
            if len(close) < MIN_BUCKET_LAPS or len(clear) < MIN_BUCKET_LAPS:
                continue
            contrasts.append(float(np.median(close) - np.median(clear)))

    raw_effect = float(np.median(contrasts)) if contrasts else float("nan")
    if len(contrasts) >= MIN_CONTRASTS:
        values = np.asarray(contrasts, dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = float(np.clip(max(0.02, 1.4826 * mad), 0.02, 0.35))
        # Public timing mixes dirty-air loss with DRS/tow benefits. The simulator only
        # models a conservative cost, so negative net effects become zero rather than
        # inventing a close-following speed boost.
        penalty_mean = float(np.clip(center, 0.0, 0.60))
        penalty_sd = scale
        enabled = True
        source = "within_driver_session_close_vs_clear"
    else:
        penalty_mean = float(fallback_mean_s)
        penalty_sd = float(fallback_sd_s)
        enabled = False
        source = "fallback_insufficient_interval_contrasts"

    model = TrafficPriorModel(
        enabled=enabled,
        close_window_s=float(close_window_s),
        clear_min_s=DEFAULT_CLEAR_MIN_S,
        clear_max_s=DEFAULT_CLEAR_MAX_S,
        raw_close_minus_clear_s=raw_effect,
        penalty_mean_s=penalty_mean,
        penalty_sd_s=penalty_sd,
        driver_session_contrasts=len(contrasts),
        close_laps=close_laps,
        clear_laps=clear_laps,
        source=source,
    )
    audit = {
        "enabled": enabled,
        "source": source,
        "sessions_with_intervals": sessions_with_intervals,
        "joined_laps": joined_laps,
        "driver_session_contrasts": len(contrasts),
        "raw_close_minus_clear_s": raw_effect if np.isfinite(raw_effect) else None,
        "method": (
            "as-of interval at lap-start (<=15s old); green-lap target minus prior-five-lap median; "
            "within-driver/session close (<=1.2s) minus clear (3-8s) robust contrasts"
        ),
        "limitations": [
            "Observed interval effects are not causal dirty-air estimates and can include DRS/tow/overtaking effects.",
            "Public timing does not expose aero balance, energy deployment, driver intent or team traffic forecasts.",
            "Negative observed close-following deltas are conservatively mapped to zero simulator penalty.",
            "This retrospective prior is not used by the strict event-time next-lap benchmark.",
        ],
    }
    return model, audit


def as_payload(model: TrafficPriorModel) -> dict[str, Any]:
    payload = asdict(model)
    if not np.isfinite(payload["raw_close_minus_clear_s"]):
        payload["raw_close_minus_clear_s"] = None
    return payload


def config_values_from_payload(
    payload: dict[str, Any] | None,
    *,
    fallback_window_s: float = DEFAULT_CLOSE_WINDOW_S,
    fallback_mean_s: float = DEFAULT_PENALTY_MEAN_S,
    fallback_sd_s: float = DEFAULT_PENALTY_SD_S,
) -> tuple[float, float, float]:
    if not isinstance(payload, dict):
        return fallback_window_s, fallback_mean_s, fallback_sd_s
    try:
        window = float(payload.get("close_window_s", fallback_window_s))
        mean = float(payload.get("penalty_mean_s", fallback_mean_s))
        sd = float(payload.get("penalty_sd_s", fallback_sd_s))
    except (TypeError, ValueError):
        return fallback_window_s, fallback_mean_s, fallback_sd_s
    if not np.isfinite(window) or not 0 < window <= 3.0:
        window = fallback_window_s
    if not np.isfinite(mean) or not 0 <= mean <= 1.0:
        mean = fallback_mean_s
    if not np.isfinite(sd) or not 0 <= sd <= 0.5:
        sd = fallback_sd_s
    return window, mean, sd
