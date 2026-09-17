"""Retrospective public-timing tyre priors for strategy simulation.

These estimates are deliberately *not* physical tyre models. OpenF1 historical stint
metadata lacks publication timestamps and public timing has no fuel load, tyre energy,
car setup or internal temperatures. The calibration therefore estimates only:

* compound pace offsets relative to the same driver's other stints after subtracting
  same-race/same-lap field pace;
* relative degradation slopes after subtracting same-race/same-lap field pace; and
* observed tyre age at genuine pit-ended stints.

The strict next-lap benchmark never consumes these retrospective features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
ALL_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
DEFAULT_PACE_DELTA = {
    "SOFT": -0.35,
    "MEDIUM": 0.0,
    "HARD": 0.35,
    "INTERMEDIATE": 2.5,
    "WET": 5.0,
}
DEFAULT_DEGRADATION = {
    "SOFT": 0.08,
    "MEDIUM": 0.06,
    "HARD": 0.04,
    "INTERMEDIATE": 0.06,
    "WET": 0.05,
}
DEFAULT_PIT_AGE = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 40.0,
    "INTERMEDIATE": 25.0,
    "WET": 30.0,
}
MIN_FIELD_CARS = 3
MIN_STINT_LAPS = 4
MIN_COMPOUND_STINTS = 5
_STINT_COLUMNS = [
    "session_key",
    "driver_number",
    "stint_number",
    "compound",
    "relative_slope",
    "reference_residual_s",
    "laps",
    "max_tyre_age",
    "max_lap_number",
]


@dataclass(frozen=True)
class TyrePriorModel:
    enabled: bool
    pace_delta_s: dict[str, float]
    relative_degradation_s_per_lap: dict[str, float]
    degradation_scale_s_per_lap: dict[str, float]
    observed_pit_age_p50: dict[str, float]
    observations: dict[str, dict[str, int]]
    source: str = "retrospective_openf1_public_timing"


def _clean_rows(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    required = {
        "session_key", "driver_number", "lap_number", "target_s", "target_valid",
        "lap_regime", "compound", "tyre_age", "stint_number",
    }
    frames = []
    for frame in datasets:
        if required - set(frame):
            continue
        selected = frame.copy()
        selected["compound"] = selected.compound.astype(str).str.upper()
        selected["target_s"] = pd.to_numeric(selected.target_s, errors="coerce")
        selected["tyre_age"] = pd.to_numeric(selected.tyre_age, errors="coerce")
        selected["lap_number"] = pd.to_numeric(selected.lap_number, errors="coerce")
        selected["stint_number"] = pd.to_numeric(selected.stint_number, errors="coerce")
        mask = (
            selected.target_valid.fillna(False).astype(bool)
            & selected.lap_regime.astype(str).eq("green")
            & selected.compound.isin(DRY_COMPOUNDS)
            & np.isfinite(selected.target_s)
            & selected.target_s.between(40.0, 300.0)
            & np.isfinite(selected.tyre_age)
            & selected.tyre_age.between(0.0, 70.0)
            & np.isfinite(selected.lap_number)
            & np.isfinite(selected.stint_number)
        )
        if "safety_car_active" in selected:
            mask &= pd.to_numeric(selected.safety_car_active, errors="coerce").fillna(0).eq(0)
        if "is_pit_out_lap" in selected:
            mask &= ~selected.is_pit_out_lap.fillna(False).astype(bool)
        selected = selected.loc[mask].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=sorted(required))
    return pd.concat(frames, ignore_index=True)


def _field_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        output = frame.copy()
        output["field_count"] = pd.Series(dtype=float)
        output["field_median_s"] = pd.Series(dtype=float)
        output["field_residual_s"] = pd.Series(dtype=float)
        return output
    output = frame.copy()
    grouped = output.groupby(["session_key", "lap_number"])["target_s"]
    output["field_count"] = grouped.transform("count")
    output["field_median_s"] = grouped.transform("median")
    output = output[output.field_count.ge(MIN_FIELD_CARS)].copy()
    output["field_residual_s"] = output.target_s - output.field_median_s
    return output


def _theil_sen(age: np.ndarray, residual: np.ndarray) -> float | None:
    age = np.asarray(age, dtype=float)
    residual = np.asarray(residual, dtype=float)
    slopes = []
    for i in range(len(age)):
        for j in range(i + 1, len(age)):
            delta_age = age[j] - age[i]
            if delta_age <= 0:
                continue
            slopes.append((residual[j] - residual[i]) / delta_age)
    if not slopes:
        return None
    return float(np.median(slopes))


def _stint_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_STINT_COLUMNS)
    rows: list[dict[str, Any]] = []
    keys = ["session_key", "driver_number", "stint_number", "compound"]
    for key, group in frame.groupby(keys, sort=True):
        group = group.sort_values("tyre_age")
        if len(group) < MIN_STINT_LAPS or group.tyre_age.max() - group.tyre_age.min() < 3:
            continue
        age = group.tyre_age.to_numpy(dtype=float)
        residual = group.field_residual_s.to_numpy(dtype=float)
        slope = _theil_sen(age, residual)
        if slope is None or not np.isfinite(slope):
            continue
        slope = float(np.clip(slope, -0.20, 0.50))
        reference_age = 3.0
        intercepts = residual - slope * (age - reference_age)
        session_key, driver, stint, compound = key
        rows.append({
            "session_key": int(session_key),
            "driver_number": str(driver),
            "stint_number": float(stint),
            "compound": str(compound),
            "relative_slope": slope,
            "reference_residual_s": float(np.median(intercepts)),
            "laps": len(group),
            "max_tyre_age": float(group.tyre_age.max()),
            "max_lap_number": float(group.lap_number.max()),
        })
    return pd.DataFrame(rows, columns=_STINT_COLUMNS)


def _compound_pace(stints: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, list[float]] = {compound: [] for compound in DRY_COMPOUNDS}
    if stints.empty:
        return {}, {compound: 0 for compound in DRY_COMPOUNDS}
    # Only within-driver/session comparisons remove most car/driver performance bias.
    for _, group in stints.groupby(["session_key", "driver_number"]):
        if group.compound.nunique() < 2:
            continue
        center = float(group.reference_residual_s.median())
        for row in group.itertuples(index=False):
            values[str(row.compound)].append(float(row.reference_residual_s - center))
    raw = {
        compound: float(np.median(samples))
        for compound, samples in values.items()
        if len(samples) >= MIN_COMPOUND_STINTS
    }
    if "MEDIUM" in raw:
        medium = raw["MEDIUM"]
        raw = {
            compound: float(np.clip(value - medium, -2.0, 2.0))
            for compound, value in raw.items()
        }
    elif raw:
        center = float(np.median(list(raw.values())))
        raw = {
            compound: float(np.clip(value - center, -2.0, 2.0))
            for compound, value in raw.items()
        }
    return raw, {compound: len(samples) for compound, samples in values.items()}


def _compound_degradation(stints: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    median, scale, counts = {}, {}, {}
    if stints.empty:
        return median, scale, {compound: 0 for compound in DRY_COMPOUNDS}
    for compound in DRY_COMPOUNDS:
        values = stints.loc[stints.compound.eq(compound), "relative_slope"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        counts[compound] = int(len(values))
        if len(values) < MIN_COMPOUND_STINTS:
            continue
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        median[compound] = float(np.clip(center, -0.05, 0.25))
        scale[compound] = float(np.clip(max(0.01, 1.4826 * mad), 0.01, 0.30))
    return median, scale, counts


def _pit_ages(stints: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    """Observed age for stints followed by another stint for the same driver/session."""
    samples: dict[str, list[float]] = {compound: [] for compound in DRY_COMPOUNDS}
    if stints.empty:
        return {}, {compound: 0 for compound in DRY_COMPOUNDS}
    for _, group in stints.groupby(["session_key", "driver_number"]):
        ordered = group.sort_values("stint_number")
        if len(ordered) < 2:
            continue
        for row in ordered.iloc[:-1].itertuples(index=False):
            samples[str(row.compound)].append(float(row.max_tyre_age))
    p50 = {
        compound: float(np.clip(np.median(values), 5.0, 60.0))
        for compound, values in samples.items()
        if len(values) >= MIN_COMPOUND_STINTS
    }
    return p50, {compound: len(values) for compound, values in samples.items()}


def calibrate_tyre_priors(datasets: list[pd.DataFrame]) -> tuple[TyrePriorModel, dict[str, Any]]:
    clean = _field_residuals(_clean_rows(datasets))
    stints = _stint_estimates(clean)
    pace, pace_counts = _compound_pace(stints)
    degradation, degradation_scale, degradation_counts = _compound_degradation(stints)
    pit_age, pit_counts = _pit_ages(stints)

    pace_delta = dict(DEFAULT_PACE_DELTA)
    pace_delta.update(pace)
    degradation_full = dict(DEFAULT_DEGRADATION)
    degradation_full.update(degradation)
    observed_pit_age = dict(DEFAULT_PIT_AGE)
    observed_pit_age.update(pit_age)
    model = TyrePriorModel(
        enabled=bool(pace or degradation or pit_age),
        pace_delta_s=pace_delta,
        relative_degradation_s_per_lap=degradation_full,
        degradation_scale_s_per_lap=degradation_scale,
        observed_pit_age_p50=observed_pit_age,
        observations={
            compound: {
                "pace_comparisons": int(pace_counts.get(compound, 0)),
                "degradation_stints": int(degradation_counts.get(compound, 0)),
                "pit_ended_stints": int(pit_counts.get(compound, 0)),
            }
            for compound in DRY_COMPOUNDS
        },
    )
    audit = {
        "enabled": model.enabled,
        "clean_green_laps": int(len(clean)),
        "eligible_stints": int(len(stints)),
        "method": (
            "same-session/lap field-median residuals; Theil-Sen-like within-stint slopes; "
            "within-driver/session compound centering"
        ),
        "source": model.source,
        "limitations": [
            "Relative degradation is not isolated physical tyre wear; traffic and strategy can remain.",
            "Observed pit age describes historical pit-ended stints, not a physical tyre-life limit.",
            "Compound metadata is retrospective because historical stint publication time is unavailable.",
            "INTERMEDIATE/WET retain explicit fallback values unless a separate wet calibration is built.",
        ],
    }
    return model, audit


def as_payload(model: TyrePriorModel) -> dict[str, Any]:
    return asdict(model)


def maps_from_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return validated simulator maps with backward-compatible fallbacks."""
    pace = dict(DEFAULT_PACE_DELTA)
    degradation = dict(DEFAULT_DEGRADATION)
    pit_age = dict(DEFAULT_PIT_AGE)
    if not isinstance(payload, dict):
        return pace, degradation, pit_age

    sources = (
        ("pace_delta_s", pace, -5.0, 5.0),
        ("relative_degradation_s_per_lap", degradation, 0.0, 0.5),
        ("observed_pit_age_p50", pit_age, 2.0, 80.0),
    )
    for field, target, lower, upper in sources:
        values = payload.get(field)
        if not isinstance(values, dict):
            continue
        for compound in ALL_COMPOUNDS:
            try:
                value = float(values[compound])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value) and lower <= value <= upper:
                target[compound] = value
    pace["MEDIUM"] = 0.0
    return pace, degradation, pit_age
