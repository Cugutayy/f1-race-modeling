"""Retrospective public-data tyre priors for strategy counterfactuals.

These estimates are deliberately not described as physical tyre models. Public lap
and stint data mixes tyre behaviour with fuel burn, traffic, track evolution and
strategy selection. The output is therefore a conservative prior for simulation,
with explicit fallbacks and provenance, not causal tyre-performance evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
DEFAULT_PACE_DELTA_S = {
    "SOFT": -0.35,
    "MEDIUM": 0.0,
    "HARD": 0.35,
    "INTERMEDIATE": 2.5,
    "WET": 5.0,
}
DEFAULT_DEGRADATION_S_PER_LAP = {
    "SOFT": 0.08,
    "MEDIUM": 0.06,
    "HARD": 0.04,
    "INTERMEDIATE": 0.06,
    "WET": 0.05,
}
DEFAULT_STINT_TARGET_LAPS = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 40.0,
    "INTERMEDIATE": 25.0,
    "WET": 30.0,
}


@dataclass(frozen=True)
class TyrePrior:
    compound: str
    pace_delta_s: float
    degradation_s_per_lap: float
    stint_target_laps: float
    lap_observations: int
    completed_stints: int
    pace_source: str
    degradation_source: str
    stint_source: str


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Invalid raw provider file: {path}")
    return value


def _clean_laps(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for frame in datasets:
        required = {"session_key", "driver_number", "lap_number", "target_s", "target_valid", "compound", "tyre_age"}
        if frame.empty or required - set(frame):
            continue
        work = frame.copy()
        work["compound"] = work.compound.astype(str).str.upper()
        work["target_s"] = pd.to_numeric(work.target_s, errors="coerce")
        work["tyre_age"] = pd.to_numeric(work.tyre_age, errors="coerce")
        work["lap_number"] = pd.to_numeric(work.lap_number, errors="coerce")
        mask = work.target_valid.fillna(False).astype(bool)
        mask &= work.compound.isin(COMPOUNDS)
        mask &= work.target_s.between(55.0, 220.0)
        mask &= work.tyre_age.between(0.0, 80.0)
        mask &= work.lap_number.between(1.0, 100.0)
        if "lap_regime" in work:
            mask &= work.lap_regime.astype(str).eq("green")
        if "is_pit_out_lap" in work:
            mask &= ~work.is_pit_out_lap.fillna(False).astype(bool)
        if "safety_car_active" in work:
            mask &= pd.to_numeric(work.safety_car_active, errors="coerce").fillna(0).le(0)
        if "rainfall" in work:
            mask &= pd.to_numeric(work.rainfall, errors="coerce").fillna(0).le(0)
        work = work.loc[mask].copy()
        if work.empty:
            continue
        # Remove large within-driver/session excursions (traffic, missed pit labels,
        # yellow residue) before fitting small public-data priors.
        group_keys = ["session_key", "driver_number"]
        median = work.groupby(group_keys).target_s.transform("median")
        absolute = (work.target_s - median).abs()
        mad = absolute.groupby([work.session_key, work.driver_number]).transform("median")
        threshold = np.maximum(2.5, 4.5 * 1.4826 * mad.fillna(0).to_numpy(dtype=float))
        work = work.loc[absolute.to_numpy(dtype=float) <= threshold]
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fit_dry_compound_model(frame: pd.DataFrame, alpha: float = 12.0) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    dry = frame[frame.compound.isin(DRY_COMPOUNDS)].copy()
    counts = dry.compound.value_counts().to_dict()
    if len(dry) < 120 or sum(int(counts.get(c, 0) >= 20) for c in DRY_COMPOUNDS) < 2:
        return dict(DEFAULT_PACE_DELTA_S), dict(DEFAULT_DEGRADATION_S_PER_LAP), {
            "source": "fallback_insufficient_green_laps",
            "rows": int(len(dry)),
            "alpha": alpha,
        }

    dry["driver_session"] = dry.session_key.astype(str) + "::" + dry.driver_number.astype(str)
    session_center = dry.groupby("session_key").lap_number.transform("median")
    features = pd.DataFrame(index=dry.index)
    features["lap_centered"] = dry.lap_number.to_numpy(dtype=float) - session_center.to_numpy(dtype=float)
    for compound in DRY_COMPOUNDS:
        features[f"compound_{compound}"] = dry.compound.eq(compound).astype(float)
        features[f"age_{compound}"] = np.where(
            dry.compound.eq(compound), dry.tyre_age.to_numpy(dtype=float), 0.0
        )
    fixed = pd.get_dummies(dry.driver_session, prefix="fe", dtype=float)
    design = pd.concat([features, fixed], axis=1)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, dry.target_s.to_numpy(dtype=float))
    coefficients = dict(zip(design.columns, model.coef_, strict=True))

    raw_pace = {compound: float(coefficients.get(f"compound_{compound}", 0.0)) for compound in DRY_COMPOUNDS}
    medium = raw_pace["MEDIUM"]
    pace = dict(DEFAULT_PACE_DELTA_S)
    degradation = dict(DEFAULT_DEGRADATION_S_PER_LAP)
    for compound in DRY_COMPOUNDS:
        if int(counts.get(compound, 0)) >= 20:
            pace[compound] = float(np.clip(raw_pace[compound] - medium, -3.0, 3.0))
            # This is a net lap-time/tyre-age trend after a linear lap-number control,
            # not isolated rubber degradation. Negative estimates are clipped to zero
            # because the simulator uses this term only for future ageing penalties.
            degradation[compound] = float(np.clip(coefficients.get(f"age_{compound}", 0.0), 0.0, 0.25))
    pace["MEDIUM"] = 0.0
    return pace, degradation, {
        "source": "ridge_driver_session_fixed_effects_with_lap_number_control",
        "rows": int(len(dry)),
        "alpha": alpha,
        "lap_counts": {compound: int(counts.get(compound, 0)) for compound in DRY_COMPOUNDS},
        "lap_number_coefficient": float(coefficients.get("lap_centered", 0.0)),
    }


def _completed_stint_lengths(raw_root: Path, session_keys: list[int]) -> dict[str, list[int]]:
    output = {compound: [] for compound in COMPOUNDS}
    for session_key in session_keys:
        stints = pd.DataFrame(_read_rows(Path(raw_root) / str(session_key) / "stints.json"))
        if stints.empty or "driver_number" not in stints or "lap_start" not in stints:
            continue
        for column in ("driver_number", "stint_number", "lap_start", "lap_end"):
            if column in stints:
                stints[column] = pd.to_numeric(stints[column], errors="coerce")
        stints["compound"] = stints.get("compound", pd.Series("UNKNOWN", index=stints.index)).astype(str).str.upper()
        stints = stints.dropna(subset=["driver_number", "lap_start"]).sort_values(
            ["driver_number", "stint_number", "lap_start"], na_position="last"
        )
        for _, group in stints.groupby("driver_number", sort=False):
            ordered = group.sort_values(["stint_number", "lap_start"], na_position="last")
            # The final stint is censored by the chequered flag or retirement and does
            # not prove the tyre would have been deliberately stopped at that length.
            for row in ordered.iloc[:-1].itertuples(index=False):
                compound = str(getattr(row, "compound", "UNKNOWN")).upper()
                start = getattr(row, "lap_start", np.nan)
                end = getattr(row, "lap_end", np.nan)
                if compound not in output or not np.isfinite(start) or not np.isfinite(end):
                    continue
                duration = int(end - start + 1)
                if 2 <= duration <= 80:
                    output[compound].append(duration)
    return output


def calibrate_tyre_priors(
    datasets: list[pd.DataFrame],
    raw_root: Path,
    session_keys: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    laps = _clean_laps(datasets)
    pace, degradation, fit_audit = _fit_dry_compound_model(laps)
    stint_lengths = _completed_stint_lengths(raw_root, session_keys)

    priors: list[TyrePrior] = []
    counts = laps.compound.value_counts().to_dict() if not laps.empty else {}
    for compound in COMPOUNDS:
        lengths = stint_lengths[compound]
        if len(lengths) >= 5:
            stint_target = float(np.clip(np.quantile(lengths, 0.75), 5.0, 60.0))
            stint_source = "completed_pre_pit_stint_p75"
        else:
            stint_target = float(DEFAULT_STINT_TARGET_LAPS[compound])
            stint_source = "fallback_insufficient_completed_stints"
        lap_count = int(counts.get(compound, 0))
        dry_learned = compound in DRY_COMPOUNDS and fit_audit["source"].startswith("ridge") and lap_count >= 20
        priors.append(TyrePrior(
            compound=compound,
            pace_delta_s=float(pace[compound]),
            degradation_s_per_lap=float(degradation[compound]),
            stint_target_laps=stint_target,
            lap_observations=lap_count,
            completed_stints=len(lengths),
            pace_source="retrospective_green_lap_fixed_effects" if dry_learned else "fallback_default",
            degradation_source="retrospective_net_age_trend" if dry_learned else "fallback_default",
            stint_source=stint_source,
        ))

    payload = {
        "schema_version": 1,
        "compounds": {item.compound: asdict(item) for item in priors},
    }
    audit = {
        "fit": fit_audit,
        "sessions": [int(key) for key in session_keys],
        "definition": (
            "retrospective green-lap compound offsets/net age trends plus the 75th percentile "
            "of non-final stints that ended in a subsequent stint"
        ),
        "limitations": [
            "Compound offsets are observational and remain confounded by fuel, traffic and strategy selection.",
            "The lap-number control is linear and cannot isolate all track-evolution effects.",
            "Net tyre-age trend is clipped nonnegative for simulation and is not a physical degradation estimate.",
            "Observed stint target is a historical pit-decision prior, not maximum safe tyre life.",
            "Historical stint metadata is retrospective; it is used for completed-race calibration, not strict next-lap evidence.",
        ],
    }
    return payload, audit


def maps_from_payload(payload: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    pace = dict(DEFAULT_PACE_DELTA_S)
    degradation = dict(DEFAULT_DEGRADATION_S_PER_LAP)
    stint = dict(DEFAULT_STINT_TARGET_LAPS)
    compounds = payload.get("compounds") if isinstance(payload, dict) else None
    if not isinstance(compounds, dict):
        return pace, degradation, stint
    for compound in COMPOUNDS:
        row = compounds.get(compound)
        if not isinstance(row, dict):
            continue
        try:
            pace_value = float(row["pace_delta_s"])
            deg_value = float(row["degradation_s_per_lap"])
            stint_value = float(row["stint_target_laps"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(pace_value) and -5 <= pace_value <= 5:
            pace[compound] = pace_value
        if np.isfinite(deg_value) and 0 <= deg_value <= 0.5:
            degradation[compound] = deg_value
        if np.isfinite(stint_value) and 2 <= stint_value <= 80:
            stint[compound] = stint_value
    pace["MEDIUM"] = 0.0
    return pace, degradation, stint
