"""Empirical public-data priors for the transparent strategy simulator.

Only quantities supported by captured historical sources are calibrated here.
Team-only fuel/setup/tyre-temperature information is never inferred. Reliability is
estimated as a shrunk public-results survival prior, not a diagnosis of an individual
car's current mechanical state. Tyre and traffic priors are retrospective public-timing
estimates, not physical tyre/aero measurements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reliability import as_payload as reliability_payload
from .reliability import calibrate_reliability, records_from_raw
from .strategy import SimulationConfig
from .traffic_calibration import as_payload as traffic_payload
from .traffic_calibration import calibrate_traffic_prior, config_values_from_payload
from .tyre_calibration import as_payload as tyre_payload
from .tyre_calibration import calibrate_tyre_priors, maps_from_payload
from .value_parsing import strict_optional_bool


@dataclass(frozen=True)
class StrategyPriors:
    pit_loss_mean_s: float
    pit_loss_sd_s: float
    safety_car_hazard_per_lap: float
    dnf_hazard_per_lap: float
    pit_observations: int
    safety_car_starts: int
    race_laps_observed: int
    dnf_events: int
    car_laps_observed: int
    sessions: tuple[int, ...]
    dnf_source: str


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Invalid raw provider file: {path}")
    return value


def _robust_location_scale(
    values: list[float], fallback_mean: float, fallback_sd: float
) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(clean) < 5:
        return fallback_mean, fallback_sd
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = max(0.5, 1.4826 * mad)
    inlier = clean[np.abs(clean - median) <= max(4.0, 4.5 * scale)]
    if len(inlier) >= 5:
        median = float(np.median(inlier))
        mad = float(np.median(np.abs(inlier - median)))
        scale = max(0.5, 1.4826 * mad)
    return median, scale


def _pit_excess(datasets: list[pd.DataFrame]) -> list[float]:
    values = []
    for frame in datasets:
        if "lap_regime" not in frame:
            continue
        mask = frame.lap_regime.astype(str).eq("pit") & frame.target_valid
        if "is_pit_out_lap" in frame:
            mask &= ~frame.is_pit_out_lap.fillna(False).astype(bool)
        selected = frame.loc[mask, ["target_s", "recent_median_5_s"]].dropna()
        excess = (
            selected.target_s.to_numpy(dtype=float)
            - selected.recent_median_5_s.to_numpy(dtype=float)
        )
        values.extend(excess[(excess >= 5.0) & (excess <= 60.0)].tolist())
    return values


def _sc_start_count(rows: list[dict[str, Any]]) -> int:
    seen: set[tuple[Any, Any, str]] = set()
    for row in rows:
        category = str(row.get("category") or "").upper()
        message = str(row.get("message") or "").upper()
        if category != "SAFETYCAR" and "SAFETY CAR" not in message and "VSC" not in message:
            continue
        if any(token in message for token in ("ENDING", "IN THIS LAP", "WITHDRAWN", "ENDED", "CLEAR")):
            continue
        if not any(token in message for token in ("DEPLOY", "VIRTUAL SAFETY CAR", "VSC")):
            continue
        seen.add((row.get("lap_number"), row.get("date"), message))
    return len(seen)


def _dnf_exposure(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return DNF failures and approximate car-lap risk intervals."""
    failures = 0
    exposure = 0
    for row in rows:
        dns = strict_optional_bool(row.get("dns"), field="openf1.session_result.dns")
        dsq = strict_optional_bool(row.get("dsq"), field="openf1.session_result.dsq")
        dnf = strict_optional_bool(row.get("dnf"), field="openf1.session_result.dnf")
        if dns is None or dsq is None or dnf is None:
            continue
        if dns or dsq:
            continue
        try:
            laps = int(row.get("number_of_laps"))
        except (TypeError, ValueError):
            continue
        if laps < 0:
            continue
        failed = dnf
        failures += int(failed)
        exposure += laps + int(failed)
    return failures, exposure


def calibrate_strategy_priors(
    datasets: list[pd.DataFrame],
    raw_root: Path,
    session_keys: list[int],
    fallback: SimulationConfig | None = None,
) -> tuple[StrategyPriors, dict[str, Any]]:
    fallback = fallback or SimulationConfig()
    pit_values = _pit_excess(datasets)
    pit_mean, pit_sd = _robust_location_scale(
        pit_values, fallback.pit_loss_mean_s, fallback.pit_loss_sd_s
    )

    total_laps = 0
    sc_starts = 0
    dnf_events = 0
    car_lap_exposure = 0
    source_files = []
    for session_key in session_keys:
        session_dir = Path(raw_root) / str(session_key)
        lap_rows = _read_rows(session_dir / "laps.json")
        control_rows = _read_rows(session_dir / "race_control.json")
        result_rows = _read_rows(session_dir / "session_result.json")
        lap_numbers = [
            int(row["lap_number"])
            for row in lap_rows
            if isinstance(row.get("lap_number"), (int, float)) and row["lap_number"] > 0
        ]
        if lap_numbers:
            total_laps += max(lap_numbers)
        sc_starts += _sc_start_count(control_rows)
        failures, exposure = _dnf_exposure(result_rows)
        dnf_events += failures
        car_lap_exposure += exposure
        source_files.append({
            "session_key": session_key,
            "laps": str(session_dir / "laps.json"),
            "intervals": str(session_dir / "intervals.json"),
            "race_control": str(session_dir / "race_control.json"),
            "session_result": str(session_dir / "session_result.json"),
            "drivers": str(session_dir / "drivers.json"),
        })

    if total_laps > 0 and sc_starts > 0:
        sc_hazard = float(np.clip(sc_starts / total_laps, 0.0001, 0.15))
        sc_source = "race_control_empirical_starts_per_race_lap"
    else:
        sc_hazard = fallback.safety_car_hazard_per_lap
        sc_source = "fallback_insufficient_race_control_events"

    if dnf_events >= 3 and car_lap_exposure >= 500:
        dnf_hazard = float(np.clip(dnf_events / car_lap_exposure, 0.00005, 0.03))
        dnf_source = "session_result_dnf_per_car_lap_exposure"
    else:
        dnf_hazard = fallback.dnf_hazard_per_lap
        dnf_source = "fallback_insufficient_session_result_exposure"

    reliability_records = records_from_raw(raw_root, session_keys)
    reliability_model, reliability_audit = calibrate_reliability(
        reliability_records,
        pooled_fallback=dnf_hazard,
    )
    tyre_model, tyre_audit = calibrate_tyre_priors(datasets)
    traffic_model, traffic_audit = calibrate_traffic_prior(
        datasets,
        raw_root,
        session_keys,
        fallback_mean_s=fallback.traffic_penalty_mean_s,
        fallback_sd_s=fallback.traffic_penalty_sd_s,
        close_window_s=fallback.traffic_window_s,
    )

    priors = StrategyPriors(
        pit_loss_mean_s=float(np.clip(pit_mean, 8.0, 45.0)),
        pit_loss_sd_s=float(np.clip(pit_sd, 0.5, 8.0)),
        safety_car_hazard_per_lap=sc_hazard,
        dnf_hazard_per_lap=dnf_hazard,
        pit_observations=len(pit_values),
        safety_car_starts=sc_starts,
        race_laps_observed=total_laps,
        dnf_events=dnf_events,
        car_laps_observed=car_lap_exposure,
        sessions=tuple(int(key) for key in session_keys),
        dnf_source=dnf_source,
    )
    audit = {
        "pit_loss_definition": "pit target lap duration minus its prior five-lap median; pit-out laps excluded",
        "safety_car_definition": "race-control deployment/start messages divided by observed race laps",
        "safety_car_source": sc_source,
        "dnf_definition": "session_result.dnf failures over approximate car-lap risk intervals; DNS/DSQ excluded",
        "dnf_source": dnf_source,
        "reliability": reliability_audit,
        "reliability_model": reliability_payload(reliability_model),
        "tyre": tyre_audit,
        "tyre_model": tyre_payload(tyre_model),
        "traffic": traffic_audit,
        "traffic_model": traffic_payload(traffic_model),
        "source_files": source_files,
        "limitations": [
            "Pit lap excess is not identical to geometric pit-lane loss and remains traffic/condition dependent.",
            "SC/VSC event frequency is a historical prior, not a causal per-lap forecast for a specific circuit.",
            "Reliability hazards are public-results priors, not current mechanical-fault diagnoses.",
            "Team and driver reliability effects are hierarchically shrunk because DNF samples are sparse.",
            "Tyre calibration is retrospective public timing, not physical tyre energy or a guaranteed tyre-life limit.",
            "Traffic calibration is an observational close-vs-clear timing contrast, not a causal aerodynamic model.",
        ],
    }
    return priors, audit


def save_strategy_priors(
    priors: StrategyPriors, audit: dict[str, Any], path: Path
) -> dict[str, Any]:
    clean_audit = dict(audit)
    reliability = clean_audit.pop("reliability_model", None)
    tyre = clean_audit.pop("tyre_model", None)
    traffic = clean_audit.pop("traffic_model", None)
    payload = {
        "schema_version": 5,
        "priors": asdict(priors),
        "reliability": reliability,
        "tyre": tyre,
        "traffic": traffic,
        "audit": clean_audit,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def load_simulation_config(
    path: Path,
    *,
    samples: int | None = None,
    seed: int | None = None,
) -> tuple[SimulationConfig, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    priors = payload.get("priors") if isinstance(payload, dict) else None
    if not isinstance(priors, dict):
        raise ValueError("Strategy prior file has no priors object")
    pace, degradation, pit_age = maps_from_payload(payload.get("tyre"))
    base = SimulationConfig()
    traffic_window, traffic_mean, traffic_sd = config_values_from_payload(
        payload.get("traffic"),
        fallback_window_s=base.traffic_window_s,
        fallback_mean_s=base.traffic_penalty_mean_s,
        fallback_sd_s=base.traffic_penalty_sd_s,
    )
    config = SimulationConfig(
        samples=int(samples if samples is not None else base.samples),
        seed=int(seed if seed is not None else base.seed),
        lap_noise_s=base.lap_noise_s,
        pit_loss_mean_s=float(priors["pit_loss_mean_s"]),
        pit_loss_sd_s=float(priors["pit_loss_sd_s"]),
        dnf_hazard_per_lap=float(priors.get("dnf_hazard_per_lap", base.dnf_hazard_per_lap)),
        safety_car_hazard_per_lap=float(priors["safety_car_hazard_per_lap"]),
        safety_car_gap_multiplier=base.safety_car_gap_multiplier,
        safety_car_pit_loss_multiplier=base.safety_car_pit_loss_multiplier,
        max_degradation_s_per_lap=base.max_degradation_s_per_lap,
        traffic_window_s=traffic_window,
        traffic_penalty_mean_s=traffic_mean,
        traffic_penalty_sd_s=traffic_sd,
        compound_pace_delta_s=pace,
        compound_degradation_s_per_lap=degradation,
        compound_stint_target_laps=pit_age,
    )
    return config, payload
