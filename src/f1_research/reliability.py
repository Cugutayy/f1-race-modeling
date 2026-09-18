"""Hierarchically shrunk discrete reliability hazards from public race results.

The model treats each completed car-lap as a survived risk interval and a DNF as one
failure interval after the last completed lap. Team and driver rates are deliberately
shrunk toward broader priors because Formula One samples are small and mechanical
failures are sparse. Shrinkage strengths are selected only by forward-in-time
survival likelihood on earlier sessions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .value_parsing import strict_optional_bool

TEAM_STRENGTHS = (250.0, 500.0, 1000.0, 2000.0)
DRIVER_STRENGTHS = (100.0, 250.0, 500.0, 1000.0)
DEFAULT_TEAM_STRENGTH = 1000.0
DEFAULT_DRIVER_STRENGTH = 500.0
MIN_TUNING_SESSIONS = 4


@dataclass(frozen=True)
class ReliabilityRecord:
    session_key: int
    session_order: int
    driver_number: int
    team_name: str
    failure: int
    exposure: int


@dataclass(frozen=True)
class ReliabilityModel:
    enabled: bool
    pooled_hazard_per_lap: float
    team_strength: float
    driver_strength: float
    team_stats: dict[str, dict[str, float]]
    driver_stats: dict[str, dict[str, float]]
    sessions: tuple[int, ...]
    failures: int
    exposure: int
    tuning_nll_per_interval: float | None
    tuning_source: str


def _team_name(value: Any) -> str:
    text = str(value or "UNKNOWN").strip()
    return text if text else "UNKNOWN"


def records_from_rows(
    session_key: int,
    session_order: int,
    result_rows: list[dict[str, Any]],
    driver_rows: list[dict[str, Any]] | None = None,
) -> list[ReliabilityRecord]:
    """Normalize one race into discrete survival records.

    DNS and DSQ are excluded because they are different processes. A finisher with N
    completed laps contributes N survived intervals. A DNF with N completed laps
    contributes N survived intervals plus one failure interval.
    """
    team_lookup: dict[int, str] = {}
    for row in driver_rows or []:
        try:
            number = int(row.get("driver_number"))
        except (TypeError, ValueError):
            continue
        team_lookup[number] = _team_name(row.get("team_name"))

    output: list[ReliabilityRecord] = []
    for row in result_rows:
        dns = strict_optional_bool(row.get("dns"), field="openf1.session_result.dns")
        dsq = strict_optional_bool(row.get("dsq"), field="openf1.session_result.dsq")
        dnf = strict_optional_bool(row.get("dnf"), field="openf1.session_result.dnf")
        if dns is None or dsq is None or dnf is None:
            continue
        if dns or dsq:
            continue
        try:
            driver_number = int(row.get("driver_number"))
            completed_laps = int(row.get("number_of_laps"))
        except (TypeError, ValueError):
            continue
        if driver_number <= 0 or completed_laps < 0:
            continue
        failure = int(dnf)
        exposure = completed_laps + failure
        if exposure <= 0:
            continue
        output.append(ReliabilityRecord(
            session_key=int(session_key),
            session_order=int(session_order),
            driver_number=driver_number,
            team_name=team_lookup.get(driver_number, "UNKNOWN"),
            failure=failure,
            exposure=exposure,
        ))
    return output


def records_from_raw(raw_root: Path, session_keys: Iterable[int]) -> list[ReliabilityRecord]:
    import json

    output: list[ReliabilityRecord] = []
    for order, session_key in enumerate(session_keys):
        session_dir = Path(raw_root) / str(int(session_key))

        def read(name: str) -> list[dict[str, Any]]:
            path = session_dir / name
            if not path.exists():
                return []
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise ValueError(f"Invalid reliability source: {path}")
            return value

        output.extend(records_from_rows(
            int(session_key),
            order,
            read("session_result.json"),
            read("drivers.json"),
        ))
    return output


def _aggregate(records: Iterable[ReliabilityRecord], key) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for record in records:
        name = str(key(record))
        bucket = output.setdefault(name, {"failures": 0.0, "exposure": 0.0})
        bucket["failures"] += float(record.failure)
        bucket["exposure"] += float(record.exposure)
    return output


def _pooled_hazard(records: Iterable[ReliabilityRecord], fallback: float) -> tuple[float, int, int]:
    rows = list(records)
    failures = int(sum(row.failure for row in rows))
    exposure = int(sum(row.exposure for row in rows))
    if exposure <= 0:
        return float(fallback), failures, exposure
    # Jeffreys smoothing prevents exact zero/one hazards in small samples.
    hazard = (failures + 0.5) / (exposure + 1.0)
    return float(np.clip(hazard, 0.00001, 0.05)), failures, exposure


def fit_reliability_model(
    records: list[ReliabilityRecord],
    *,
    pooled_fallback: float,
    team_strength: float = DEFAULT_TEAM_STRENGTH,
    driver_strength: float = DEFAULT_DRIVER_STRENGTH,
    enabled: bool | None = None,
    tuning_nll_per_interval: float | None = None,
    tuning_source: str = "fixed_default_strengths",
) -> ReliabilityModel:
    if team_strength <= 0 or driver_strength <= 0:
        raise ValueError("Reliability shrinkage strengths must be positive")
    pooled, failures, exposure = _pooled_hazard(records, pooled_fallback)
    if enabled is None:
        enabled = failures >= 3 and exposure >= 500

    team_stats = _aggregate(records, lambda row: row.team_name)
    for stats in team_stats.values():
        stats["hazard_per_lap"] = float(np.clip(
            (stats["failures"] + team_strength * pooled) / (stats["exposure"] + team_strength),
            0.00001,
            0.05,
        ))

    driver_stats = _aggregate(records, lambda row: row.driver_number)
    sessions = tuple(sorted({row.session_key for row in records}))
    return ReliabilityModel(
        enabled=bool(enabled),
        pooled_hazard_per_lap=pooled,
        team_strength=float(team_strength),
        driver_strength=float(driver_strength),
        team_stats=team_stats,
        driver_stats=driver_stats,
        sessions=sessions,
        failures=failures,
        exposure=exposure,
        tuning_nll_per_interval=tuning_nll_per_interval,
        tuning_source=tuning_source,
    )


def predict_hazard(model: ReliabilityModel | dict[str, Any], driver_number: int, team_name: str | None) -> float:
    if isinstance(model, dict):
        model = ReliabilityModel(**model)
    pooled = float(model.pooled_hazard_per_lap)
    if not model.enabled:
        return pooled
    team = _team_name(team_name)
    team_stats = model.team_stats.get(team)
    team_hazard = float(team_stats.get("hazard_per_lap")) if team_stats else pooled
    driver_stats = model.driver_stats.get(str(int(driver_number)))
    if not driver_stats:
        return team_hazard
    hazard = (
        float(driver_stats["failures"]) + model.driver_strength * team_hazard
    ) / (float(driver_stats["exposure"]) + model.driver_strength)
    return float(np.clip(hazard, 0.00001, 0.05))


def _survival_nll(record: ReliabilityRecord, hazard: float) -> float:
    hazard = float(np.clip(hazard, 1e-9, 1 - 1e-9))
    survived = max(0, record.exposure - record.failure)
    return float(-(record.failure * np.log(hazard) + survived * np.log1p(-hazard)))


def tune_shrinkage(
    records: list[ReliabilityRecord],
    *,
    pooled_fallback: float,
    team_strengths: tuple[float, ...] = TEAM_STRENGTHS,
    driver_strengths: tuple[float, ...] = DRIVER_STRENGTHS,
) -> tuple[float, float, list[dict[str, float]], str]:
    """Choose shrinkage by forward survival likelihood without future-session labels."""
    session_orders = sorted({row.session_order for row in records})
    if len(session_orders) < MIN_TUNING_SESSIONS:
        return DEFAULT_TEAM_STRENGTH, DEFAULT_DRIVER_STRENGTH, [], "insufficient_sessions_default_strengths"

    trials: list[dict[str, float]] = []
    for team_strength in team_strengths:
        for driver_strength in driver_strengths:
            total_nll = 0.0
            total_exposure = 0
            scored_sessions = 0
            for validation_order in session_orders[2:]:
                train = [row for row in records if row.session_order < validation_order]
                validation = [row for row in records if row.session_order == validation_order]
                if not train or not validation:
                    continue
                candidate = fit_reliability_model(
                    train,
                    pooled_fallback=pooled_fallback,
                    team_strength=team_strength,
                    driver_strength=driver_strength,
                    enabled=True,
                )
                for row in validation:
                    hazard = predict_hazard(candidate, row.driver_number, row.team_name)
                    total_nll += _survival_nll(row, hazard)
                    total_exposure += row.exposure
                scored_sessions += 1
            if total_exposure <= 0:
                continue
            trials.append({
                "team_strength": float(team_strength),
                "driver_strength": float(driver_strength),
                "nll_per_interval": float(total_nll / total_exposure),
                "scored_sessions": float(scored_sessions),
                "scored_exposure": float(total_exposure),
            })

    if not trials:
        return DEFAULT_TEAM_STRENGTH, DEFAULT_DRIVER_STRENGTH, [], "no_valid_forward_trials"
    best = min(trials, key=lambda row: (
        row["nll_per_interval"], row["team_strength"], row["driver_strength"]
    ))
    return (
        float(best["team_strength"]),
        float(best["driver_strength"]),
        trials,
        "forward_survival_likelihood",
    )


def calibrate_reliability(
    records: list[ReliabilityRecord],
    *,
    pooled_fallback: float,
) -> tuple[ReliabilityModel, dict[str, Any]]:
    team_strength, driver_strength, trials, source = tune_shrinkage(
        records,
        pooled_fallback=pooled_fallback,
    )
    enabled = sum(row.failure for row in records) >= 3 and sum(row.exposure for row in records) >= 500
    selected_nll = None
    if trials:
        selected = min(trials, key=lambda row: (
            row["nll_per_interval"], row["team_strength"], row["driver_strength"]
        ))
        selected_nll = float(selected["nll_per_interval"])
    model = fit_reliability_model(
        records,
        pooled_fallback=pooled_fallback,
        team_strength=team_strength,
        driver_strength=driver_strength,
        enabled=enabled,
        tuning_nll_per_interval=selected_nll,
        tuning_source=source,
    )
    audit = {
        "definition": "discrete car-lap survival hazard; DNF adds one failure interval; DNS/DSQ excluded",
        "enabled": model.enabled,
        "pooled_hazard_per_lap": model.pooled_hazard_per_lap,
        "failures": model.failures,
        "exposure": model.exposure,
        "team_strength": model.team_strength,
        "driver_strength": model.driver_strength,
        "tuning_source": source,
        "selected_nll_per_interval": selected_nll,
        "trial_count": len(trials),
        "limitations": [
            "This is a reliability prior, not a diagnosis of a current mechanical fault.",
            "Public results do not identify failure mode, component life or private sensor warnings.",
            "Team and driver effects are shrunk heavily because race-level failure samples are sparse.",
        ],
    }
    return model, audit


def reliability_overrides_from_state(
    snapshot: dict[str, Any],
    model: ReliabilityModel | dict[str, Any] | None,
) -> dict[int, float]:
    if model is None:
        return {}
    if isinstance(model, dict):
        model = ReliabilityModel(**model)
    output: dict[int, float] = {}
    for row in snapshot.get("drivers", []):
        try:
            number = int(row.get("driver_number"))
        except (TypeError, ValueError):
            continue
        output[number] = predict_hazard(model, number, row.get("team_name"))
    return output


def as_payload(model: ReliabilityModel) -> dict[str, Any]:
    return asdict(model)
