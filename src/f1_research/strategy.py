"""Transparent live race/strategy Monte Carlo engine.

This is deliberately a simulation layer, not a claim that public broadcast data
contains team-only fuel, tyre-temperature or setup information. Every uncertain
quantity has an explicit default and can later be replaced by a learned model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .tyre_calibration import DEFAULT_DEGRADATION, DEFAULT_PACE_DELTA, DEFAULT_PIT_AGE


@dataclass(frozen=True)
class SimulationConfig:
    samples: int = 20_000
    seed: int = 42
    lap_noise_s: float = 0.32
    pit_loss_mean_s: float = 22.0
    pit_loss_sd_s: float = 1.6
    dnf_hazard_per_lap: float = 0.0018
    safety_car_hazard_per_lap: float = 0.012
    safety_car_gap_multiplier: float = 0.30
    safety_car_pit_loss_multiplier: float = 0.58
    max_degradation_s_per_lap: float = 0.20
    compound_pace_delta_s: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PACE_DELTA)
    )
    compound_degradation_s_per_lap: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DEGRADATION)
    )
    compound_stint_target_laps: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PIT_AGE)
    )

    def __post_init__(self):
        if self.samples < 1000:
            raise ValueError("samples must be >= 1000 for stable scenario summaries")
        if not 0 <= self.dnf_hazard_per_lap < 1 or not 0 <= self.safety_car_hazard_per_lap < 1:
            raise ValueError("hazards must be probabilities in [0, 1)")
        for name, mapping in (
            ("compound_pace_delta_s", self.compound_pace_delta_s),
            ("compound_degradation_s_per_lap", self.compound_degradation_s_per_lap),
            ("compound_stint_target_laps", self.compound_stint_target_laps),
        ):
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"{name} must be a non-empty mapping")
            if any(not np.isfinite(float(value)) for value in mapping.values()):
                raise ValueError(f"{name} contains a non-finite value")
        if any(not 0 <= float(value) <= 0.5 for value in self.compound_degradation_s_per_lap.values()):
            raise ValueError("compound degradation priors must be in [0, 0.5]")
        if any(not 2 <= float(value) <= 80 for value in self.compound_stint_target_laps.values()):
            raise ValueError("compound stint targets must be in [2, 80]")


@dataclass(frozen=True)
class PaceOverride:
    pace_s: float
    uncertainty_s: float
    source: str = "learned_model"

    def __post_init__(self):
        if not np.isfinite(self.pace_s) or self.pace_s <= 0:
            raise ValueError("pace override must be a positive finite lap time")
        if not np.isfinite(self.uncertainty_s) or self.uncertainty_s <= 0:
            raise ValueError("pace override uncertainty must be positive and finite")


@dataclass(frozen=True)
class DriverInput:
    driver_number: int
    label: str
    current_position: int
    gap_to_leader_s: float
    pace_s: float
    pace_uncertainty_s: float
    degradation_s_per_lap: float
    tyre_age: int
    compound: str
    pit_stops: int = 0
    pace_source: str = "recent_laps"
    dnf_hazard_per_lap: float = 0.0018


@dataclass(frozen=True)
class Strategy:
    pit_in_laps: int | None = None
    next_compound: str = "MEDIUM"
    label: str = "Observed/default"


@dataclass
class SimulationResult:
    driver_number: int
    label: str
    expected_position: float
    win_probability: float
    podium_probability: float
    top10_probability: float
    position_p10: int
    position_p90: int
    dnf_probability: float
    mean_remaining_time_s: float


def _robust_pace(laps: list[float], fallback: float | None = None) -> tuple[float, float, float]:
    values = np.asarray([x for x in laps if np.isfinite(x) and x > 0], dtype=float)
    if len(values) == 0:
        if fallback is None or not np.isfinite(fallback):
            raise ValueError("No usable pace observation")
        return float(fallback), 1.5, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(1.0, 4.5 * 1.4826 * mad)
    clean = values[np.abs(values - median) <= threshold]
    if len(clean) < 2:
        clean = values
    pace = float(np.median(clean[-5:]))
    uncertainty = max(0.15, float(np.std(clean[-5:], ddof=1)) if len(clean[-5:]) > 1 else 0.8)
    slope = 0.0
    if len(clean) >= 4:
        recent = clean[-6:]
        pair_slopes = [
            (recent[j] - recent[i]) / (j - i)
            for i in range(len(recent))
            for j in range(i + 1, len(recent))
        ]
        slope = float(np.median(pair_slopes)) if pair_slopes else 0.0
    return pace, uncertainty, max(0.0, slope)


def drivers_from_state(
    snapshot: dict[str, Any],
    config: SimulationConfig | None = None,
    pace_overrides: dict[int, PaceOverride] | None = None,
    dnf_hazard_overrides: dict[int, float] | None = None,
) -> list[DriverInput]:
    config = config or SimulationConfig()
    pace_overrides = pace_overrides or {}
    dnf_hazard_overrides = dnf_hazard_overrides or {}
    rows = snapshot.get("drivers", [])
    if not isinstance(rows, list):
        raise ValueError("state.drivers must be a list")
    observed = []
    for row in rows:
        position = row.get("position")
        if not isinstance(position, int) or position < 1:
            continue
        driver_number = int(row["driver_number"])
        gap = row.get("gap_to_leader_s")
        gap = (
            0.0
            if position == 1
            else float(gap)
            if isinstance(gap, (int, float)) and np.isfinite(gap)
            else None
        )
        laps = row.get("recent_laps_s") or []
        fallback = row.get("last_lap_s")
        try:
            pace, uncertainty, degradation = _robust_pace(laps, fallback)
        except ValueError:
            continue
        source = "recent_laps"
        override = pace_overrides.get(driver_number)
        if override is not None:
            pace = float(override.pace_s)
            uncertainty = float(override.uncertainty_s)
            source = override.source
        dnf_hazard = float(dnf_hazard_overrides.get(driver_number, config.dnf_hazard_per_lap))
        if not np.isfinite(dnf_hazard) or not 0 <= dnf_hazard < 1:
            raise ValueError(f"Invalid DNF hazard for driver {driver_number}")
        observed.append((row, position, gap, pace, uncertainty, degradation, source, dnf_hazard))
    if len(observed) < 2:
        raise ValueError("At least two drivers need position and pace observations")

    known_gaps = [item[2] for item in observed if item[2] is not None]
    step = max(
        1.0,
        float(np.median(np.diff(sorted(set(known_gaps))))) if len(set(known_gaps)) >= 2 else 2.0,
    )
    result = []
    for row, position, gap, pace, uncertainty, degradation, source, dnf_hazard in observed:
        if gap is None:
            gap = step * (position - 1)
        result.append(
            DriverInput(
                driver_number=int(row["driver_number"]),
                label=row.get("acronym") or row.get("full_name") or str(row["driver_number"]),
                current_position=position,
                gap_to_leader_s=max(0.0, float(gap)),
                pace_s=pace,
                pace_uncertainty_s=min(max(uncertainty, 0.15), 3.0),
                degradation_s_per_lap=min(degradation, config.max_degradation_s_per_lap),
                tyre_age=max(0, int(row.get("tyre_age") or 0)),
                compound=str(row.get("compound") or "MEDIUM").upper(),
                pit_stops=max(0, int(row.get("pit_stops") or 0)),
                pace_source=source,
                dnf_hazard_per_lap=dnf_hazard,
            )
        )
    return sorted(result, key=lambda item: item.current_position)


def _compound_value(mapping: dict[str, float], compound: str, fallback: float) -> float:
    value = mapping.get(str(compound).upper(), fallback)
    return float(value) if np.isfinite(float(value)) else float(fallback)


def _default_pit_offset(driver: DriverInput, config: SimulationConfig) -> int | None:
    target = _compound_value(config.compound_stint_target_laps, driver.compound, 28.0)
    remaining = target - driver.tyre_age
    return max(1, int(np.ceil(remaining))) if remaining <= 12 else None


def _relative_compound_delta(
    start_compound: str,
    current_compound: str,
    config: SimulationConfig,
) -> float:
    """Relative fresh-tyre pace prior; current observed pace already includes wear."""
    start = _compound_value(config.compound_pace_delta_s, start_compound, 0.0)
    current = _compound_value(config.compound_pace_delta_s, current_compound, 0.0)
    return current - start


def simulate(
    drivers: list[DriverInput],
    laps_remaining: int,
    strategies: dict[int, Strategy] | None = None,
    config: SimulationConfig | None = None,
) -> tuple[list[SimulationResult], dict[str, Any]]:
    """Simulate coherent finishing orders from the current race state.

    ``pace_s`` is the observed/modelled pace at the current tyre state. Before a stop,
    only additional ageing relative to the current tyre age is added. After a stop,
    the simulator backs out a fresh-current-compound baseline, applies the empirical
    compound offset and then ages the new compound with its historical net age-trend
    prior. These public-data priors remain observational rather than tyre physics.
    """
    config = config or SimulationConfig()
    strategies = strategies or {}
    if laps_remaining < 1:
        raise ValueError("laps_remaining must be positive")
    if len(drivers) < 2:
        raise ValueError("At least two drivers required")
    if len({d.driver_number for d in drivers}) != len(drivers):
        raise ValueError("Duplicate driver numbers")

    rng = np.random.default_rng(config.seed)
    n, m = config.samples, len(drivers)
    remaining = np.zeros((n, m), dtype=float)
    dnf = np.zeros((n, m), dtype=bool)

    sc_probability = 1 - (1 - config.safety_car_hazard_per_lap) ** laps_remaining
    sc = rng.random(n) < sc_probability
    sc_lap = rng.integers(1, laps_remaining + 1, n)

    for j, driver in enumerate(drivers):
        strategy = strategies.get(driver.driver_number)
        pit_offset = strategy.pit_in_laps if strategy else _default_pit_offset(driver, config)
        next_compound = (strategy.next_compound if strategy else "MEDIUM").upper()
        gap = np.full(n, driver.gap_to_leader_s, dtype=float)
        gap[sc] *= config.safety_car_gap_multiplier
        total = gap
        age = np.full(n, driver.tyre_age, dtype=float)
        compound = driver.compound
        pit_done = False
        live_degradation = min(driver.degradation_s_per_lap, config.max_degradation_s_per_lap)
        fresh_current_pace = driver.pace_s - live_degradation * driver.tyre_age

        for lap in range(1, laps_remaining + 1):
            if pit_done:
                compound_delta = _relative_compound_delta(driver.compound, compound, config)
                new_degradation = _compound_value(
                    config.compound_degradation_s_per_lap,
                    compound,
                    live_degradation,
                )
                lap_mean = fresh_current_pace + compound_delta + new_degradation * age
            else:
                lap_mean = driver.pace_s + live_degradation * (age - driver.tyre_age)
            noise = rng.normal(0, max(config.lap_noise_s, driver.pace_uncertainty_s * 0.35), n)
            total += lap_mean + noise
            age += 1

            if pit_offset is not None and lap == pit_offset:
                pit_loss = rng.normal(config.pit_loss_mean_s, config.pit_loss_sd_s, n)
                pit_loss = np.maximum(8.0, pit_loss)
                pit_loss[sc & (sc_lap == lap)] *= config.safety_car_pit_loss_multiplier
                total += pit_loss
                age[:] = 0
                compound = next_compound
                pit_done = True

        dnf_probability = 1 - (1 - driver.dnf_hazard_per_lap) ** laps_remaining
        dnf[:, j] = rng.random(n) < dnf_probability
        remaining[:, j] = total

    ranking_score = remaining + dnf.astype(float) * 1_000_000.0
    orders = np.argsort(ranking_score, axis=1, kind="stable")
    ranks = np.empty_like(orders)
    np.put_along_axis(ranks, orders, np.arange(1, m + 1)[None, :], axis=1)

    results = []
    for j, driver in enumerate(drivers):
        r = ranks[:, j]
        results.append(
            SimulationResult(
                driver_number=driver.driver_number,
                label=driver.label,
                expected_position=float(r.mean()),
                win_probability=float((r == 1).mean()),
                podium_probability=float((r <= min(3, m)).mean()),
                top10_probability=float((r <= min(10, m)).mean()),
                position_p10=int(np.quantile(r, 0.10, method="inverted_cdf")),
                position_p90=int(np.quantile(r, 0.90, method="inverted_cdf")),
                dnf_probability=float(dnf[:, j].mean()),
                mean_remaining_time_s=float(remaining[:, j].mean()),
            )
        )
    audit = {
        "samples": n,
        "seed": config.seed,
        "laps_remaining": laps_remaining,
        "safety_car_any_probability": sc_probability,
        "assumptions": asdict(config),
        "pace_sources": {str(driver.driver_number): driver.pace_source for driver in drivers},
        "dnf_hazards_per_lap": {
            str(driver.driver_number): driver.dnf_hazard_per_lap for driver in drivers
        },
        "strategy_overrides": {str(key): asdict(value) for key, value in strategies.items()},
        "status": "research simulation; not calibrated team strategy software",
    }
    return results, audit


def predict_from_state(
    snapshot: dict[str, Any],
    total_laps: int,
    strategies: dict[int, Strategy] | None = None,
    config: SimulationConfig | None = None,
    pace_overrides: dict[int, PaceOverride] | None = None,
    dnf_hazard_overrides: dict[int, float] | None = None,
) -> dict[str, Any]:
    current_lap = snapshot.get("current_lap")
    if not isinstance(current_lap, int) or current_lap < 1:
        raise ValueError("Current lap is unavailable")
    laps_remaining = total_laps - current_lap
    if laps_remaining < 1:
        raise ValueError("Race has no future laps to simulate")
    drivers = drivers_from_state(snapshot, config, pace_overrides, dnf_hazard_overrides)
    results, audit = simulate(drivers, laps_remaining, strategies, config)
    return {
        "schema_version": 4,
        "analysis_kind": "live_race_monte_carlo",
        "session_key": snapshot.get("session_key"),
        "state_updated_at": snapshot.get("updated_at"),
        "current_lap": current_lap,
        "total_laps": total_laps,
        "predictions": [asdict(item) for item in results],
        "audit": audit,
    }


def compare_pit_windows(
    snapshot: dict[str, Any],
    total_laps: int,
    driver_number: int,
    offsets: tuple[int, ...] = (1, 2, 3, 4, 5),
    compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    config: SimulationConfig | None = None,
    pace_overrides: dict[int, PaceOverride] | None = None,
    dnf_hazard_overrides: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Counterfactual pit scenarios using common random seeds for lower comparison noise."""
    config = config or SimulationConfig()
    output = []
    for offset in offsets:
        for compound in compounds:
            strategy = Strategy(offset, compound, f"Pit +{offset} / {compound}")
            report = predict_from_state(
                snapshot,
                total_laps,
                strategies={driver_number: strategy},
                config=config,
                pace_overrides=pace_overrides,
                dnf_hazard_overrides=dnf_hazard_overrides,
            )
            row = next((p for p in report["predictions"] if p["driver_number"] == driver_number), None)
            if row is None:
                raise ValueError(f"Driver {driver_number} not present in live state")
            output.append({"pit_in_laps": offset, "compound": compound, **row})
    return output
