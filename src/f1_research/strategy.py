"""Transparent live race/strategy Monte Carlo engine.

This is deliberately a simulation layer, not a claim that public broadcast data
contains team-only fuel, tyre-temperature or setup information. Every uncertain
quantity has an explicit default and can later be replaced by a learned model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .data_truth import validate_simulation_observations
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
    traffic_window_s: float = 1.20
    traffic_penalty_mean_s: float = 0.12
    traffic_penalty_sd_s: float = 0.05
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
        if not 0 < self.safety_car_gap_multiplier <= 1:
            raise ValueError("safety_car_gap_multiplier must be in (0, 1]")
        if not 0 < self.safety_car_pit_loss_multiplier <= 1:
            raise ValueError("safety_car_pit_loss_multiplier must be in (0, 1]")
        if self.traffic_window_s < 0:
            raise ValueError("traffic_window_s must be non-negative")
        if self.traffic_penalty_mean_s < 0 or self.traffic_penalty_sd_s < 0:
            raise ValueError("traffic penalties must be non-negative")
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

    missing = validate_simulation_observations(snapshot)
    if missing:
        details = "; ".join(
            f"{driver}: {','.join(fields)}" for driver, fields in sorted(missing.items())
        )
        raise ValueError(f"Live state is incomplete for simulation: {details}")

    result = []
    for row in rows:
        position = row.get("position")
        if not isinstance(position, int) or position < 1:
            continue
        driver_number = int(row["driver_number"])
        gap = 0.0 if position == 1 else float(row["gap_to_leader_s"])
        laps = row.get("recent_laps_s") or []
        fallback = row.get("last_lap_s")
        try:
            pace, uncertainty, degradation = _robust_pace(laps, fallback)
        except ValueError as exc:
            raise ValueError(f"Driver {driver_number} has no usable pace observation") from exc
        source = "recent_laps"
        override = pace_overrides.get(driver_number)
        if override is not None:
            pace = float(override.pace_s)
            uncertainty = float(override.uncertainty_s)
            source = override.source
        dnf_hazard = float(dnf_hazard_overrides.get(driver_number, config.dnf_hazard_per_lap))
        if not np.isfinite(dnf_hazard) or not 0 <= dnf_hazard < 1:
            raise ValueError(f"Invalid DNF hazard for driver {driver_number}")
        compound = str(row["compound"]).upper()
        tyre_age = int(row["tyre_age"])
        result.append(
            DriverInput(
                driver_number=driver_number,
                label=row.get("acronym") or row.get("full_name") or str(driver_number),
                current_position=position,
                gap_to_leader_s=max(0.0, gap),
                pace_s=pace,
                pace_uncertainty_s=min(max(uncertainty, 0.15), 3.0),
                degradation_s_per_lap=min(degradation, config.max_degradation_s_per_lap),
                tyre_age=tyre_age,
                compound=compound,
                pit_stops=max(0, int(row.get("pit_stops") or 0)),
                pace_source=source,
                dnf_hazard_per_lap=dnf_hazard,
            )
        )
    if len(result) < 2:
        raise ValueError("At least two complete drivers are required for simulation")
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


def _compress_gaps(total: np.ndarray, mask: np.ndarray, multiplier: float) -> None:
    """Compress only the selected Monte Carlo samples around their current leader."""
    if not np.any(mask):
        return
    selected = total[mask]
    leader = selected.min(axis=1, keepdims=True)
    total[mask] = leader + (selected - leader) * multiplier


def _sample_first_event_lap(
    rng: np.random.Generator,
    samples: int,
    laps_remaining: int,
    hazard_per_lap: float,
) -> np.ndarray:
    """Sample the first event from a constant discrete hazard, 0 meaning no event."""
    if hazard_per_lap <= 0:
        return np.zeros(samples, dtype=int)
    event_lap = rng.geometric(hazard_per_lap, size=samples)
    return np.where(event_lap <= laps_remaining, event_lap, 0).astype(int)


def _traffic_penalty(
    total: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
    disabled_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Approximate close-following loss from the simulated order at lap start.

    This is intentionally a conservative state-derived heuristic. It is not a learned
    aerodynamic/DRS model. Random numbers are drawn for every sample/position so pit
    strategy comparisons retain common-random-number structure as far as possible.
    """
    n, m = total.shape
    penalty = np.zeros_like(total)
    if m < 2 or config.traffic_window_s <= 0 or config.traffic_penalty_mean_s <= 0:
        return penalty, 0

    order = np.argsort(total, axis=1, kind="stable")
    ordered = np.take_along_axis(total, order, axis=1)
    gaps = np.diff(ordered, axis=1)
    close = (gaps > 0) & (gaps < config.traffic_window_s)
    if disabled_samples is not None:
        close &= ~disabled_samples[:, None]

    intensity = np.clip(1.0 - gaps / config.traffic_window_s, 0.0, 1.0)
    mean = config.traffic_penalty_mean_s * intensity
    draw = rng.normal(mean, config.traffic_penalty_sd_s, size=(n, m - 1))
    draw = np.where(close, np.maximum(0.0, draw), 0.0)

    ordered_penalty = np.zeros_like(total)
    ordered_penalty[:, 1:] = draw
    np.put_along_axis(penalty, order, ordered_penalty, axis=1)
    return penalty, int(close.sum())


def simulate(
    drivers: list[DriverInput],
    laps_remaining: int,
    strategies: dict[int, Strategy] | None = None,
    config: SimulationConfig | None = None,
) -> tuple[list[SimulationResult], dict[str, Any]]:
    """Simulate coherent finishing orders from the current race state.

    The engine advances all cars one lap at a time. That matters because a future
    Safety Car must compress the *then-current* gaps, not today's gaps, and a pit stop
    can move a car into traffic for subsequent laps. Traffic loss is a small explicit
    heuristic based on simulated time-to-car-ahead; it is not represented as team
    aerodynamic or overtaking software.
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
    total = np.tile(
        np.asarray([driver.gap_to_leader_s for driver in drivers], dtype=float),
        (n, 1),
    )
    dnf = np.zeros((n, m), dtype=bool)

    sc_probability = 1 - (1 - config.safety_car_hazard_per_lap) ** laps_remaining
    sc_lap = _sample_first_event_lap(
        rng,
        n,
        laps_remaining,
        config.safety_car_hazard_per_lap,
    )
    sc_occurs = sc_lap > 0

    pit_offsets: list[int | None] = []
    next_compounds: list[str] = []
    compounds = [driver.compound for driver in drivers]
    ages = np.asarray([driver.tyre_age for driver in drivers], dtype=float)
    pit_done = np.zeros(m, dtype=bool)
    live_degradation = np.asarray(
        [min(driver.degradation_s_per_lap, config.max_degradation_s_per_lap) for driver in drivers],
        dtype=float,
    )
    fresh_current_pace = np.asarray(
        [driver.pace_s for driver in drivers], dtype=float
    ) - live_degradation * ages
    pace_uncertainty = np.asarray(
        [max(config.lap_noise_s, driver.pace_uncertainty_s * 0.35) for driver in drivers],
        dtype=float,
    )

    for driver in drivers:
        strategy = strategies.get(driver.driver_number)
        next_compound = (strategy.next_compound if strategy else "MEDIUM").upper()
        if next_compound not in config.compound_pace_delta_s:
            raise ValueError(f"Unsupported strategy compound: {next_compound}")
        pit_offset = strategy.pit_in_laps if strategy else _default_pit_offset(driver, config)
        if pit_offset is not None:
            if (
                isinstance(pit_offset, bool)
                or not isinstance(pit_offset, int)
                or pit_offset < 0
                or pit_offset >= laps_remaining
            ):
                raise ValueError(
                    "pit_in_laps must be an integer from 0 (pit now) through laps_remaining - 1"
                )
        pit_offsets.append(pit_offset)
        next_compounds.append(next_compound)

    # Draw the complete pit-loss random field before scenario-dependent decisions.
    # This keeps downstream RNG consumption aligned across counterfactual pit timings,
    # which is required for genuine common-random-number comparisons.
    pit_loss_draws = np.maximum(
        8.0,
        rng.normal(
            config.pit_loss_mean_s,
            config.pit_loss_sd_s,
            size=(laps_remaining, n, m),
        ),
    )

    # offset=0 means pit immediately, before the first future racing lap. It is
    # intentionally distinct from offset=1, which means run one more lap then pit.
    for j, pit_offset in enumerate(pit_offsets):
        if pit_offset != 0:
            continue
        total[:, j] += pit_loss_draws[0, :, j]
        compounds[j] = next_compounds[j]
        pit_done[j] = True
        ages[j] = 0.0

    traffic_events = 0
    sc_samples_by_lap: dict[str, int] = {}

    for lap in range(1, laps_remaining + 1):
        sc_now = sc_occurs & (sc_lap == lap)
        if np.any(sc_now):
            _compress_gaps(total, sc_now, config.safety_car_gap_multiplier)
            sc_samples_by_lap[str(lap)] = int(sc_now.sum())

        traffic, events = _traffic_penalty(total, rng, config, disabled_samples=sc_now)
        traffic_events += events

        lap_mean = np.empty(m, dtype=float)
        for j, driver in enumerate(drivers):
            if pit_done[j]:
                compound_delta = _relative_compound_delta(driver.compound, compounds[j], config)
                new_degradation = _compound_value(
                    config.compound_degradation_s_per_lap,
                    compounds[j],
                    live_degradation[j],
                )
                lap_mean[j] = fresh_current_pace[j] + compound_delta + new_degradation * ages[j]
            else:
                lap_mean[j] = driver.pace_s + live_degradation[j] * (ages[j] - driver.tyre_age)

        noise = rng.normal(0.0, pace_uncertainty, size=(n, m))
        total += lap_mean[None, :] + noise + traffic

        pitting = np.zeros(m, dtype=bool)
        for j, pit_offset in enumerate(pit_offsets):
            if pit_offset is None or pit_done[j] or lap != pit_offset:
                continue
            pit_loss = pit_loss_draws[lap, :, j].copy()
            pit_loss[sc_now] *= config.safety_car_pit_loss_multiplier
            total[:, j] += pit_loss
            compounds[j] = next_compounds[j]
            pit_done[j] = True
            pitting[j] = True

        ages += 1.0
        ages[pitting] = 0.0

    for j, driver in enumerate(drivers):
        dnf_probability = 1 - (1 - driver.dnf_hazard_per_lap) ** laps_remaining
        dnf[:, j] = rng.random(n) < dnf_probability

    ranking_score = total + dnf.astype(float) * 1_000_000.0
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
                mean_remaining_time_s=float(total[:, j].mean()),
            )
        )
    audit = {
        "samples": n,
        "seed": config.seed,
        "laps_remaining": laps_remaining,
        "safety_car_any_probability": sc_probability,
        "safety_car_event_time_model": "first-event truncated geometric from per-lap hazard",
        "safety_car_gap_application": "dynamic_at_sampled_sc_lap",
        "safety_car_samples_by_lap": sc_samples_by_lap,
        "safety_car_duration_model": "not_modelled; pit discount applies on sampled start lap only",
        "traffic_model": "simulated-gap heuristic; not empirically calibrated aero/DRS model",
        "traffic_close_following_events": traffic_events,
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
        "schema_version": 5,
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