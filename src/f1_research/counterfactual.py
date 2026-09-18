"""Counterfactual pit-window analysis using common-random-number simulation."""
from __future__ import annotations

from typing import Any

from .strategy import SimulationConfig, Strategy, predict_from_state


def pit_window(snapshot: dict[str, Any], *, total_laps: int, driver_number: int,
               offsets: tuple[int, ...] = (1, 2, 3, 5),
               compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
               config: SimulationConfig | None = None) -> dict[str, Any]:
    config = config or SimulationConfig()
    remaining = total_laps - int(snapshot.get("current_lap") or 0)
    if remaining < 1:
        raise ValueError("race has no remaining laps")
    if not offsets or any(
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset >= remaining
        for offset in offsets
    ):
        raise ValueError(
            "pit offsets must be unique integers from 0 (pit now) through remaining_laps - 1"
        )
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            "pit offsets must be unique integers from 0 (pit now) through remaining_laps - 1"
        )
    normalized_compounds = tuple(str(compound).upper() for compound in compounds)
    if not normalized_compounds or len(set(normalized_compounds)) != len(normalized_compounds):
        raise ValueError("strategy compounds must be non-empty and unique")
    unsupported = sorted(set(normalized_compounds) - set(config.compound_pace_delta_s))
    if unsupported:
        raise ValueError(f"unsupported strategy compounds: {', '.join(unsupported)}")
    scenarios = []
    for compound in normalized_compounds:
        for offset in offsets:
            strategy = Strategy(
                pit_in_laps=offset,
                next_compound=compound,
                label=(f"pit-now:{compound}" if offset == 0 else f"pit+{offset}:{compound}"),
            )
            report = predict_from_state(
                snapshot, total_laps,
                strategies={int(driver_number): strategy},
                config=config,
            )
            driver = next(
                (row for row in report["predictions"] if int(row["driver_number"]) == int(driver_number)),
                None,
            )
            if driver is None:
                raise ValueError(f"driver {driver_number} is not simulation-eligible")
            scenarios.append({
                "pit_in_laps": offset,
                "compound": compound,
                "expected_position": driver["expected_position"],
                "win_probability": driver["win_probability"],
                "podium_probability": driver["podium_probability"],
                "top10_probability": driver["top10_probability"],
                "dnf_probability": driver["dnf_probability"],
            })
    scenarios.sort(key=lambda row: (row["expected_position"], -row["win_probability"], row["pit_in_laps"]))
    return {
        "schema_version": 1,
        "analysis_kind": "counterfactual_pit_window",
        "driver_number": int(driver_number),
        "samples_per_scenario": config.samples,
        "seed": config.seed,
        "best_by_expected_position": scenarios[0],
        "scenarios": scenarios,
        "note": "Counterfactual public-data simulation; not team strategy software.",
    }
