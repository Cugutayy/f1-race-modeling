"""Counterfactual pit-window analysis using common-random-number simulation."""
from __future__ import annotations

from typing import Any

from .strategy import SimulationConfig, Strategy, predict_from_state


def pit_window(snapshot: dict[str, Any], *, total_laps: int, driver_number: int,
               offsets: tuple[int, ...] = (0, 1, 2, 3, 5),
               compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
               config: SimulationConfig | None = None) -> dict[str, Any]:
    config = config or SimulationConfig()
    if not offsets or any(offset < 0 for offset in offsets):
        raise ValueError("pit offsets must be non-negative")
    scenarios = []
    for compound in compounds:
        for offset in offsets:
            strategy = Strategy(
                pit_in_laps=max(1, offset),
                next_compound=compound,
                label=f"pit+{offset}:{compound}",
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
