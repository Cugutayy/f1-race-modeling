"""Compose strict next-lap intelligence with the transparent race simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .lap_strict import STRICT_FEATURES, predict_live_strict
from .strategy import PaceOverride, SimulationConfig, compare_pit_windows, predict_from_state


def load_strict_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Strict pace artifact must be a dictionary")
    if artifact.get("task") != "next_lap_strict_mixture":
        raise ValueError("Artifact is not a strict next-lap mixture")
    if artifact.get("features") != STRICT_FEATURES:
        raise ValueError("Strict pace feature schema mismatch")
    if artifact.get("retrospective_stint_features_used") is not False:
        raise ValueError("Strict artifact does not prove retrospective stint exclusion")
    return artifact


def pace_overrides_from_frame(frame: pd.DataFrame, artifact: dict[str, Any]) -> dict[int, PaceOverride]:
    """Convert green-lap predictions into race-simulator pace priors.

    Split-conformal radius is distribution-free and is not a Gaussian sigma. For the
    simulator it is conservatively mapped to a dispersion scale by dividing the 90%
    radius by 1.645, then bounded. This mapping is an explicit engineering
    approximation and is included in the returned source label/audit.
    """
    if frame.empty:
        return {}
    radius = float(artifact.get("conformal_radius_s", np.nan))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Strict artifact has invalid conformal radius")
    sigma = float(np.clip(radius / 1.645, 0.15, 3.0))
    output: dict[int, PaceOverride] = {}
    for row in frame.itertuples(index=False):
        pace = float(row.predicted_green_lap_s)
        if not np.isfinite(pace) or pace <= 0:
            continue
        output[int(row.driver_number)] = PaceOverride(
            pace_s=pace,
            uncertainty_s=sigma,
            source="strict_next_lap_conformal",
        )
    return output


def combined_live_report(
    snapshot: dict[str, Any],
    total_laps: int,
    artifact: dict[str, Any],
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    pace = predict_live_strict(artifact, snapshot)
    overrides = pace_overrides_from_frame(pace, artifact)
    race = predict_from_state(
        snapshot,
        total_laps,
        config=config,
        pace_overrides=overrides,
    )
    race["pace_model"] = {
        "task": artifact.get("task"),
        "selected_regressor": artifact.get("selected_regressor"),
        "trained_through_session": artifact.get("trained_through_session"),
        "calibration_session": artifact.get("calibration_session"),
        "sealed_test_session": artifact.get("sealed_test_session"),
        "conformal_alpha": artifact.get("conformal_alpha"),
        "conformal_radius_s": artifact.get("conformal_radius_s"),
        "override_drivers": sorted(overrides),
        "uncertainty_mapping": "conformal_radius / 1.645, clipped to [0.15, 3.0] seconds",
    }
    race["pace_predictions"] = pace.to_dict("records")
    return race


def combined_pit_windows(
    snapshot: dict[str, Any],
    total_laps: int,
    driver_number: int,
    artifact: dict[str, Any],
    config: SimulationConfig | None = None,
    offsets: tuple[int, ...] = (1, 2, 3, 4, 5),
    compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> list[dict[str, Any]]:
    pace = predict_live_strict(artifact, snapshot)
    overrides = pace_overrides_from_frame(pace, artifact)
    return compare_pit_windows(
        snapshot,
        total_laps,
        driver_number,
        offsets=offsets,
        compounds=compounds,
        config=config,
        pace_overrides=overrides,
    )
