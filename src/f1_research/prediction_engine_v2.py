"""Composable prediction engine v2.

Keeps pace, race-state and strategy outputs separate, then fuses them into one
auditable forecast envelope. The engine never manufactures unavailable signals.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import math

@dataclass(frozen=True)
class PredictionQuality:
    state_age_s: float | None
    pace_available: bool
    uncertainty_available: bool
    field_coverage: float
    degraded: bool
    reasons: tuple[str, ...]

def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))

def assess_quality(state: dict[str, Any], pace_predictions: list[dict[str, Any]] | None) -> PredictionQuality:
    drivers = state.get("drivers") or []
    pace = pace_predictions or []
    usable = sum(1 for row in drivers if _finite(row.get("last_lap_s")) or len(row.get("pace_laps_s") or []) >= 3)
    coverage = usable / len(drivers) if drivers else 0.0
    age = state.get("state_age_s")
    reasons: list[str] = []
    if not drivers: reasons.append("no_drivers")
    if coverage < 0.8: reasons.append("low_field_pace_coverage")
    if _finite(age) and float(age) > 20: reasons.append("stale_state")
    pace_available = bool(pace)
    if not pace_available: reasons.append("strict_pace_unavailable")
    uncertainty = bool(pace) and all(
        _finite(row.get("green_lap_lower_s")) and _finite(row.get("green_lap_upper_s"))
        for row in pace
    )
    if pace and not uncertainty: reasons.append("pace_uncertainty_incomplete")
    return PredictionQuality(
        state_age_s=float(age) if _finite(age) else None,
        pace_available=pace_available,
        uncertainty_available=uncertainty,
        field_coverage=coverage,
        degraded=bool(reasons),
        reasons=tuple(reasons),
    )

def build_driver_forecasts(
    state: dict[str, Any],
    race_predictions: list[dict[str, Any]],
    pace_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    drivers = {int(row["driver_number"]): row for row in state.get("drivers") or [] if row.get("driver_number") is not None}
    race = {int(row["driver_number"]): row for row in race_predictions if row.get("driver_number") is not None}
    pace = {int(row["driver_number"]): row for row in pace_predictions or [] if row.get("driver_number") is not None}
    output = []
    for number in sorted(set(drivers) | set(race) | set(pace)):
        d, r, p = drivers.get(number, {}), race.get(number, {}), pace.get(number, {})
        predicted = p.get("predicted_green_lap_s")
        recent = p.get("recent_median_5_s")
        pace_delta = float(predicted) - float(recent) if _finite(predicted) and _finite(recent) else None
        output.append({
            "driver_number": number,
            "label": r.get("label") or d.get("acronym") or d.get("full_name") or f"#{number}",
            "position": d.get("position"),
            "expected_position": r.get("expected_position"),
            "win_probability": r.get("win_probability"),
            "podium_probability": r.get("podium_probability"),
            "top10_probability": r.get("top10_probability"),
            "dnf_probability": r.get("dnf_probability"),
            "predicted_green_lap_s": predicted,
            "green_lap_lower_s": p.get("green_lap_lower_s"),
            "green_lap_upper_s": p.get("green_lap_upper_s"),
            "pace_delta_vs_recent_median_s": pace_delta,
            "p_green": p.get("p_green"),
            "p_pit": p.get("p_pit"),
            "p_neutralized": p.get("p_neutralized"),
        })
    return output

def prediction_envelope(
    *,
    state: dict[str, Any],
    race_predictions: list[dict[str, Any]],
    pace_predictions: list[dict[str, Any]] | None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = assess_quality(state, pace_predictions)
    return {
        "schema_version": 2,
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "forecast_at": state.get("updated_at"),
        "quality": asdict(quality),
        "drivers": build_driver_forecasts(state, race_predictions, pace_predictions),
        "model": model_metadata or {},
        "semantics": {
            "pace": "next green-lap conditional pace",
            "race": "Monte Carlo marginal outcome distribution",
            "interval": "empirical conformal interval; not Gaussian confidence interval",
        },
    }
