"""Contract-first frontend view model; keeps observed state separate from model output."""
from __future__ import annotations

from typing import Any


def build_dashboard_payload(*, state: dict[str, Any], prediction: dict[str, Any] | None,
                            quality_status: str, race_control: dict[str, Any] | None = None) -> dict[str, Any]:
    drivers = state.get("drivers") if isinstance(state.get("drivers"), list) else []
    predicted = {}
    if prediction:
        report = prediction.get("report") or prediction
        for row in report.get("predictions") or []:
            if isinstance(row, dict) and row.get("driver_number") is not None:
                predicted[int(row["driver_number"])] = row
    leaderboard = []
    for row in sorted(
        (x for x in drivers if isinstance(x, dict) and isinstance(x.get("position"), int)),
        key=lambda x: x["position"],
    ):
        number = int(row["driver_number"])
        forecast = predicted.get(number, {})
        leaderboard.append({
            "driver_number": number,
            "label": row.get("acronym") or row.get("full_name") or str(number),
            "position": row["position"],
            "gap_to_leader_s": row.get("gap_to_leader_s"),
            "laps_behind": row.get("laps_behind"),
            "compound": row.get("compound"),
            "tyre_age": row.get("tyre_age"),
            "pit_stops": row.get("pit_stops"),
            "win_probability": forecast.get("win_probability"),
            "podium_probability": forecast.get("podium_probability"),
            "expected_position": forecast.get("expected_position"),
        })
    return {
        "schema_version": 1,
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "quality_status": quality_status,
        "race_control": race_control or {"state": "UNKNOWN"},
        "leaderboard": leaderboard,
        "observed_state_updated_at": state.get("updated_at"),
        "prediction_id": prediction.get("prediction_id") if prediction else None,
        "disclaimer": "Observed provider state and modelled probabilities are distinct fields.",
    }
