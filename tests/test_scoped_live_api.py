from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from f1_research import scoped_live_api
from f1_research.strategy import SimulationConfig


def _state():
    now = datetime.now(UTC).isoformat()
    return {
        "session_key": 9693,
        "current_lap": 30,
        "updated_at": now,
        "latest_provider_event_at": now,
        "drivers": [
            {
                "driver_number": 1, "acronym": "AAA", "position": 1,
                "gap_to_leader_s": 0.0, "laps_behind": None,
                "recent_laps_s": [89.0, 89.1], "last_lap_s": 89.1,
                "compound": "MEDIUM", "tyre_age": 8, "pit_stops": 0,
            },
            {
                "driver_number": 4, "acronym": "BBB", "position": 2,
                "gap_to_leader_s": 3.5, "laps_behind": None,
                "recent_laps_s": [89.2, 89.3], "last_lap_s": 89.3,
                "compound": "HARD", "tyre_age": 14, "pit_stops": 1,
            },
            {
                "driver_number": 14, "acronym": "CCC", "position": 3,
                "gap_to_leader_s": None, "laps_behind": 1,
                "gap_to_leader_raw": "+1 LAP",
            },
        ],
    }


def _patch_common(monkeypatch):
    monkeypatch.setattr(scoped_live_api.base, "_read_state", _state)
    monkeypatch.setattr(
        scoped_live_api.base,
        "_trusted_live_audit",
        lambda _state: {
            "status": "trusted_live",
            "simulation_eligible_drivers": 2,
            "classification_only_drivers": 1,
        },
    )
    monkeypatch.setattr(
        scoped_live_api.base,
        "_simulation_config",
        lambda samples: (SimulationConfig(samples=samples), {"source": "test"}),
    )
    monkeypatch.setattr(scoped_live_api.base, "_load_artifact", lambda: None)
    monkeypatch.setattr(
        scoped_live_api,
        "reliability_overrides_from_state",
        lambda _state, _model: {},
    )


def test_live_report_simulates_exact_time_subset_but_returns_full_observed_state(monkeypatch):
    _patch_common(monkeypatch)
    captured = {}

    def fake_predict(snapshot, total_laps, **_kwargs):
        captured["drivers"] = [row["driver_number"] for row in snapshot["drivers"]]
        captured["total_laps"] = total_laps
        return {"predictions": [], "audit": {}}

    monkeypatch.setattr(scoped_live_api, "predict_from_state", fake_predict)
    report = scoped_live_api._scoped_live_report(57, 1000)

    assert captured == {"drivers": [1, 4], "total_laps": 57}
    assert [row["driver_number"] for row in report["state"]["drivers"]] == [1, 4, 14]
    assert report["classification_only"][0]["driver_number"] == 14
    assert report["classification_only"][0]["laps_behind"] == 1
    assert report["audit"]["probability_scope"] == "exact_time_lead_lap_conditional"
    assert report["audit"]["invented_lap_deficit_seconds"] is False
    assert report["simulation_scope"]["included_driver_numbers"] == [1, 4]


def test_strategy_endpoint_rejects_classification_only_driver(monkeypatch):
    _patch_common(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        scoped_live_api.strategy(driver_number=14, total_laps=57, samples=1000, _=None)
    assert exc.value.status_code == 409
    assert "classification-only" in str(exc.value.detail)
