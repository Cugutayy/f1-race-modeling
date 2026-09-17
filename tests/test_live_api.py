import json
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from f1_research import live_api


def _state(updated_at: str | None = None):
    updated_at = updated_at or datetime.now(UTC).isoformat()
    drivers = []
    for number, pace, gap in ((1, 89.90, 0.0), (2, 90.10, 2.4), (3, 90.30, 5.1)):
        drivers.append({
            "driver_number": number,
            "acronym": f"D{number}",
            "position": number,
            "gap_to_leader_s": gap,
            "recent_laps_s": [pace + 0.08, pace + 0.04, pace, pace + 0.02, pace + 0.05],
            "last_lap_s": pace + 0.05,
            "compound": "MEDIUM",
            "tyre_age": 8 + number,
            "pit_stops": 0,
        })
    return {
        "session_key": 99,
        "session_name": "Race",
        "current_lap": 5,
        "updated_at": updated_at,
        "drivers": drivers,
    }


@pytest.fixture
def gateway_files(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    events_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    events = [
        {"topic": "v1/car_data", "received_at": "2026-09-17T12:00:00+00:00", "payload": {
            "driver_number": 1, "date": "2026-09-17T12:00:00+00:00", "speed": 280,
            "throttle": 75, "brake": 0, "rpm": 11000, "n_gear": 7, "drs": 12,
        }},
        {"topic": "car_data", "received_at": "2026-09-17T12:00:01+00:00", "payload": {
            "driver_number": 2, "date": "2026-09-17T12:00:01+00:00", "speed": 271,
            "throttle": 63, "brake": 0, "rpm": 10700, "n_gear": 7, "drs": 10,
        }},
        {"topic": "car_data", "received_at": "2026-09-17T12:00:02+00:00", "payload": {
            "driver_number": 1, "date": "2026-09-17T12:00:02+00:00", "speed": 301,
            "throttle": 100, "brake": 0, "rpm": 11800, "n_gear": 8, "drs": 12,
        }},
    ]
    events_path.write_text(
        "\n".join([json.dumps(events[0]), "{malformed", json.dumps(events[1]), json.dumps(events[2])]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("F1_LIVE_STATE_PATH", str(state_path))
    monkeypatch.setenv("F1_LIVE_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("F1_STRICT_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    monkeypatch.setenv("F1_STRATEGY_PRIORS_PATH", str(tmp_path / "missing-priors.json"))
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    live_api._artifact_cache["key"] = None
    live_api._artifact_cache["value"] = None
    return state_path, events_path


def test_gateway_authorization_is_optional_but_enforced_when_configured(monkeypatch):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    assert live_api._authorize(None) is None

    monkeypatch.setenv("F1_API_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        live_api._authorize(None)
    assert exc.value.status_code == 401
    assert live_api._authorize("Bearer secret-token") is None


def test_live_report_has_coherent_fallback_probabilities(gateway_files):
    report = live_api._live_report(total_laps=12, samples=1000)
    assert report["pace_status"] == "fallback_recent_laps"
    assert report["pace_model"]["status"] == "fallback_recent_laps"
    assert report["state"]["session_key"] == 99
    predictions = report["predictions"]
    assert len(predictions) == 3
    assert sum(row["win_probability"] for row in predictions) == pytest.approx(1.0)
    assert sum(row["podium_probability"] for row in predictions) == pytest.approx(3.0)
    assert report["strategy_prior_source"]["source"] == "built_in_defaults"


def test_telemetry_tail_filters_driver_and_ignores_malformed_lines(gateway_files):
    samples = live_api._telemetry(driver_number=1, limit=10)
    assert [row["speed_kmh"] for row in samples] == [280, 301]
    assert [row["gear"] for row in samples] == [7, 8]
    assert all(row["date"] for row in samples)


def test_healthz_exposes_state_without_requiring_model(gateway_files):
    response = live_api.healthz(None)
    payload = json.loads(response.body)
    assert payload["ok"] is True
    assert payload["strict_model"] is False
    assert payload["strategy_priors"] is False
    assert payload["session_key"] == 99
    assert payload["current_lap"] == 5
