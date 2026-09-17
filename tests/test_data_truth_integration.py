import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from f1_research import live_api
from f1_research.strategy import SimulationConfig, drivers_from_state


def _drivers():
    return [
        {
            "driver_number": 1,
            "acronym": "A",
            "position": 1,
            "gap_to_leader_s": 0.0,
            "recent_laps_s": [90.0, 89.9, 90.1],
            "last_lap_s": 90.1,
            "compound": "MEDIUM",
            "tyre_age": 10,
            "pit_stops": 0,
        },
        {
            "driver_number": 2,
            "acronym": "B",
            "position": 2,
            "gap_to_leader_s": 2.2,
            "recent_laps_s": [90.2, 90.3, 90.1],
            "last_lap_s": 90.1,
            "compound": "HARD",
            "tyre_age": 15,
            "pit_stops": 0,
        },
    ]


def _write_gateway(tmp_path, monkeypatch, *, provider_age_s=0, stream_age_s=0, connected=True):
    now = datetime.now(UTC)
    state = {
        "session_key": 123,
        "session_name": "Race",
        "current_lap": 10,
        "updated_at": now.isoformat(),
        "latest_provider_event_at": (now - timedelta(seconds=provider_age_s)).isoformat(),
        "drivers": _drivers(),
    }
    manifest = {
        "schema_version": 3,
        "captured_rows": 20,
        "stream": {
            "connection_state": "connected" if connected else "reconnecting",
            "last_message_at": (now - timedelta(seconds=stream_age_s)).isoformat(),
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("F1_LIVE_STATE_PATH", str(state_path))
    monkeypatch.setenv("F1_LIVE_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("F1_LIVE_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("F1_STRICT_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("F1_STRATEGY_PRIORS_PATH", str(tmp_path / "missing-priors.json"))
    monkeypatch.setenv("F1_MAX_LIVE_AGE_S", "20")
    monkeypatch.setenv("F1_REQUIRE_LIVE_STREAM", "1")
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    live_api._artifact_cache["key"] = None
    live_api._artifact_cache["value"] = None
    return state_path, manifest_path


def test_live_prediction_rejects_old_provider_event_even_when_state_file_was_written_now(tmp_path, monkeypatch):
    _write_gateway(tmp_path, monkeypatch, provider_age_s=120, stream_age_s=0)
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=30, samples=1000)
    assert exc.value.status_code == 503
    assert "stale" in str(exc.value.detail).lower()


def test_live_prediction_rejects_disconnected_or_stale_transport(tmp_path, monkeypatch):
    _write_gateway(tmp_path, monkeypatch, provider_age_s=0, stream_age_s=0, connected=False)
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=30, samples=1000)
    assert exc.value.status_code == 503
    assert "not connected" in str(exc.value.detail).lower()

    _, manifest_path = _write_gateway(
        tmp_path, monkeypatch, provider_age_s=0, stream_age_s=90, connected=True
    )
    assert manifest_path.exists()
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=30, samples=1000)
    assert exc.value.status_code == 503
    assert "fresh messages" in str(exc.value.detail).lower()


def test_strategy_rejects_missing_nonleader_gap_instead_of_generating_one():
    state = {"drivers": _drivers()}
    state["drivers"][1]["gap_to_leader_s"] = None
    with pytest.raises(ValueError, match="gap_to_leader_s"):
        drivers_from_state(state, SimulationConfig(samples=1000))


def test_strategy_rejects_unknown_tyre_state_instead_of_medium_age_zero_defaults():
    state = {"drivers": _drivers()}
    state["drivers"][1]["compound"] = None
    state["drivers"][1]["tyre_age"] = None
    with pytest.raises(ValueError) as exc:
        drivers_from_state(state, SimulationConfig(samples=1000))
    message = str(exc.value)
    assert "compound" in message
    assert "tyre_age" in message


def test_health_reports_data_truth_failure_without_crashing(tmp_path, monkeypatch):
    _write_gateway(tmp_path, monkeypatch, provider_age_s=120, stream_age_s=0)
    payload = json.loads(live_api.healthz(None).body)
    assert payload["ok"] is True
    assert payload["trusted_live_ready"] is False
    assert payload["provider_event_age_s"] >= 100
    assert "stale" in payload["trusted_live_error"].lower()
