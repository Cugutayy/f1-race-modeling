import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from f1_research import live_api


def _state(updated_at: str | None = None, provider_event_at: str | None = None):
    updated_at = updated_at or datetime.now(UTC).isoformat()
    provider_event_at = provider_event_at or updated_at
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
        "latest_provider_event_at": provider_event_at,
        "rejected_stale_messages": 2,
        "rejected_provider_order_messages": 1,
        "rejected_invalid_timestamp_messages": 0,
        "drivers": drivers,
    }


def _evidence():
    return {
        "schema_version": 1,
        "evidence_kind": "retrospective_sealed_historical_benchmark",
        "benchmark_run_id": "abc123",
        "model_release": {
            "git_sha": "a" * 40,
            "model_id": "modern::catboost",
            "model_sha256": "b" * 64,
            "model_manifest_sha256": "c" * 64,
            "feature_schema_sha256": "d" * 64,
            "training_data_sha256": "e" * 64,
            "calibration_sha256": "f" * 64,
            "trained_until": "2026-01-01T00:00:00+00:00",
        },
        "provider": "Jolpica",
        "years": [2022, 2023, 2024, 2025, 2026],
        "protocol": "fit -> tuning -> calibration -> sealed test",
        "sealed_test_events": 12,
        "sealed_test_event_ids": [f"R{i}" for i in range(12)],
        "selected_modern": {"name": "catboost", "params": {"depth": 4}},
        "ensemble_weights": {"modern": 0.75, "qualifying": 0.25, "pl": 0.0},
        "models": [
            {"model": "rank_ensemble", "winner_log_loss": 1.08, "position_mae": 3.19},
            {"model": "qualifying_order", "winner_log_loss": 1.30, "position_mae": 3.14},
        ],
        "uncertainty": None,
        "limitations": ["Retrospective evidence is not a prospective guarantee."],
    }


@pytest.fixture
def gateway_files(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    events_path = tmp_path / "events.jsonl"
    manifest_path = tmp_path / "manifest.json"
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
    manifest_path.write_text(json.dumps({
        "schema_version": 3,
        "captured_rows": 1234,
        "capture_bytes": 98765,
        "stream": {
            "connection_state": "connected",
            "connect_count": 2,
            "disconnect_count": 1,
            "last_message_at": datetime.now(UTC).isoformat(),
            "last_error": None,
        },
    }), encoding="utf-8")
    monkeypatch.setenv("F1_LIVE_STATE_PATH", str(state_path))
    monkeypatch.setenv("F1_LIVE_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("F1_LIVE_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("F1_STRICT_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    monkeypatch.setenv("F1_STRATEGY_PRIORS_PATH", str(tmp_path / "missing-priors.json"))
    monkeypatch.setenv("F1_MODEL_EVIDENCE_PATH", str(tmp_path / "model_evidence.json"))
    monkeypatch.setenv("F1_MAX_LIVE_AGE_S", "20")
    monkeypatch.setenv("F1_REQUIRE_LIVE_STREAM", "1")
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.delenv("F1_ALLOW_UNAUTHENTICATED_API", raising=False)
    monkeypatch.delenv("F1_ALLOW_DEFAULT_PRIORS", raising=False)
    monkeypatch.delenv("F1_ALLOW_PACE_FALLBACK", raising=False)
    live_api._artifact_cache["key"] = None
    live_api._artifact_cache["value"] = None
    return state_path, events_path, manifest_path


def test_gateway_authorization_fails_closed_unless_explicitly_overridden(monkeypatch):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.delenv("F1_ALLOW_UNAUTHENTICATED_API", raising=False)
    with pytest.raises(HTTPException) as missing:
        live_api._authorize(None)
    assert missing.value.status_code == 503
    assert "not configured" in str(missing.value.detail)

    monkeypatch.setenv("F1_ALLOW_UNAUTHENTICATED_API", "1")
    assert live_api._authorize(None) is None

    monkeypatch.setenv("F1_API_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        live_api._authorize(None)
    assert exc.value.status_code == 401
    assert live_api._authorize("Bearer secret-token") is None


def test_websocket_authorization_matches_fail_closed_http_policy(monkeypatch):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.delenv("F1_ALLOW_UNAUTHENTICATED_API", raising=False)
    assert live_api._websocket_auth_close_code(None) == 1013

    monkeypatch.setenv("F1_ALLOW_UNAUTHENTICATED_API", "1")
    assert live_api._websocket_auth_close_code(None) is None

    monkeypatch.setenv("F1_API_TOKEN", "secret-token")
    assert live_api._websocket_auth_close_code(None) == 4401
    assert live_api._websocket_auth_close_code("Bearer wrong-token") == 4401
    assert live_api._websocket_auth_close_code("Bearer secret-token") is None


def test_live_report_has_coherent_fallback_probabilities_only_with_research_overrides(
    gateway_files, monkeypatch
):
    monkeypatch.setenv("F1_ALLOW_DEFAULT_PRIORS", "1")
    monkeypatch.setenv("F1_ALLOW_PACE_FALLBACK", "1")
    report = live_api._live_report(total_laps=12, samples=1000)
    assert report["pace_status"] == "fallback_recent_laps"
    assert report["pace_model"]["status"] == "fallback_recent_laps"
    assert report["state"]["session_key"] == 99
    assert report["data_truth"]["status"] == "trusted_live"
    predictions = report["predictions"]
    assert len(predictions) == 3
    assert sum(row["win_probability"] for row in predictions) == pytest.approx(1.0)
    podium_total = sum(row["podium_probability"] for row in predictions)
    top10_total = sum(row["top10_probability"] for row in predictions)
    assert podium_total == pytest.approx(report["audit"]["expected_podium_slots_filled"])
    assert top10_total == pytest.approx(report["audit"]["expected_top10_slots_filled"])
    assert 0.0 <= podium_total <= min(3, len(predictions))
    assert 0.0 <= top10_total <= len(predictions)
    assert report["strategy_prior_source"]["source"] == "built_in_defaults"




def test_live_report_rejects_missing_calibrated_priors_by_default(gateway_files):
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert exc.value.status_code == 503
    assert "strategy priors" in str(exc.value.detail).lower()


def test_live_report_rejects_missing_strict_model_by_default(gateway_files, monkeypatch):
    monkeypatch.setattr(
        live_api,
        "_simulation_config",
        lambda samples: (live_api.SimulationConfig(samples=samples), {"source": "fixture"}),
    )
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert exc.value.status_code == 503
    assert "strict live pace model" in str(exc.value.detail).lower()

def test_live_report_rejects_stale_provider_event_even_when_file_is_fresh(gateway_files):
    state_path = gateway_files[0]
    now = datetime.now(UTC)
    state_path.write_text(
        json.dumps(_state(
            updated_at=now.isoformat(),
            provider_event_at=(now - timedelta(seconds=60)).isoformat(),
        )),
        encoding="utf-8",
    )
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert exc.value.status_code == 503
    assert "stale" in str(exc.value.detail).lower()


def test_live_report_rejects_disconnected_or_stale_stream(gateway_files):
    manifest_path = gateway_files[2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stream"]["connection_state"] = "reconnecting"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert exc.value.status_code == 503
    assert "not connected" in str(exc.value.detail).lower()


def test_live_report_rejects_missing_gap_instead_of_fabricating_one(gateway_files):
    state_path = gateway_files[0]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["drivers"][1]["gap_to_leader_s"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert exc.value.status_code == 503
    assert "gap_to_leader_s" in str(exc.value.detail)


def test_live_and_strategy_reject_unknown_tyre_state(gateway_files):
    state_path = gateway_files[0]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["drivers"][1]["compound"] = None
    state["drivers"][1]["tyre_age"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(HTTPException) as live_exc:
        live_api._live_report(total_laps=12, samples=1000)
    assert live_exc.value.status_code == 503
    assert "compound" in str(live_exc.value.detail)
    assert "tyre_age" in str(live_exc.value.detail)

    with pytest.raises(HTTPException) as strategy_exc:
        live_api.strategy(driver_number=1, total_laps=12, samples=1000, _=None)
    assert strategy_exc.value.status_code == 503


def test_telemetry_tail_filters_driver_and_ignores_malformed_lines(gateway_files):
    samples = live_api._telemetry(driver_number=1, limit=10)
    assert [row["speed_kmh"] for row in samples] == [280, 301]
    assert [row["gear"] for row in samples] == [7, 8]
    assert all(row["date"] for row in samples)


def test_healthz_exposes_transport_state_and_evidence_availability(gateway_files):
    response = live_api.healthz(None)
    payload = json.loads(response.body)
    assert payload["ok"] is True
    assert payload["strict_model"] is False
    assert payload["strategy_priors"] is False
    assert payload["model_evidence"] is False
    assert payload["session_key"] == 99
    assert payload["current_lap"] == 5
    assert payload["connection_state"] == "connected"
    assert payload["live_stream_healthy"] is True
    assert payload["trusted_live_ready"] is True
    assert payload["trusted_live_error"] is None
    assert payload["provider_event_age_s"] <= 20
    assert payload["last_message_age_s"] <= 20
    assert payload["connect_count"] == 2
    assert payload["disconnect_count"] == 1
    assert payload["capture_rows"] == 1234
    assert payload["rejected_stale_messages"] == 2
    assert payload["rejected_provider_order_messages"] == 1
    assert payload["rejected_invalid_timestamp_messages"] == 0


def test_healthz_does_not_call_stale_transport_trusted_live(gateway_files):
    manifest_path = gateway_files[2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stream"]["last_message_at"] = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = json.loads(live_api.healthz(None).body)
    assert payload["connection_state"] == "connected"
    assert payload["last_message_age_s"] >= 59
    assert payload["live_stream_healthy"] is False
    assert payload["trusted_live_ready"] is False
    assert "fresh messages" in payload["trusted_live_error"]


def test_evidence_endpoint_returns_valid_sealed_artifact(gateway_files):
    path = live_api._evidence_path()
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    payload = json.loads(live_api.evidence(None).body)
    assert payload["sealed_test_events"] == 12
    assert payload["selected_modern"]["name"] == "catboost"
    assert payload["models"][0]["model"] == "rank_ensemble"
    health = json.loads(live_api.healthz(None).body)
    assert health["model_evidence"] is True


def test_evidence_endpoint_reports_missing_artifact(gateway_files):
    with pytest.raises(HTTPException) as exc:
        live_api.evidence(None)
    assert exc.value.status_code == 404


def test_evidence_endpoint_rejects_malformed_schema(gateway_files):
    path = live_api._evidence_path()
    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        live_api.evidence(None)
    assert exc.value.status_code == 503
