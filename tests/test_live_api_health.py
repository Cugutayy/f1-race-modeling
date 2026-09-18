from pathlib import Path

from fastapi.testclient import TestClient

from f1_research import live_api


def test_readyz_requires_installed_production_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.setattr(live_api, "_read_state", lambda: {"session_key": 1, "current_lap": 2})
    monkeypatch.setattr(live_api, "_trusted_live_audit", lambda state: {"verified": True})
    monkeypatch.setattr(live_api, "_load_artifact", lambda: None)
    monkeypatch.setattr(live_api, "_priors_path", lambda: tmp_path / "missing-priors.json")
    monkeypatch.setattr(
        live_api, "_read_model_evidence",
        lambda: (_ for _ in ()).throw(live_api.HTTPException(status_code=404, detail="missing")),
    )
    response = TestClient(live_api.app).get("/readyz")
    assert response.status_code == 503
    assert set(response.json()["detail"]["missing"]) == {
        "strict_model", "strategy_priors", "model_evidence",
    }


def test_modelz_exposes_sealed_evidence(monkeypatch):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.setattr(live_api, "_load_artifact", lambda: {"schema_version": 7})
    monkeypatch.setattr(live_api, "_read_model_evidence", lambda: {
        "evidence_kind": "retrospective_sealed_historical_benchmark",
        "sealed_test_events": 12,
        "benchmark_run_id": "sealed-abc",
    })
    response = TestClient(live_api.app).get("/modelz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_pace_model"]["artifact_schema_version"] == 7
    assert payload["race_outcome_model_evidence"]["sealed_test_events"] == 12
    assert payload["race_outcome_model_evidence"]["benchmark_run_id"] == "sealed-abc"


def test_strict_live_prediction_is_written_to_immutable_ledger(monkeypatch, tmp_path):
    model = tmp_path / "pace.joblib"
    model.write_bytes(b"strict-model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"captured_rows": 10}')
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(live_api, "_model_path", lambda: model)
    monkeypatch.setattr(live_api, "_manifest_path", lambda: manifest)
    monkeypatch.setattr(live_api, "_prediction_ledger_path", lambda: ledger)
    state = {
        "session_key": 9693, "current_lap": 12,
        "updated_at": "2025-03-16T04:30:00Z",
        "latest_provider_event_at": "2025-03-16T04:29:59Z",
    }
    report = {
        "analysis_kind": "live_race_monte_carlo",
        "predictions": [{"driver_number": 1, "win_probability": 0.5}],
        "pace_predictions": [{"driver_number": 1, "predicted_green_lap_s": 90.0}],
        "strategy_prior_source": {"source": "test"},
    }
    prediction_id = live_api._record_live_prediction(state, report, "strict_model")
    assert prediction_id
    assert ledger.exists()
    import json
    row = json.loads(ledger.read_text().strip())
    assert row["prediction_id"] == prediction_id
    assert row["event_id"] == "9693"
    assert len(row["model_sha256"]) == 64
    assert len(row["evidence_sha256"]) == 64


def test_fallback_prediction_is_not_recorded_as_strict_model_evidence(monkeypatch, tmp_path):
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(live_api, "_prediction_ledger_path", lambda: ledger)
    prediction_id = live_api._record_live_prediction(
        {"session_key": 1, "updated_at": "2026-01-01T00:00:00Z"},
        {"predictions": []},
        "fallback_recent_laps",
    )
    assert prediction_id is None
    assert not ledger.exists()
