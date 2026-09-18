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
        "run_id": "sealed-abc",
    })
    response = TestClient(live_api.app).get("/modelz")
    assert response.status_code == 200
    assert response.json()["sealed_test_events"] == 12
    assert response.json()["run_id"] == "sealed-abc"
