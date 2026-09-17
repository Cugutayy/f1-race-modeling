import pytest
import uvicorn
from fastapi import HTTPException

from f1_research import live_api, railway_app


def test_railway_liveness_is_minimal_and_main_api_auth_stays_enforced(monkeypatch):
    assert railway_app.railway_healthz() == {"ok": True}

    monkeypatch.setenv("F1_API_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        live_api._authorize(None)
    assert exc.value.status_code == 401
    assert live_api._authorize("Bearer secret-token") is None


def test_railway_entrypoint_reads_port_from_environment(monkeypatch):
    called = {}

    def fake_run(app, **kwargs):
        called["app"] = app
        called.update(kwargs)

    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setattr(uvicorn, "run", fake_run)

    railway_app.main()

    assert called == {
        "app": "f1_research.railway_app:app",
        "host": "0.0.0.0",
        "port": 9123,
        "log_level": "info",
    }
