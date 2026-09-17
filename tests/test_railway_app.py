import pytest
from fastapi import HTTPException

from f1_research import live_api
from f1_research.railway_app import railway_healthz


def test_railway_liveness_is_minimal_and_main_api_auth_stays_enforced(monkeypatch):
    assert railway_healthz() == {"ok": True}

    monkeypatch.setenv("F1_API_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc:
        live_api._authorize(None)
    assert exc.value.status_code == 401
    assert live_api._authorize("Bearer secret-token") is None
