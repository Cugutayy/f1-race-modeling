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

    for name in ("OPENF1_TOKEN", "OPENF1_USERNAME", "OPENF1_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setattr(uvicorn, "run", fake_run)

    railway_app.main()

    assert called.pop("app") is railway_app.app
    assert called == {
        "host": "0.0.0.0",
        "port": 9123,
        "log_level": "info",
    }


def test_capture_autostart_requires_real_provider_credentials(monkeypatch):
    for name in ("OPENF1_TOKEN", "OPENF1_USERNAME", "OPENF1_PASSWORD", "F1_CAPTURE_AUTOSTART"):
        monkeypatch.delenv(name, raising=False)

    assert railway_app._has_openf1_credentials() is False
    assert railway_app._capture_autostart_enabled() is False

    monkeypatch.setenv("OPENF1_TOKEN", "provider-token")
    assert railway_app._has_openf1_credentials() is True
    assert railway_app._capture_autostart_enabled() is True

    monkeypatch.setenv("F1_CAPTURE_AUTOSTART", "off")
    assert railway_app._capture_autostart_enabled() is False


def test_capture_accepts_username_password_pair_and_uses_persistent_volume(monkeypatch):
    monkeypatch.delenv("OPENF1_TOKEN", raising=False)
    monkeypatch.setenv("OPENF1_USERNAME", "user@example.test")
    monkeypatch.setenv("OPENF1_PASSWORD", "secret")
    monkeypatch.delenv("F1_CAPTURE_AUTOSTART", raising=False)
    monkeypatch.delenv("F1_CAPTURE_OUTPUT", raising=False)
    monkeypatch.delenv("F1_CAPTURE_SESSION_KEY", raising=False)

    assert railway_app._capture_autostart_enabled() is True
    command = railway_app._capture_command()
    assert command[1:4] == ["-m", "f1_research.openf1_live", "capture"]
    assert command[4:] == ["--output", "/data/live", "--session-key", "latest"]

    monkeypatch.setenv("F1_CAPTURE_OUTPUT", "/tmp/capture")
    monkeypatch.setenv("F1_CAPTURE_SESSION_KEY", "12345")
    assert railway_app._capture_command()[4:] == [
        "--output",
        "/tmp/capture",
        "--session-key",
        "12345",
    ]
