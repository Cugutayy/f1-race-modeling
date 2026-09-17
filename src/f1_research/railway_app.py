"""Railway-facing ASGI wrapper around the protected live API.

Railway's platform health checker cannot attach our bearer token. Keep the
application's operational `/healthz` and every data endpoint protected, while
exposing only a minimal process-liveness probe for infrastructure.

When real OpenF1 credentials are configured, this wrapper also supervises the
capture process in the same container. That lets the API and capture worker share
the Railway `/data` volume without exposing provider credentials to the browser.
No credentials means no live capture is started; the API remains available and
continues to report unavailable/stale live state honestly.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI

from .live_api import app as live_app

LOGGER = logging.getLogger("f1_research.railway")
FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

app: FastAPI = live_app

_capture_stop = threading.Event()
_capture_thread: threading.Thread | None = None
_capture_process: subprocess.Popen[str] | None = None
_capture_lock = threading.Lock()


@app.get("/railway_healthz", include_in_schema=False)
def railway_healthz() -> dict[str, bool]:
    """Return process liveness only; do not expose race state or credentials."""
    return {"ok": True}


def _has_openf1_credentials() -> bool:
    """Return whether authenticated OpenF1 live streaming can be attempted."""
    if os.environ.get("OPENF1_TOKEN", "").strip():
        return True
    return bool(
        os.environ.get("OPENF1_USERNAME", "").strip()
        and os.environ.get("OPENF1_PASSWORD", "").strip()
    )


def _capture_autostart_enabled() -> bool:
    """Autostart only with real provider credentials and no explicit opt-out."""
    raw = os.environ.get("F1_CAPTURE_AUTOSTART", "auto").strip().lower()
    if raw in FALSE_VALUES:
        return False
    return _has_openf1_credentials()


def _capture_command() -> list[str]:
    output = Path(os.environ.get("F1_CAPTURE_OUTPUT", "/data/live"))
    session_key = os.environ.get("F1_CAPTURE_SESSION_KEY", "latest").strip() or "latest"
    return [
        sys.executable,
        "-m",
        "f1_research.openf1_live",
        "capture",
        "--output",
        str(output),
        "--session-key",
        session_key,
    ]


def _terminate_capture_process() -> None:
    global _capture_process
    with _capture_lock:
        process = _capture_process
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _capture_supervisor() -> None:
    """Restart the provider capture process without taking down the API."""
    global _capture_process
    restart_delay = max(2.0, float(os.environ.get("F1_CAPTURE_RESTART_DELAY_S", "15")))
    command = _capture_command()
    while not _capture_stop.is_set():
        LOGGER.info("Starting authenticated OpenF1 capture into %s", command[5])
        try:
            process = subprocess.Popen(command, text=True)
        except OSError:
            LOGGER.exception("Unable to start OpenF1 capture process")
            if _capture_stop.wait(restart_delay):
                break
            continue
        with _capture_lock:
            _capture_process = process
        while process.poll() is None and not _capture_stop.wait(0.5):
            pass
        if _capture_stop.is_set():
            _terminate_capture_process()
            break
        return_code = process.poll()
        LOGGER.warning(
            "OpenF1 capture exited with code %s; retrying in %.1fs",
            return_code,
            restart_delay,
        )
        with _capture_lock:
            if _capture_process is process:
                _capture_process = None
        if _capture_stop.wait(restart_delay):
            break
    with _capture_lock:
        _capture_process = None


def _start_capture_supervisor() -> None:
    global _capture_thread
    if not _capture_autostart_enabled():
        LOGGER.info("OpenF1 live capture is disabled because credentials are absent or autostart is off")
        return
    if _capture_thread is not None and _capture_thread.is_alive():
        return
    _capture_stop.clear()
    thread = threading.Thread(
        target=_capture_supervisor,
        name="openf1-capture-supervisor",
        daemon=True,
    )
    _capture_thread = thread
    thread.start()


def _stop_capture_supervisor() -> None:
    global _capture_thread
    _capture_stop.set()
    _terminate_capture_process()
    thread = _capture_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=10)
    _capture_thread = None


app.add_event_handler("startup", _start_capture_supervisor)
app.add_event_handler("shutdown", _stop_capture_supervisor)


def main() -> None:
    """Run the Railway wrapper without relying on shell expansion of ``$PORT``."""
    import uvicorn

    host = os.environ.get("F1_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("F1_API_PORT", "8000")))
    uvicorn.run("f1_research.railway_app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
