"""Read-only HTTP gateway for the live F1 intelligence stack.

This process is intended to run beside the persistent OpenF1 capture worker, not on
Vercel. Vercel/Next.js should proxy to it with a server-side token so model files,
raw capture files and provider credentials never reach the browser.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from .live_intelligence import combined_live_report, combined_pit_windows, load_strict_artifact
from .strategy import SimulationConfig, predict_from_state
from .strategy_calibration import load_simulation_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"
DEFAULT_PRIORS = ROOT / "reports" / "local" / "lap-strict" / "strategy_priors.json"
MAX_STATE_BYTES = 20 * 1024 * 1024

app = FastAPI(title="F1 Race Intelligence API", version="1.0.0", docs_url="/docs")

_artifact_cache: dict[str, Any] = {"key": None, "value": None}


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser().resolve()


def _state_path() -> Path:
    return _path("F1_LIVE_STATE_PATH", DEFAULT_STATE)


def _model_path() -> Path:
    return _path("F1_STRICT_MODEL_PATH", DEFAULT_MODEL)


def _priors_path() -> Path:
    return _path("F1_STRATEGY_PRIORS_PATH", DEFAULT_PRIORS)


def _authorize(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("F1_API_TOKEN")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _read_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Live state is unavailable")
    size = path.stat().st_size
    if size <= 0 or size > MAX_STATE_BYTES:
        raise HTTPException(status_code=503, detail="Live state failed size validation")
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Live state is unreadable") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=503, detail="Live state must be an object")
    return value


def _state_age_s(state: dict[str, Any]) -> float | None:
    raw = state.get("updated_at")
    if not raw:
        return None
    try:
        updated = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds())


def _load_artifact() -> dict[str, Any] | None:
    path = _model_path()
    if not path.exists():
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _artifact_cache["key"] != key:
        _artifact_cache["value"] = load_strict_artifact(path)
        _artifact_cache["key"] = key
    return _artifact_cache["value"]


def _simulation_config(samples: int) -> tuple[SimulationConfig, dict[str, Any]]:
    path = _priors_path()
    if not path.exists():
        return SimulationConfig(samples=samples), {
            "source": "built_in_defaults",
            "warning": "strategy_priors.json is unavailable",
        }
    config, payload = load_simulation_config(path, samples=samples)
    return config, {"source": str(path), **payload}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _live_report(total_laps: int, samples: int) -> dict[str, Any]:
    state = _read_state()
    config, prior_audit = _simulation_config(samples)
    artifact = _load_artifact()
    try:
        if artifact is not None:
            report = combined_live_report(state, total_laps, artifact, config=config)
            pace_status = "strict_model"
        else:
            report = predict_from_state(state, total_laps, config=config)
            report["pace_predictions"] = []
            report["pace_model"] = {
                "task": None,
                "status": "fallback_recent_laps",
                "warning": "Strict model artifact is unavailable",
            }
            pace_status = "fallback_recent_laps"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    report["state"] = state
    report["state_age_s"] = _state_age_s(state)
    report["strategy_prior_source"] = prior_audit
    report["pace_status"] = pace_status
    return _safe(report)


@app.get("/healthz")
def healthz(_: None = Depends(_authorize)) -> JSONResponse:
    state_path = _state_path()
    model_path = _model_path()
    priors_path = _priors_path()
    state = _read_state() if state_path.exists() else {}
    return JSONResponse(_safe({
        "ok": state_path.exists(),
        "state_path": str(state_path),
        "state_age_s": _state_age_s(state),
        "strict_model": model_path.exists(),
        "strategy_priors": priors_path.exists(),
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
    }))


@app.get("/v1/live")
def live(
    total_laps: int = Query(ge=2, le=100),
    samples: int = Query(default=4000, ge=1000, le=50000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    return JSONResponse(_live_report(total_laps, samples))


@app.get("/v1/strategy")
def strategy(
    driver_number: int = Query(ge=1, le=999),
    total_laps: int = Query(ge=2, le=100),
    samples: int = Query(default=4000, ge=1000, le=50000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    state = _read_state()
    config, prior_audit = _simulation_config(samples)
    artifact = _load_artifact()
    try:
        if artifact is not None:
            scenarios = combined_pit_windows(
                state,
                total_laps,
                driver_number,
                artifact,
                config=config,
            )
            pace_status = "strict_model"
        else:
            from .strategy import compare_pit_windows

            scenarios = compare_pit_windows(
                state,
                total_laps,
                driver_number,
                config=config,
            )
            pace_status = "fallback_recent_laps"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(_safe({
        "driver_number": driver_number,
        "session_key": state.get("session_key"),
        "state_updated_at": state.get("updated_at"),
        "state_age_s": _state_age_s(state),
        "pace_status": pace_status,
        "strategy_prior_source": prior_audit,
        "scenarios": scenarios,
    }))
