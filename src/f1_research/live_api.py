"""Read-only HTTP gateway for the live F1 intelligence stack.

This process is intended to run beside the persistent OpenF1 capture worker, not on
Vercel. Vercel/Next.js should proxy to it with a server-side token so model files,
raw capture files and provider credentials never reach the browser.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from .live_intelligence import combined_live_report, combined_pit_windows, load_strict_artifact
from .reliability import reliability_overrides_from_state
from .strategy import SimulationConfig, compare_pit_windows, predict_from_state
from .strategy_calibration import load_simulation_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"
DEFAULT_PRIORS = ROOT / "reports" / "local" / "lap-strict" / "strategy_priors.json"
MAX_STATE_BYTES = 20 * 1024 * 1024
MAX_TELEMETRY_TAIL_BYTES = 4 * 1024 * 1024

app = FastAPI(title="F1 Race Intelligence API", version="1.0.0", docs_url="/docs")

_artifact_cache: dict[str, Any] = {"key": None, "value": None}


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser().resolve()


def _state_path() -> Path:
    return _path("F1_LIVE_STATE_PATH", DEFAULT_STATE)


def _events_path() -> Path:
    configured = os.environ.get("F1_LIVE_EVENTS_PATH")
    return Path(configured).expanduser().resolve() if configured else _state_path().with_name("events.jsonl")


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


def _tail_lines(path: Path, max_bytes: int = MAX_TELEMETRY_TAIL_BYTES) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read()
    if start > 0:
        newline = data.find(b"\n")
        data = data[newline + 1:] if newline >= 0 else b""
    return data.decode("utf-8", errors="replace").splitlines()


def _telemetry(driver_number: int, limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in reversed(_tail_lines(_events_path())):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = str(item.get("topic") or "").removeprefix("v1/")
        if topic != "car_data":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            number = int(payload.get("driver_number"))
        except (TypeError, ValueError):
            continue
        if number != driver_number:
            continue
        output.append({
            "date": payload.get("date") or item.get("received_at"),
            "speed_kmh": payload.get("speed"),
            "throttle_pct": payload.get("throttle"),
            "brake": payload.get("brake"),
            "rpm": payload.get("rpm"),
            "gear": payload.get("n_gear"),
            "drs": payload.get("drs"),
        })
        if len(output) >= limit:
            break
    output.reverse()
    return _safe(output)


def _live_report(total_laps: int, samples: int) -> dict[str, Any]:
    state = _read_state()
    config, prior_audit = _simulation_config(samples)
    reliability_model = prior_audit.get("reliability") if isinstance(prior_audit, dict) else None
    reliability_overrides = reliability_overrides_from_state(state, reliability_model)
    artifact = _load_artifact()
    try:
        if artifact is not None:
            report = combined_live_report(
                state,
                total_laps,
                artifact,
                config=config,
                reliability_model=reliability_model,
            )
            pace_status = "strict_model"
        else:
            report = predict_from_state(
                state,
                total_laps,
                config=config,
                dnf_hazard_overrides=reliability_overrides,
            )
            report["pace_predictions"] = []
            report["pace_model"] = {
                "task": None,
                "status": "fallback_recent_laps",
                "warning": "Strict model artifact is unavailable",
            }
            report["reliability_model"] = {
                "enabled": bool(reliability_model and reliability_model.get("enabled")),
                "override_drivers": sorted(reliability_overrides),
                "source": (
                    "hierarchical_public_results_survival"
                    if reliability_overrides
                    else "pooled_config_fallback"
                ),
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


@app.get("/v1/telemetry")
def telemetry(
    driver_number: int = Query(ge=1, le=999),
    limit: int = Query(default=500, ge=10, le=2000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    return JSONResponse({
        "driver_number": driver_number,
        "samples": _telemetry(driver_number, limit),
    })


@app.get("/v1/strategy")
def strategy(
    driver_number: int = Query(ge=1, le=999),
    total_laps: int = Query(ge=2, le=100),
    samples: int = Query(default=4000, ge=1000, le=50000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    state = _read_state()
    config, prior_audit = _simulation_config(samples)
    reliability_model = prior_audit.get("reliability") if isinstance(prior_audit, dict) else None
    reliability_overrides = reliability_overrides_from_state(state, reliability_model)
    artifact = _load_artifact()
    try:
        if artifact is not None:
            scenarios = combined_pit_windows(
                state,
                total_laps,
                driver_number,
                artifact,
                config=config,
                reliability_model=reliability_model,
            )
            pace_status = "strict_model"
        else:
            scenarios = compare_pit_windows(
                state,
                total_laps,
                driver_number,
                config=config,
                dnf_hazard_overrides=reliability_overrides,
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
        "reliability_status": (
            "hierarchical_public_results_survival"
            if reliability_overrides
            else "pooled_config_fallback"
        ),
        "strategy_prior_source": prior_audit,
        "scenarios": scenarios,
    }))


def main() -> None:
    """Run the persistent-worker API locally or on an always-on host."""
    import uvicorn

    host = os.environ.get("F1_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("F1_API_PORT", "8000")))
    uvicorn.run("f1_research.live_api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
