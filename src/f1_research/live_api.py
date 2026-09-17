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

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from .data_truth import assert_trusted_live_state, audit_payload
from .live_intelligence import combined_live_report, combined_pit_windows, load_strict_artifact
from .live_quality import classify as classify_live_quality
from .reliability import reliability_overrides_from_state
from .monitoring import snapshot as monitoring_snapshot
from .strategy import SimulationConfig, compare_pit_windows, predict_from_state
from .strategy_calibration import load_simulation_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"
DEFAULT_PRIORS = ROOT / "reports" / "local" / "lap-strict" / "strategy_priors.json"
DEFAULT_EVIDENCE = ROOT / "reports" / "local" / "model_evidence.json"
MAX_STATE_BYTES = 20 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_TELEMETRY_TAIL_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_LIVE_AGE_S = 20.0

app = FastAPI(title="F1 Race Intelligence API", version="1.0.0", docs_url="/docs")

_artifact_cache: dict[str, Any] = {"key": None, "value": None}


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser().resolve()


def _state_path() -> Path:
    return _path("F1_LIVE_STATE_PATH", DEFAULT_STATE)


def _events_path() -> Path:
    configured = os.environ.get("F1_LIVE_EVENTS_PATH")
    return Path(configured).expanduser().resolve() if configured else _state_path().with_name("events.jsonl")


def _manifest_path() -> Path:
    configured = os.environ.get("F1_LIVE_MANIFEST_PATH")
    return Path(configured).expanduser().resolve() if configured else _state_path().with_name("manifest.json")


def _model_path() -> Path:
    return _path("F1_STRICT_MODEL_PATH", DEFAULT_MODEL)


def _priors_path() -> Path:
    return _path("F1_STRATEGY_PRIORS_PATH", DEFAULT_PRIORS)


def _evidence_path() -> Path:
    return _path("F1_MODEL_EVIDENCE_PATH", DEFAULT_EVIDENCE)


def _max_live_age_s() -> float:
    raw = os.environ.get("F1_MAX_LIVE_AGE_S", str(DEFAULT_MAX_LIVE_AGE_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("F1_MAX_LIVE_AGE_S must be numeric") from exc
    if not np.isfinite(value) or not 1.0 <= value <= 300.0:
        raise RuntimeError("F1_MAX_LIVE_AGE_S must be between 1 and 300 seconds")
    return value


def _require_live_stream() -> bool:
    raw = os.environ.get("F1_REQUIRE_LIVE_STREAM", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


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


def _read_capture_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        if path.stat().st_size <= 0 or path.stat().st_size > MAX_MANIFEST_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_model_evidence() -> dict[str, Any]:
    path = _evidence_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model evidence artifact is not installed")
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_EVIDENCE_BYTES:
            raise HTTPException(status_code=503, detail="Model evidence failed size validation")
        value = json.loads(path.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Model evidence is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise HTTPException(status_code=503, detail="Model evidence schema is unsupported")
    if value.get("evidence_kind") != "retrospective_sealed_historical_benchmark":
        raise HTTPException(status_code=503, detail="Model evidence kind is unsupported")
    if not isinstance(value.get("models"), list) or not isinstance(value.get("sealed_test_events"), int):
        raise HTTPException(status_code=503, detail="Model evidence is incomplete")
    return value


def _age_s(raw: Any) -> float | None:
    if not raw:
        return None
    try:
        observed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - observed.astimezone(UTC)).total_seconds())


def _state_age_s(state: dict[str, Any]) -> float | None:
    return _age_s(state.get("updated_at"))


def _trusted_live_audit(state: dict[str, Any]) -> dict[str, Any]:
    max_age = _max_live_age_s()
    try:
        state_audit = assert_trusted_live_state(state, max_age_s=max_age)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    manifest = _read_capture_manifest()
    stream = manifest.get("stream") if isinstance(manifest.get("stream"), dict) else {}
    connection_state = stream.get("connection_state")
    last_message_age = _age_s(stream.get("last_message_at"))
    if _require_live_stream():
        if connection_state != "connected":
            raise HTTPException(
                status_code=503,
                detail=f"Live provider stream is not connected: {connection_state or 'unknown'}",
            )
        if last_message_age is None or last_message_age > max_age:
            raise HTTPException(
                status_code=503,
                detail="Live provider stream has no fresh messages",
            )

    return {
        **audit_payload(state_audit),
        "transport_required": _require_live_stream(),
        "connection_state": connection_state,
        "last_message_age_s": last_message_age,
    }


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
    truth_audit = _trusted_live_audit(state)
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
    report["data_truth"] = truth_audit
    report["strategy_prior_source"] = prior_audit
    report["pace_status"] = pace_status
    return _safe(report)


@app.get("/healthz")
def healthz(_: None = Depends(_authorize)) -> JSONResponse:
    state_path = _state_path()
    model_path = _model_path()
    priors_path = _priors_path()
    evidence_path = _evidence_path()
    state = _read_state() if state_path.exists() else {}
    manifest = _read_capture_manifest()
    stream = manifest.get("stream") if isinstance(manifest.get("stream"), dict) else {}
    connection_state = stream.get("connection_state")
    last_message_age_s = _age_s(stream.get("last_message_at"))
    live_stream_healthy = bool(
        connection_state == "connected"
        and last_message_age_s is not None
        and last_message_age_s <= _max_live_age_s()
    )
    trusted_live_ready = False
    trusted_live_error = None
    if state_path.exists():
        try:
            _trusted_live_audit(state)
            trusted_live_ready = True
        except HTTPException as exc:
            trusted_live_error = str(exc.detail)
    quality = classify_live_quality(state_age_s=_state_age_s(state), provider_age_s=_age_s(state.get("latest_provider_event_at")), connection_state=connection_state, max_age_s=_max_live_age_s())
    return JSONResponse(_safe({
        "ok": state_path.exists(),
        "quality_status": quality.status,
        "quality_reasons": list(quality.reasons),
        "state_path": str(state_path),
        "state_age_s": _state_age_s(state),
        "provider_event_age_s": _age_s(state.get("latest_provider_event_at")),
        "strict_model": model_path.exists(),
        "strategy_priors": priors_path.exists(),
        "model_evidence": evidence_path.exists(),
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "rejected_stale_messages": state.get("rejected_stale_messages", 0),
        "rejected_provider_order_messages": state.get("rejected_provider_order_messages", 0),
        "rejected_invalid_timestamp_messages": state.get("rejected_invalid_timestamp_messages", 0),
        "capture_rows": manifest.get("captured_rows"),
        "capture_bytes": manifest.get("capture_bytes"),
        "connection_state": connection_state,
        "live_stream_healthy": live_stream_healthy,
        "trusted_live_ready": trusted_live_ready,
        "trusted_live_error": trusted_live_error,
        "max_live_age_s": _max_live_age_s(),
        "last_message_age_s": last_message_age_s,
        "connect_count": stream.get("connect_count"),
        "disconnect_count": stream.get("disconnect_count"),
        "last_stream_error": stream.get("last_error"),
    }))


@app.get("/v1/metrics")
def metrics(_: None = Depends(_authorize)) -> JSONResponse:
    state = _read_state()
    return JSONResponse(_safe(monitoring_snapshot(
        state,
        state_age_s=_state_age_s(state),
        provider_age_s=_age_s(state.get("latest_provider_event_at")),
    )))


@app.get("/v1/evidence")
def evidence(_: None = Depends(_authorize)) -> JSONResponse:
    return JSONResponse(_safe(_read_model_evidence()))


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
    truth_audit = _trusted_live_audit(state)
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
        "data_truth": truth_audit,
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