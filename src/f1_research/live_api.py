"""Read-only HTTP gateway for the live F1 intelligence stack.

This process is intended to run beside the persistent OpenF1 capture worker, not on
Vercel. Vercel/Next.js should proxy to it with a server-side token so model files,
raw capture files and provider credentials never reach the browser.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .data_truth import assert_trusted_live_state, audit_payload
from .live_intelligence import combined_live_report, combined_pit_windows, load_strict_artifact
from .live_protocol import encode as encode_live_envelope
from .live_protocol import envelope as live_envelope
from .live_quality import classify as classify_live_quality
from .model_registry import sha256_file
from .monitoring import snapshot as monitoring_snapshot
from .prediction_ledger import append_jsonl, make_record, sha256_json
from .race_control import STATES
from .reliability import reliability_overrides_from_state
from .strategy import SimulationConfig, compare_pit_windows, predict_from_state
from .strategy_calibration import load_simulation_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"
DEFAULT_PRIORS = ROOT / "reports" / "local" / "lap-strict" / "strategy_priors.json"
DEFAULT_STRICT_RELEASE_MANIFEST = (
    ROOT / "reports" / "local" / "lap-strict" / "strict_release_manifest.json"
)
DEFAULT_EVIDENCE = ROOT / "reports" / "local" / "model_evidence.json"
DEFAULT_PREDICTION_LEDGER = ROOT / "reports" / "local" / "live" / "predictions.jsonl"
MAX_STATE_BYTES = 20 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_STRICT_RELEASE_BYTES = 4 * 1024 * 1024
MAX_TELEMETRY_TAIL_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_LIVE_AGE_S = 20.0

app = FastAPI(title="F1 Race Intelligence API", version="1.0.0", docs_url="/docs")

_artifact_cache: dict[str, Any] = {"key": None, "value": None, "release": None}


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


def _strict_release_manifest_path() -> Path:
    return _path("F1_STRICT_RELEASE_MANIFEST_PATH", DEFAULT_STRICT_RELEASE_MANIFEST)


def _evidence_path() -> Path:
    return _path("F1_MODEL_EVIDENCE_PATH", DEFAULT_EVIDENCE)


def _prediction_ledger_path() -> Path:
    return _path("F1_PREDICTION_LEDGER_PATH", DEFAULT_PREDICTION_LEDGER)



def _max_live_age_s() -> float:
    raw = os.environ.get("F1_MAX_LIVE_AGE_S", str(DEFAULT_MAX_LIVE_AGE_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("F1_MAX_LIVE_AGE_S must be numeric") from exc
    if not np.isfinite(value) or not 1.0 <= value <= 300.0:
        raise RuntimeError("F1_MAX_LIVE_AGE_S must be between 1 and 300 seconds")
    return value


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be an explicit boolean")


def _require_live_stream() -> bool:
    return _env_flag("F1_REQUIRE_LIVE_STREAM", default=True)


def _authorize(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("F1_API_TOKEN")
    if not expected:
        if _env_flag("F1_ALLOW_UNAUTHENTICATED_API", default=False):
            return
        raise HTTPException(
            status_code=503,
            detail="F1_API_TOKEN is not configured; unauthenticated API access is disabled",
        )
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
    model_release = value.get("model_release")
    if not isinstance(model_release, dict):
        raise HTTPException(status_code=503, detail="Model evidence has no verified model release binding")
    git_sha = str(model_release.get("git_sha") or "").strip().lower()
    if len(git_sha) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in git_sha
    ):
        raise HTTPException(status_code=503, detail="Model evidence release Git SHA is invalid")
    for field in (
        "model_sha256",
        "model_manifest_sha256",
        "feature_schema_sha256",
        "training_data_sha256",
        "calibration_sha256",
    ):
        _valid_sha256(model_release.get(field), field=f"model_release.{field}")
    if not str(model_release.get("model_id") or "").strip():
        raise HTTPException(status_code=503, detail="Model evidence release model_id is missing")
    return value


def _canonical_json_sha256(value: Any) -> str:
    return sha256_json(value)


def _valid_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise HTTPException(status_code=503, detail=f"{field} is not a valid SHA-256 digest")
    return digest


def _read_strict_release_manifest() -> dict[str, Any]:
    path = _strict_release_manifest_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Strict release manifest is not installed")
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_STRICT_RELEASE_BYTES:
            raise HTTPException(status_code=503, detail="Strict release manifest failed size validation")
        value = json.loads(path.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Strict release manifest is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise HTTPException(status_code=503, detail="Strict release manifest schema is unsupported")
    if value.get("evidence_kind") != "strict_live_pace_release":
        raise HTTPException(status_code=503, detail="Strict release manifest kind is unsupported")
    if value.get("feature_policy") != "strict_asof_only":
        raise HTTPException(status_code=503, detail="Strict release feature policy is unsupported")
    if value.get("retrospective_stint_features_used") is not False:
        raise HTTPException(
            status_code=503,
            detail="Strict release permits retrospective stint features",
        )
    return value


def _file_signature(path: Path) -> tuple[str, int | None, int | None]:
    if not path.exists():
        return (str(path), None, None)
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _strict_runtime_key() -> tuple[Any, ...]:
    return (
        _file_signature(_model_path()),
        _file_signature(_priors_path()),
        _file_signature(_strict_release_manifest_path()),
        _env_flag("F1_ALLOW_UNVERIFIED_STRICT_MODEL", default=False),
    )


def _verify_strict_release(artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    if _env_flag("F1_ALLOW_UNVERIFIED_STRICT_MODEL", default=False):
        return {
            "verified": False,
            "production_eligible": False,
            "override": "F1_ALLOW_UNVERIFIED_STRICT_MODEL",
        }

    manifest = _read_strict_release_manifest()
    if manifest.get("artifact_schema_version") != 4:
        raise HTTPException(status_code=503, detail="Strict release requires artifact schema v4")
    pace_mode = manifest.get("pace_prediction_mode")
    if pace_mode not in {"recent_median_5_baseline", "recent_median_5_residual"}:
        raise HTTPException(status_code=503, detail="Strict release pace prediction mode is unsupported")
    baseline_guard = manifest.get("baseline_guard")
    if not isinstance(baseline_guard, dict):
        raise HTTPException(status_code=503, detail="Strict release baseline guard is missing")
    if baseline_guard.get("challenger_selected") is not (pace_mode == "recent_median_5_residual"):
        raise HTTPException(status_code=503, detail="Strict release baseline guard is inconsistent")
    model_path = _model_path()
    priors_path = _priors_path()
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Strict live pace model is unavailable")
    if not priors_path.exists():
        raise HTTPException(status_code=503, detail="Calibrated strategy priors are unavailable")

    model_sha = _valid_sha256(manifest.get("model_sha256"), field="model_sha256")
    priors_sha = _valid_sha256(
        manifest.get("strategy_priors_sha256"),
        field="strategy_priors_sha256",
    )
    feature_sha = _valid_sha256(
        manifest.get("feature_schema_sha256"),
        field="feature_schema_sha256",
    )
    source_sha = _valid_sha256(
        manifest.get("source_evidence_sha256"),
        field="source_evidence_sha256",
    )
    if sha256_file(model_path) != model_sha:
        raise HTTPException(status_code=503, detail="Strict model SHA-256 does not match release manifest")
    if sha256_file(priors_path) != priors_sha:
        raise HTTPException(
            status_code=503,
            detail="Strategy-prior SHA-256 does not match release manifest",
        )

    calibration_sessions = manifest.get("calibration_sessions")
    sealed_test_session = manifest.get("sealed_test_session")
    if (
        not isinstance(calibration_sessions, list)
        or len(calibration_sessions) < 2
        or any(not isinstance(item, int) or item <= 0 for item in calibration_sessions)
        or len(set(calibration_sessions)) != len(calibration_sessions)
        or not isinstance(sealed_test_session, int)
        or sealed_test_session <= 0
        or sealed_test_session in calibration_sessions
    ):
        raise HTTPException(
            status_code=503,
            detail="Strict release calibration/test session contract is invalid",
        )

    radii = manifest.get("conformal_radii_s")
    required_levels = ("0.50", "0.80", "0.90", "0.95")
    if not isinstance(radii, dict) or set(radii) != set(required_levels):
        raise HTTPException(status_code=503, detail="Strict release conformal levels are incomplete")
    radius_values = []
    for level in required_levels:
        try:
            value = float(radii[level])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Strict release conformal radius is invalid") from exc
        if not math.isfinite(value) or value <= 0:
            raise HTTPException(status_code=503, detail="Strict release conformal radius is invalid")
        radius_values.append(value)
    if radius_values != sorted(radius_values):
        raise HTTPException(status_code=503, detail="Strict release conformal radii are not monotonic")

    if artifact is not None:
        if artifact.get("schema_version") != 4:
            raise HTTPException(status_code=503, detail="Strict model schema is not v4")
        if artifact.get("task") != "next_lap_strict_mixture":
            raise HTTPException(status_code=503, detail="Strict model task does not match release contract")
        features = artifact.get("features")
        if not isinstance(features, list) or _canonical_json_sha256(features) != feature_sha:
            raise HTTPException(status_code=503, detail="Strict model feature schema hash mismatch")
        if artifact.get("retrospective_stint_features_used") is not False:
            raise HTTPException(status_code=503, detail="Strict model uses retrospective stint features")
        if artifact.get("pace_prediction_mode") != pace_mode:
            raise HTTPException(status_code=503, detail="Strict model pace prediction mode mismatch")
        if artifact.get("baseline_guard") != baseline_guard:
            raise HTTPException(status_code=503, detail="Strict model baseline guard mismatch")
        if pace_mode == "recent_median_5_baseline" and artifact.get("pace_regressor") is not None:
            raise HTTPException(status_code=503, detail="Baseline strict model unexpectedly carries a regressor")
        if pace_mode == "recent_median_5_residual" and not hasattr(
            artifact.get("pace_regressor"), "predict"
        ):
            raise HTTPException(status_code=503, detail="Residual strict model is missing its regressor")
        if artifact.get("calibration_sessions") != calibration_sessions:
            raise HTTPException(status_code=503, detail="Strict model calibration sessions mismatch")
        if artifact.get("sealed_test_session") != sealed_test_session:
            raise HTTPException(status_code=503, detail="Strict model sealed-test session mismatch")
        artifact_radii = artifact.get("conformal_radii_s")
        if not isinstance(artifact_radii, dict):
            raise HTTPException(status_code=503, detail="Strict model conformal radii are missing")
        for level, expected in zip(required_levels, radius_values, strict=True):
            try:
                actual = float(artifact_radii[level])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Strict model conformal radii mismatch",
                ) from exc
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise HTTPException(status_code=503, detail="Strict model conformal radii mismatch")

    return {
        "verified": True,
        "production_eligible": True,
        "git_sha": manifest.get("git_sha"),
        "model_sha256": model_sha,
        "strategy_priors_sha256": priors_sha,
        "source_evidence_sha256": source_sha,
        "artifact_schema_version": 4,
        "pace_prediction_mode": pace_mode,
        "baseline_guard": baseline_guard,
        "validation_scope": manifest.get("validation_scope"),
        "prospective_validation": manifest.get("prospective_validation"),
        "calibration_sessions": calibration_sessions,
        "sealed_test_session": sealed_test_session,
        "conformal_radii_s": radii,
    }


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
    key = _strict_runtime_key()
    if _artifact_cache["key"] != key:
        artifact = load_strict_artifact(path)
        release = _verify_strict_release(artifact)
        _artifact_cache["value"] = artifact
        _artifact_cache["release"] = release
        _artifact_cache["key"] = key
    return _artifact_cache["value"]


def _strict_release_metadata() -> dict[str, Any]:
    artifact = _load_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Strict live pace model is unavailable")
    release = _artifact_cache.get("release")
    if not isinstance(release, dict):
        raise HTTPException(status_code=503, detail="Strict release verification is unavailable")
    return release


def _simulation_config(samples: int) -> tuple[SimulationConfig, dict[str, Any]]:
    path = _priors_path()
    if not path.exists():
        if not _env_flag("F1_ALLOW_DEFAULT_PRIORS", default=False):
            raise HTTPException(
                status_code=503,
                detail="Calibrated strategy priors are unavailable; built-in defaults are disabled",
            )
        return SimulationConfig(samples=samples), {
            "source": "built_in_defaults",
            "warning": (
                "strategy_priors.json is unavailable; explicit research override "
                "F1_ALLOW_DEFAULT_PRIORS is active"
            ),
            "production_eligible": False,
        }
    config, payload = load_simulation_config(path, samples=samples)
    return config, {"source": str(path), "production_eligible": True, **payload}


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



def _locations(limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in reversed(_tail_lines(_events_path(), max_bytes=12 * 1024 * 1024)):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = str(item.get("topic") or "").removeprefix("v1/")
        if topic != "location":
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            number = int(payload.get("driver_number"))
            x = float(payload.get("x"))
            y = float(payload.get("y"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        output.append({
            "date": payload.get("date") or item.get("received_at"),
            "driver_number": number,
            "x": x,
            "y": y,
        })
        if len(output) >= limit:
            break
    output.reverse()
    return _safe(output)


def _record_live_prediction(state: dict[str, Any], report: dict[str, Any], pace_status: str) -> str | None:
    """Persist production predictions only when the strict model and capture manifest are present."""
    model_path = _model_path()
    manifest_path = _manifest_path()
    if pace_status != "strict_model" or not model_path.exists() or not manifest_path.exists():
        return None
    cutoff = state.get("latest_provider_event_at") or state.get("updated_at")
    if not isinstance(cutoff, str) or not cutoff:
        return None
    record = make_record(
        event_id=str(state.get("session_key") or "unknown"),
        forecast_origin="live_race_state",
        model_id="strict_live_pace+race_simulator",
        model_sha256=sha256_file(model_path),
        features={
            "session_key": state.get("session_key"),
            "current_lap": state.get("current_lap"),
            "state_updated_at": state.get("updated_at"),
            "provider_cutoff_at": cutoff,
            "state_sha256": sha256_json(state),
        },
        evidence_sha256=sha256_file(manifest_path),
        cutoff_at=cutoff,
        payload={
            "analysis_kind": report.get("analysis_kind"),
            "predictions": report.get("predictions"),
            "pace_predictions": report.get("pace_predictions"),
            "strategy_prior_source": report.get("strategy_prior_source"),
        },
    )
    append_jsonl(_prediction_ledger_path(), record)
    return record.prediction_id


def _live_report(total_laps: int, samples: int) -> dict[str, Any]:
    state = _read_state()
    truth_audit = _trusted_live_audit(state)
    config, prior_audit = _simulation_config(samples)
    reliability_model = prior_audit.get("reliability") if isinstance(prior_audit, dict) else None
    reliability_overrides = reliability_overrides_from_state(state, reliability_model)
    artifact = _load_artifact()
    if artifact is None and not _env_flag("F1_ALLOW_PACE_FALLBACK", default=False):
        raise HTTPException(
            status_code=503,
            detail="Strict live pace model is unavailable; recent-lap fallback is disabled",
        )
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
    report["prediction_id"] = _record_live_prediction(state, report, pace_status)
    return _safe(report)


@app.get("/healthz")
def healthz(_: None = Depends(_authorize)) -> JSONResponse:
    state_path = _state_path()
    model_path = _model_path()
    priors_path = _priors_path()
    evidence_path = _evidence_path()
    strict_release_path = _strict_release_manifest_path()
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

    strict_release_verified = False
    strict_release_error = None
    strict_release = None
    if model_path.exists():
        try:
            strict_release = _strict_release_metadata()
            strict_release_verified = bool(strict_release.get("verified"))
        except HTTPException as exc:
            strict_release_error = str(exc.detail)

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
        "strict_release_manifest": strict_release_path.exists(),
        "strict_release_verified": strict_release_verified,
        "strict_release_error": strict_release_error,
        "strict_release_git_sha": (
            strict_release.get("git_sha") if isinstance(strict_release, dict) else None
        ),
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



@app.get("/readyz")
def readyz(_: None = Depends(_authorize)) -> JSONResponse:
    state = _read_state()
    truth = _trusted_live_audit(state)
    missing = []
    strict_release = None
    try:
        artifact = _load_artifact()
        if artifact is None:
            missing.append("strict_model")
        else:
            strict_release = _strict_release_metadata()
    except HTTPException:
        missing.append("strict_release_contract")
    if not _priors_path().exists():
        missing.append("strategy_priors")
    try:
        evidence = _read_model_evidence()
    except HTTPException:
        missing.append("model_evidence")
        evidence = None
    if missing:
        raise HTTPException(status_code=503, detail={"missing": sorted(set(missing))})
    return JSONResponse(_safe({
        "ready": True,
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "data_truth": truth,
        "strict_release": strict_release,
        "model_evidence_run": evidence.get("benchmark_run_id") if isinstance(evidence, dict) else None,
    }))


@app.get("/providerz")
def providerz(_: None = Depends(_authorize)) -> JSONResponse:
    state = _read_state()
    manifest = _read_capture_manifest()
    stream = manifest.get("stream") if isinstance(manifest.get("stream"), dict) else {}
    quality = classify_live_quality(
        state_age_s=_state_age_s(state),
        provider_age_s=_age_s(state.get("latest_provider_event_at")),
        connection_state=stream.get("connection_state"),
        max_age_s=_max_live_age_s(),
    )
    status_code = 200 if quality.status == "LIVE" else 503
    return JSONResponse(_safe({
        "provider": "OpenF1",
        "status": quality.status,
        "reasons": list(quality.reasons),
        "connection_state": stream.get("connection_state"),
        "last_message_age_s": _age_s(stream.get("last_message_at")),
        "provider_event_age_s": _age_s(state.get("latest_provider_event_at")),
    }), status_code=status_code)


@app.get("/modelz")
def modelz(_: None = Depends(_authorize)) -> JSONResponse:
    artifact = _load_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Strict live pace model is unavailable")
    strict_release = _strict_release_metadata()
    evidence = _read_model_evidence()
    return JSONResponse(_safe({
        "ready": True,
        "live_pace_model": {
            "artifact_schema_version": artifact.get("schema_version") if isinstance(artifact, dict) else None,
            "selected_regressor": artifact.get("selected_regressor") if isinstance(artifact, dict) else None,
            "evidence_scope": "strict live pace artifact; separate from race-outcome benchmark evidence",
            "release": strict_release,
        },
        "race_outcome_model_evidence": {
            "evidence_kind": evidence.get("evidence_kind"),
            "sealed_test_events": evidence.get("sealed_test_events"),
            "benchmark_run_id": evidence.get("benchmark_run_id"),
            "source_provenance_sha256": evidence.get("source_provenance_sha256"),
            "model_release": evidence.get("model_release"),
        },
    }))


def _websocket_auth_close_code(authorization: str | None) -> int | None:
    expected = os.environ.get("F1_API_TOKEN")
    if not expected:
        return None if _env_flag("F1_ALLOW_UNAUTHENTICATED_API", default=False) else 1013
    return None if authorization == f"Bearer {expected}" else 4401


@app.websocket("/v1/ws")
async def live_socket(websocket: WebSocket) -> None:
    close_code = _websocket_auth_close_code(websocket.headers.get("authorization"))
    if close_code is not None:
        await websocket.close(code=close_code)
        return
    await websocket.accept()
    sequence = 0
    last_hash = None
    try:
        while True:
            try:
                state = _read_state()
            except HTTPException:
                await asyncio.sleep(0.5)
                continue
            state_hash = live_envelope(
                sequence=1, event_type="state_probe", state=state, payload={}
            ).state_sha256
            if state_hash != last_hash:
                sequence += 1
                item = live_envelope(
                    sequence=sequence,
                    event_type="state_update",
                    state=state,
                    payload={
                        "session_key": state.get("session_key"),
                        "current_lap": state.get("current_lap"),
                        "updated_at": state.get("updated_at"),
                    },
                    provider_time=state.get("latest_provider_event_at"),
                )
                await websocket.send_text(encode_live_envelope(item))
                last_hash = state_hash
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.get("/v1/race-control")
def race_control_summary(_: None = Depends(_authorize)) -> JSONResponse:
    state = _read_state()
    normalized = state.get("track_state")
    if normalized not in STATES:
        normalized = "UNKNOWN"
    return JSONResponse(_safe({
        "schema_version": 1,
        "state": normalized,
        "changed_at": state.get("track_state_changed_at"),
        "message": state.get("track_state_message"),
        "raw": {
            "session_status": state.get("status"),
            "flag": state.get("flag"),
            "safety_car": state.get("safety_car"),
        },
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



@app.get("/v1/predictions/history")
def prediction_history(
    limit: int = Query(default=200, ge=1, le=2000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    rows: list[dict[str, Any]] = []
    for line in reversed(_tail_lines(_prediction_ledger_path(), max_bytes=8 * 1024 * 1024)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("prediction_id"):
            rows.append(row)
            if len(rows) >= limit:
                break
    rows.reverse()
    return JSONResponse(_safe({"count": len(rows), "predictions": rows}))


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



@app.get("/v1/locations")
def locations(
    limit: int = Query(default=5000, ge=100, le=20000),
    _: None = Depends(_authorize),
) -> JSONResponse:
    return JSONResponse({"samples": _locations(limit)})


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
    if artifact is None and not _env_flag("F1_ALLOW_PACE_FALLBACK", default=False):
        raise HTTPException(
            status_code=503,
            detail="Strict live pace model is unavailable; recent-lap strategy fallback is disabled",
        )
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