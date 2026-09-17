"""Prediction orchestration: trusted state -> simulation -> immutable evidence ledger."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_truth import assert_trusted_live_state
from .prediction_ledger import append_jsonl, make_record, sha256_json
from .simulation_scope import annotate_simulation_report, split_simulation_scope
from .strategy import SimulationConfig, predict_from_state


def forecast(*, state: dict[str, Any], total_laps: int, model_id: str,
             model_sha256: str, evidence_sha256: str, cutoff_at: str,
             ledger_path: Path, samples: int = 20_000,
             max_age_s: float = 20.0) -> dict[str, Any]:
    audit = assert_trusted_live_state(state, max_age_s=max_age_s)
    simulation_state, classification_only = split_simulation_scope(state)
    report = predict_from_state(simulation_state, total_laps, config=SimulationConfig(samples=samples))
    annotate_simulation_report(report, classification_only)
    features = {
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "state_sha256": sha256_json(state),
        "simulation_eligible_drivers": audit.simulation_eligible_drivers,
        "classification_only_drivers": audit.classification_only_drivers,
    }
    payload = {
        "session_key": state.get("session_key"),
        "current_lap": state.get("current_lap"),
        "simulation": report,
    }
    record = make_record(
        event_id=str(state.get("session_key") or "unknown"),
        forecast_origin="live_race_state",
        model_id=model_id,
        model_sha256=model_sha256,
        features=features,
        evidence_sha256=evidence_sha256,
        cutoff_at=cutoff_at,
        payload=payload,
    )
    append_jsonl(ledger_path, record)
    return {
        "schema_version": 1,
        "prediction_id": record.prediction_id,
        "feature_sha256": record.feature_sha256,
        "model_sha256": model_sha256,
        "evidence_sha256": evidence_sha256,
        "report": report,
    }
