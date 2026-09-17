"""Immutable prediction ledger for auditable live and historical forecasts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    created_at: str
    event_id: str
    forecast_origin: str
    model_id: str
    model_sha256: str
    feature_sha256: str
    evidence_sha256: str
    cutoff_at: str
    payload: dict[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def make_record(*, event_id: str, forecast_origin: str, model_id: str,
                model_sha256: str, features: dict[str, Any], evidence_sha256: str,
                cutoff_at: str, payload: dict[str, Any], created_at: str | None = None) -> PredictionRecord:
    if not all(isinstance(v, str) and v.strip() for v in
               (event_id, forecast_origin, model_id, model_sha256, evidence_sha256, cutoff_at)):
        raise ValueError("prediction identity/provenance fields must be non-empty strings")
    feature_sha = sha256_json(features)
    body = {
        "event_id": event_id, "forecast_origin": forecast_origin, "model_id": model_id,
        "model_sha256": model_sha256, "feature_sha256": feature_sha,
        "evidence_sha256": evidence_sha256, "cutoff_at": cutoff_at, "payload": payload,
    }
    return PredictionRecord(
        prediction_id=sha256_json(body)[:24],
        created_at=created_at or datetime.now(UTC).isoformat(),
        **body,
    )


def append_jsonl(path: Path, record: PredictionRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(asdict(record)).decode()
    if path.exists():
        # Fail closed if an id was already emitted with different bytes.
        for line in path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            if old.get("prediction_id") == record.prediction_id:
                if sha256_json({k: v for k, v in old.items() if k != "created_at"}) != sha256_json(
                    {k: v for k, v in asdict(record).items() if k != "created_at"}
                ):
                    raise ValueError("prediction_id collision with different immutable content")
                return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
