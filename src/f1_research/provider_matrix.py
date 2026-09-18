"""Aggregate real provider-audit artifacts into a fail-closed data-truth matrix.

This module never invents race observations. Every matrix row must come from a
persisted ``reconciliation.json`` produced by :mod:`provider_audit`; the source file
is hashed so the summary remains traceable to immutable evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .revision import validate_git_sha

EXPECTED_RECONCILIATION_SCHEMA = 4
EXPECTED_ARTIFACT_SCHEMA = 2
_ALLOWED_STATUS = {"PASS", "PASS_WITH_GAPS", "FAIL"}


@dataclass(frozen=True)
class MatrixEvent:
    year: int
    round_number: int
    openf1_session_key: int
    source: str
    source_sha256: str
    producer_git_sha: str
    verification_status: str
    passed: bool
    event_identity_verified: bool
    hard_mismatch_count: int
    warning_count: int
    insufficient_hard_count: int
    audit_gap_count: int
    insufficient_secondary_count: int
    provider_error_count: int
    provider_error_keys: tuple[str, ...]
    event_identity_failures: tuple[str, ...]
    reconciliation_performed: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(value).strip() not in {str(number), f"{number}.0"} and not isinstance(value, int):
        raise ValueError(f"{field} must be an exact integer")
    if number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _load_event(path: Path) -> MatrixEvent:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: reconciliation artifact must be a JSON object")

    schema = _exact_int(payload.get("schema_version"), field="schema_version", minimum=1)
    artifact_schema = _exact_int(
        payload.get("artifact_schema_version"), field="artifact_schema_version", minimum=1
    )
    reconciliation_raw = payload.get("reconciliation_schema_version")
    if schema != EXPECTED_RECONCILIATION_SCHEMA:
        raise ValueError(
            f"{path}: unsupported reconciliation schema {schema}; "
            f"expected {EXPECTED_RECONCILIATION_SCHEMA}"
        )
    if artifact_schema != EXPECTED_ARTIFACT_SCHEMA:
        raise ValueError(
            f"{path}: unsupported artifact schema {artifact_schema}; "
            f"expected {EXPECTED_ARTIFACT_SCHEMA}"
        )

    status = payload.get("verification_status")
    if status not in _ALLOWED_STATUS:
        raise ValueError(f"{path}: invalid verification_status {status!r}")
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        raise ValueError(f"{path}: passed must be explicit boolean")
    if passed != (status in {"PASS", "PASS_WITH_GAPS"}):
        raise ValueError(f"{path}: passed and verification_status disagree")

    # Collection/normalization can fail before reconciliation exists. The audit envelope
    # deliberately records reconciliation_schema_version=None in that case. A passing
    # artifact, or a FAIL produced after reconciliation, must still carry schema v4.
    if reconciliation_raw is None:
        if status != "FAIL":
            raise ValueError(f"{path}: passing artifact has no reconciliation schema")
        reconciliation_performed = False
    else:
        reconciliation_schema = _exact_int(
            reconciliation_raw, field="reconciliation_schema_version", minimum=1
        )
        if reconciliation_schema != schema:
            raise ValueError(
                f"{path}: unsupported reconciliation schema {schema}/{reconciliation_schema}; "
                f"expected {EXPECTED_RECONCILIATION_SCHEMA}"
            )
        reconciliation_performed = True

    identity = payload.get("event_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("verified"), bool):
        raise ValueError(f"{path}: event_identity.verified must be explicit boolean")
    if passed and identity["verified"] is not True:
        raise ValueError(f"{path}: a passing audit cannot have failed event identity")

    provider_errors = payload.get("provider_errors")
    if not isinstance(provider_errors, dict):
        raise ValueError(f"{path}: provider_errors must be an object")
    if passed and provider_errors:
        raise ValueError(f"{path}: a passing audit cannot contain provider errors")

    return MatrixEvent(
        year=_exact_int(payload.get("year"), field="year", minimum=1950),
        round_number=_exact_int(payload.get("round"), field="round", minimum=1),
        openf1_session_key=_exact_int(
            payload.get("openf1_session_key"), field="openf1_session_key", minimum=1
        ),
        source=str(path),
        source_sha256=_sha256(path),
        producer_git_sha=validate_git_sha(
            payload.get("producer_git_sha"),
            field="producer_git_sha",
        ),
        verification_status=status,
        passed=passed,
        event_identity_verified=identity["verified"],
        hard_mismatch_count=_exact_int(
            payload.get("hard_mismatch_count"), field="hard_mismatch_count"
        ),
        warning_count=_exact_int(payload.get("warning_count"), field="warning_count"),
        insufficient_hard_count=_exact_int(
            payload.get("insufficient_hard_count"), field="insufficient_hard_count"
        ),
        audit_gap_count=_exact_int(payload.get("audit_gap_count", 0), field="audit_gap_count"),
        insufficient_secondary_count=_exact_int(
            payload.get("insufficient_secondary_count"), field="insufficient_secondary_count"
        ),
        provider_error_count=len(provider_errors),
        provider_error_keys=tuple(sorted(str(key) for key in provider_errors)),
        event_identity_failures=tuple(str(item) for item in (identity.get("failures") or [])),
        reconciliation_performed=reconciliation_performed,
    )


def build_matrix(paths: Iterable[Path]) -> dict[str, Any]:
    events = [_load_event(Path(path)) for path in paths]
    if not events:
        raise ValueError("At least one reconciliation artifact is required")

    identities = [(event.year, event.round_number) for event in events]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate season/round artifacts are forbidden")
    session_keys = [event.openf1_session_key for event in events]
    if len(session_keys) != len(set(session_keys)):
        raise ValueError("Duplicate OpenF1 session keys are forbidden")

    producer_revisions = {event.producer_git_sha for event in events}
    if len(producer_revisions) != 1:
        raise ValueError("All provider-audit artifacts must come from the same Git revision")

    ordered = sorted(events, key=lambda event: (event.year, event.round_number))
    rows = [event.__dict__ for event in ordered]
    producer_git_sha = next(iter(producer_revisions))
    return {
        "matrix_schema_version": 1,
        "reconciliation_schema_version": EXPECTED_RECONCILIATION_SCHEMA,
        "artifact_schema_version": EXPECTED_ARTIFACT_SCHEMA,
        "producer_git_sha": producer_git_sha,
        "event_count": len(rows),
        "pass_count": sum(row["verification_status"] == "PASS" for row in rows),
        "pass_with_gaps_count": sum(
            row["verification_status"] == "PASS_WITH_GAPS" for row in rows
        ),
        "fail_count": sum(row["verification_status"] == "FAIL" for row in rows),
        "hard_mismatch_count": sum(row["hard_mismatch_count"] for row in rows),
        "audit_gap_count": sum(row["audit_gap_count"] for row in rows),
        "provider_error_count": sum(row["provider_error_count"] for row in rows),
        "events": rows,
        "policy": {
            "real_artifacts_only": True,
            "source_sha256_required": True,
            "majority_vote": False,
            "repair_disagreements": False,
            "missing_evidence_is_not_zero": True,
        },
    }


def write_matrix(matrix: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_truth_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    rows = matrix["events"]
    with (output / "data_truth_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate real provider reconciliation artifacts")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    matrix = build_matrix(args.artifacts)
    write_matrix(matrix, args.output)
    print(json.dumps({key: value for key, value in matrix.items() if key != "events"}, indent=2))


if __name__ == "__main__":
    main()
