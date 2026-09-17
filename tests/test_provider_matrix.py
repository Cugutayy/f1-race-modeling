import json
from pathlib import Path

import pytest

from f1_research.provider_matrix import build_matrix, write_matrix


def _artifact(path: Path, **overrides) -> Path:
    payload = {
        "schema_version": 4,
        "artifact_schema_version": 1,
        "reconciliation_schema_version": 4,
        "year": 2025,
        "round": 1,
        "openf1_session_key": 9693,
        "passed": True,
        "verification_status": "PASS_WITH_GAPS",
        "hard_mismatch_count": 0,
        "warning_count": 0,
        "insufficient_hard_count": 27,
        "insufficient_secondary_count": 80,
        "provider_errors": {},
        "event_identity": {"verified": True},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matrix_preserves_real_artifact_identity_and_hash(tmp_path: Path):
    source = _artifact(tmp_path / "reconciliation.json")
    matrix = build_matrix([source])

    assert matrix["event_count"] == 1
    assert matrix["pass_with_gaps_count"] == 1
    assert matrix["hard_mismatch_count"] == 0
    assert matrix["policy"]["real_artifacts_only"] is True
    row = matrix["events"][0]
    assert row["year"] == 2025
    assert row["round_number"] == 1
    assert row["openf1_session_key"] == 9693
    assert len(row["source_sha256"]) == 64

    output = tmp_path / "matrix"
    write_matrix(matrix, output)
    assert (output / "data_truth_matrix.json").exists()
    assert (output / "data_truth_matrix.csv").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 3}, "unsupported reconciliation schema"),
        ({"reconciliation_schema_version": 3}, "unsupported reconciliation schema"),
        ({"reconciliation_schema_version": None}, "passing artifact has no reconciliation schema"),
        ({"artifact_schema_version": 2}, "unsupported artifact schema"),
        ({"passed": True, "verification_status": "FAIL"}, "passed and verification_status disagree"),
        ({"event_identity": {"verified": False}}, "passing audit cannot have failed event identity"),
        ({"provider_errors": {"OpenF1": "outage"}}, "passing audit cannot contain provider errors"),
    ],
)
def test_matrix_rejects_inconsistent_or_stale_artifacts(tmp_path: Path, overrides, message):
    source = _artifact(tmp_path / "bad.json", **overrides)
    with pytest.raises(ValueError, match=message):
        build_matrix([source])


def test_matrix_rejects_duplicate_event_identity(tmp_path: Path):
    first = _artifact(tmp_path / "a.json")
    second = _artifact(tmp_path / "b.json", openf1_session_key=9999)
    with pytest.raises(ValueError, match="Duplicate season/round"):
        build_matrix([first, second])


def test_matrix_rejects_duplicate_session_key(tmp_path: Path):
    first = _artifact(tmp_path / "a.json")
    second = _artifact(tmp_path / "b.json", round=2)
    with pytest.raises(ValueError, match="Duplicate OpenF1 session"):
        build_matrix([first, second])


def test_matrix_requires_at_least_one_real_artifact():
    with pytest.raises(ValueError, match="At least one"):
        build_matrix([])


def test_matrix_preserves_pre_reconciliation_failure_without_inventing_schema(tmp_path: Path):
    source = _artifact(
        tmp_path / "failed.json",
        passed=False,
        verification_status="FAIL",
        reconciliation_schema_version=None,
        provider_errors={"OpenF1": "HTTPError: upstream unavailable"},
        event_identity={"verified": False},
        hard_mismatch_count=1,
    )
    matrix = build_matrix([source])

    assert matrix["fail_count"] == 1
    row = matrix["events"][0]
    assert row["passed"] is False
    assert row["reconciliation_performed"] is False
    assert row["provider_error_count"] == 1


def test_matrix_marks_completed_reconciliation_explicitly(tmp_path: Path):
    source = _artifact(tmp_path / "reconciled.json")
    row = build_matrix([source])["events"][0]
    assert row["reconciliation_performed"] is True
