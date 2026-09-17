from pathlib import Path

import f1_research.provider_audit as provider_audit


def test_failure_report_has_explicit_schema_contract() -> None:
    report = provider_audit._failure_report(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        provider_errors={"OpenF1": "HTTPError: unavailable"},
        raw_snapshots={},
    )

    assert report["schema_version"] == 4
    assert report["artifact_schema_version"] == 1
    assert report["reconciliation_schema_version"] is None
    assert report["verification_status"] == "FAIL"


def test_success_preserves_core_reconciliation_schema(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        provider_audit,
        "collect_jolpica_raw",
        lambda *_args: {"results": [], "pitstops": []},
    )
    monkeypatch.setattr(
        provider_audit,
        "collect_openf1_raw",
        lambda *_args: {
            "session_result": [],
            "drivers": [],
            "pit": [],
            "starting_grid": None,
        },
    )
    monkeypatch.setattr(
        provider_audit,
        "collect_fastf1_partial",
        lambda *_args: {"results_available": True, "results": [], "laps": None},
    )
    monkeypatch.setattr(
        provider_audit,
        "build_event_identity",
        lambda **_kwargs: {"verified": True, "failures": [], "providers": {}},
    )
    monkeypatch.setattr(
        provider_audit,
        "normalize_jolpica_results",
        lambda *_args: ([object()], {}),
    )
    monkeypatch.setattr(
        provider_audit,
        "normalize_openf1_results",
        lambda *_args: [object()],
    )
    monkeypatch.setattr(
        provider_audit,
        "normalize_fastf1_results",
        lambda *_args: [object()],
    )
    monkeypatch.setattr(
        provider_audit,
        "reconcile_results",
        lambda _normalized: {
            "schema_version": 4,
            "kind": "cross_provider_completed_race_reconciliation",
            "passed": True,
            "verification_status": "PASS",
            "hard_mismatch_count": 0,
            "warning_count": 0,
            "insufficient_hard_count": 0,
            "insufficient_secondary_count": 0,
            "mismatches": [],
            "insufficient_hard_evidence": [],
            "insufficient_secondary": [],
            "policy": {},
        },
    )

    report = provider_audit.audit_completed_race(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        cache=tmp_path / "cache",
        output=tmp_path / "audit",
    )

    assert report["schema_version"] == 4
    assert report["artifact_schema_version"] == 1
    assert report["reconciliation_schema_version"] == 4
    assert report["passed"] is True
