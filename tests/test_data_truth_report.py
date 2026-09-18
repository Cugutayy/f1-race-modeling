from f1_research.data_truth_report import render_markdown


def test_report_preserves_evidence_gaps_and_hash():
    matrix = {
        "event_count": 1,
        "pass_count": 0,
        "pass_with_gaps_count": 1,
        "fail_count": 0,
        "hard_mismatch_count": 0,
        "audit_gap_count": 27,
        "provider_error_count": 0,
        "events": [{
            "year": 2025,
            "round_number": 1,
            "openf1_session_key": 9693,
            "verification_status": "PASS_WITH_GAPS",
            "hard_mismatch_count": 0,
            "insufficient_hard_count": 0,
            "audit_gap_count": 27,
            "insufficient_secondary_count": 80,
            "reconciliation_performed": True,
            "source_sha256": "a" * 64,
        }],
    }
    report = render_markdown(matrix)
    assert "PASS_WITH_GAPS" in report
    assert "| 0 | 27 | 80 | yes |" in report
    assert "Audit capability gaps: 27" in report
    assert "a" * 64 in report
    assert "never converted to zero or false" in report
    assert "official FIA certification" in report


def test_report_rejects_empty_matrix():
    try:
        render_markdown({"events": []})
    except ValueError as exc:
        assert "at least one event" in str(exc)
    else:
        raise AssertionError("empty matrix must fail closed")
