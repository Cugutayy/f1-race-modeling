from f1_research.provider_reconciliation import ResultRow, normalize_openf1_results, reconcile_results


def _row(provider, *, number, position, laps, status, points=0.0, code=None):
    return ResultRow(
        provider=provider,
        driver_number=number,
        position=position,
        laps=laps,
        status_class=status,
        points=points,
        driver_code=code or str(number),
    )


def _provider_rows(provider, *, target_position, target_status="dnf"):
    return [
        _row(provider, number=4, position=1, laps=57, status="finished", points=25.0, code="NOR"),
        _row(
            provider,
            number=30,
            position=target_position,
            laps=46,
            status=target_status,
            points=0.0,
            code="LAW",
        ),
    ]


def test_nonfinisher_missing_position_is_evidence_gap_not_hard_mismatch():
    report = reconcile_results({
        "Jolpica": _provider_rows("Jolpica", target_position=2),
        "FastF1": _provider_rows("FastF1", target_position=2),
        "OpenF1": _provider_rows("OpenF1", target_position=None),
    })

    assert report["passed"] is True
    assert report["verification_status"] == "PASS_WITH_GAPS"
    assert report["hard_mismatch_count"] == 0
    assert report["insufficient_hard_count"] == 0
    assert report["audit_gap_count"] > 0
    assert any(item["field"] == "position" for item in report["audit_gaps"])


def test_classified_driver_missing_position_is_explicit_gap_not_contradiction():
    report = reconcile_results({
        "Jolpica": _provider_rows("Jolpica", target_position=2, target_status="finished"),
        "OpenF1": _provider_rows("OpenF1", target_position=None, target_status="finished"),
    })

    assert report["passed"] is True
    assert report["verification_status"] == "PASS_WITH_GAPS"
    assert report["hard_mismatch_count"] == 0
    assert any(
        row["provider"] == "OpenF1"
        and row["field"] == "position"
        and "classified" in row["reason"]
        for row in report["insufficient_hard_evidence"]
    )


def test_missing_start_status_is_audited_without_blocking_release_evidence():
    report = reconcile_results({
        "Jolpica": [
            _row("Jolpica", number=4, position=1, laps=57, status="finished", code="NOR"),
            _row("Jolpica", number=30, position=None, laps=0, status="dnf", code="LAW"),
        ],
        "OpenF1": [
            _row("OpenF1", number=4, position=1, laps=57, status="finished", code="NOR"),
            _row("OpenF1", number=30, position=None, laps=0, status="dns", code="LAW"),
        ],
    })

    assert report["passed"] is True
    assert report["hard_mismatch_count"] == 0
    assert report["insufficient_hard_count"] == 0
    assert any(item["field"] == "start_status" for item in report["audit_gaps"])


def test_explicit_start_status_disagreement_remains_hard_failure():
    report = reconcile_results({
        "Jolpica": [
            _row("Jolpica", number=4, position=1, laps=57, status="finished", code="NOR"),
            _row("Jolpica", number=30, position=None, laps=1, status="dnf", code="LAW"),
        ],
        "OpenF1": [
            _row("OpenF1", number=4, position=1, laps=57, status="finished", code="NOR"),
            _row("OpenF1", number=30, position=None, laps=0, status="dns", code="LAW"),
        ],
    })

    assert report["passed"] is False
    assert any(
        row["field"] == "start_status" and row["severity"] == "hard"
        for row in report["mismatches"]
    )

def test_openf1_points_are_preserved_as_secondary_evidence():
    rows = normalize_openf1_results([
        {
            "driver_number": 4,
            "position": 1,
            "number_of_laps": 57,
            "points": 25.0,
            "dnf": False,
            "dns": False,
            "dsq": False,
            "gap_to_leader": 0,
        },
        {
            "driver_number": 1,
            "position": 2,
            "number_of_laps": 57,
            "points": 18.0,
            "dnf": False,
            "dns": False,
            "dsq": False,
            "gap_to_leader": 4.2,
        },
    ])
    by_number = {row.driver_number: row for row in rows}
    assert by_number[4].points == 25.0
    assert by_number[1].points == 18.0
