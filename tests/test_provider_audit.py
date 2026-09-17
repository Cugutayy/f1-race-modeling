import json

from f1_research import provider_audit


def _jolpica():
    return {
        "results": {
            "MRData": {
                "total": "2",
                "RaceTable": {
                    "Races": [{
                        "season": "2025",
                        "round": "1",
                        "raceName": "Audit GP",
                        "date": "2025-03-16",
                        "Circuit": {"circuitId": "audit_ring"},
                        "Results": [
                            {
                                "number": "1", "position": "1", "positionText": "1",
                                "grid": "1", "laps": "57", "points": "25", "status": "Finished",
                                "Driver": {"driverId": "alpha", "code": "AAA"},
                                "Constructor": {"constructorId": "a"},
                            },
                            {
                                "number": "2", "position": "2", "positionText": "2",
                                "grid": "2", "laps": "57", "points": "18", "status": "Finished",
                                "Driver": {"driverId": "beta", "code": "BBB"},
                                "Constructor": {"constructorId": "b"},
                            },
                        ],
                    }]
                },
            }
        },
        "pitstops": {"MRData": {"RaceTable": {"Races": [{"PitStops": []}]}}},
        "provenance": [],
    }


def _openf1():
    return {
        "session": {"session_key": 9693, "session_name": "Race", "year": 2025},
        "session_result": [
            {"driver_number": 1, "position": 1, "number_of_laps": 57,
             "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 0},
            {"driver_number": 2, "position": 2, "number_of_laps": 57,
             "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 3.2},
        ],
        "drivers": [
            {"driver_number": 1, "name_acronym": "AAA"},
            {"driver_number": 2, "name_acronym": "BBB"},
        ],
        "laps": [],
        "pit": [],
    }


def test_failure_report_never_marks_provider_failure_as_pass():
    report = provider_audit._failure_report(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        provider_errors={"FastF1": "DataNotLoadedError: timing feed unavailable"},
        raw_snapshots={"OpenF1": {"ok": True}},
    )
    assert report["passed"] is False
    assert report["hard_mismatch_count"] == 1
    assert report["policy"]["provider_failure_is_hard"] is True
    assert report["policy"]["majority_vote"] is False


def test_partial_provider_failure_still_emits_raw_and_reconciliation_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_audit, "collect_jolpica_raw", lambda *_args, **_kwargs: _jolpica())
    monkeypatch.setattr(provider_audit, "collect_openf1_raw", lambda *_args, **_kwargs: _openf1())

    def fail_fastf1(*_args, **_kwargs):
        raise RuntimeError("timing feed unavailable")

    monkeypatch.setattr(provider_audit, "collect_fastf1_partial", fail_fastf1)

    output = tmp_path / "audit"
    report = provider_audit.audit_completed_race(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        cache=tmp_path / "cache",
        output=output,
    )

    assert report["passed"] is False
    assert "FastF1" in report["provider_errors"]
    assert report["hard_mismatch_count"] >= 1
    assert (output / "raw" / "jolpica.json").exists()
    assert (output / "raw" / "openf1.json").exists()
    assert not (output / "raw" / "fastf1.json").exists()
    assert (output / "reconciliation.json").exists()
    assert (output / "provider_errors.json").exists()

    on_disk = json.loads((output / "reconciliation.json").read_text(encoding="utf-8"))
    assert on_disk["passed"] is False
    assert on_disk["provider_errors"]["FastF1"].startswith("RuntimeError")
