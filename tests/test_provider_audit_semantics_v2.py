from pathlib import Path

from f1_research import provider_audit


def test_failure_report_has_explicit_verification_status_and_hard_evidence_shape():
    report = provider_audit._failure_report(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        provider_errors={"FastF1": "timing unavailable"},
        raw_snapshots={"OpenF1": {"ok": True}},
    )
    assert report["passed"] is False
    assert report["verification_status"] == "FAIL"
    assert report["insufficient_hard_count"] == 0
    assert report["insufficient_hard_evidence"] == []


def test_audit_writes_insufficient_hard_evidence_csv(tmp_path, monkeypatch):
    jolpica = {
        "results": {"MRData": {"RaceTable": {"Races": [{
            "season": "2025", "round": "1", "raceName": "Audit GP", "date": "2025-03-16",
            "time": "04:00:00Z",
            "Circuit": {"circuitId": "audit", "circuitName": "Audit Ring",
                        "Location": {"locality": "Audit City", "country": "Australia"}},
            "Results": [
                {"number": "1", "position": "1", "positionText": "1", "grid": "1",
                 "laps": "57", "points": "25", "status": "Finished",
                 "Driver": {"driverId": "a", "code": "AAA"}, "Constructor": {"constructorId": "a"}},
                {"number": "30", "position": "2", "positionText": "2", "grid": "2",
                 "laps": "46", "points": "0", "status": "Retired",
                 "Driver": {"driverId": "b", "code": "BBB"}, "Constructor": {"constructorId": "b"}},
            ]
        }]}}},
        "pitstops": {"MRData": {"RaceTable": {"Races": [{"PitStops": []}]}}},
        "provenance": [],
    }
    openf1 = {
        "session": {"session_key": 9693, "meeting_key": 1254, "session_name": "Race",
                    "session_type": "Race", "year": 2025,
                    "date_start": "2025-03-16T04:00:00+00:00",
                    "country_name": "Australia", "location": "Audit City",
                    "is_cancelled": False},
        "meeting": {"meeting_key": 1254, "meeting_name": "Audit GP", "year": 2025,
                    "country_name": "Australia", "location": "Audit City",
                    "is_cancelled": False},
        "session_result": [
            {"driver_number": 1, "position": 1, "number_of_laps": 57, "points": 25,
             "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 0},
            {"driver_number": 30, "position": None, "number_of_laps": 46, "points": 0,
             "dnf": True, "dns": False, "dsq": False, "gap_to_leader": None},
        ],
        "drivers": [{"driver_number": 1, "name_acronym": "AAA"},
                    {"driver_number": 30, "name_acronym": "BBB"}],
        "laps": [],
        "pit": [],
    }
    fastf1 = {
        "provider": "FastF1", "fastf1_version": "test",
        "event": {"EventName": "Audit GP", "RoundNumber": 1,
                  "EventDate": "2025-03-16 00:00:00", "Country": "Australia",
                  "Location": "Audit City"},
        "results": [
            {"DriverNumber": "1", "Position": 1, "Laps": 57, "Status": "Finished",
             "Abbreviation": "AAA", "Points": 25},
            {"DriverNumber": "30", "Position": 2, "Laps": 46, "Status": "Retired",
             "Abbreviation": "BBB", "Points": 0},
        ],
        "laps": None, "results_available": True, "laps_available": False,
        "collection_errors": {"laps": "unavailable"},
    }

    monkeypatch.setattr(provider_audit, "collect_jolpica_raw", lambda *_a, **_k: jolpica)
    monkeypatch.setattr(provider_audit, "collect_openf1_raw", lambda *_a, **_k: openf1)
    monkeypatch.setattr(provider_audit, "collect_fastf1_partial", lambda *_a, **_k: fastf1)

    output = Path(tmp_path) / "audit"
    report = provider_audit.audit_completed_race(
        year=2025, round_number=1, openf1_session_key=9693,
        cache=Path(tmp_path) / "cache", output=output,
    )
    assert report["passed"] is True
    assert report["verification_status"] == "PASS_WITH_GAPS"
    assert report["insufficient_hard_count"] == 0
    assert report["audit_gap_count"] > 0
    assert report["event_identity"]["verified"] is True
    assert report["event_metadata"]["OpenF1"]["event_name"] == "Audit GP"
    assert (output / "insufficient_hard_evidence.csv").exists()
    assert (output / "audit_gaps.csv").exists()
