import copy

import pandas as pd
import pytest

from f1_research.provider_reconciliation import (
    ResultRow,
    normalize_fastf1_results,
    normalize_jolpica_results,
    normalize_openf1_results,
    reconcile_results,
)


def _jolpica_payload():
    def result(number, position, laps, status, code):
        return {
            "number": str(number),
            "position": str(position),
            "positionText": str(position),
            "points": "0",
            "laps": str(laps),
            "status": status,
            "Driver": {
                "driverId": code.lower(),
                "code": code,
                "givenName": code,
                "familyName": "Driver",
            },
            "Constructor": {"constructorId": "test"},
        }

    rows = [
        result(1, 1, 57, "Finished", "AAA"),
        result(4, 2, 57, "Finished", "BBB"),
        result(14, 3, 56, "+1 Lap", "CCC"),
        result(18, 4, 32, "Retired", "DDD"),
    ]
    return {
        "MRData": {
            "total": "4",
            "RaceTable": {
                "Races": [{
                    "season": "2026",
                    "round": "7",
                    "raceName": "Verification Grand Prix",
                    "date": "2026-05-31",
                    "Circuit": {"circuitId": "verification_ring"},
                    "Results": rows,
                }]
            },
        }
    }


def _openf1_rows():
    return [
        {"driver_number": 1, "position": 1, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 0},
        {"driver_number": 4, "position": 2, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 7.2},
        {"driver_number": 14, "position": 3, "number_of_laps": 56,
         "dnf": False, "dns": False, "dsq": False, "gap_to_leader": "+1 LAP"},
        {"driver_number": 18, "position": 4, "number_of_laps": 32,
         "dnf": True, "dns": False, "dsq": False, "gap_to_leader": "+25 LAPS"},
    ]


def _openf1_drivers():
    return [
        {"driver_number": 1, "name_acronym": "AAA", "full_name": "AAA Driver"},
        {"driver_number": 4, "name_acronym": "BBB", "full_name": "BBB Driver"},
        {"driver_number": 14, "name_acronym": "CCC", "full_name": "CCC Driver"},
        {"driver_number": 18, "name_acronym": "DDD", "full_name": "DDD Driver"},
    ]


def _fastf1_results():
    return [
        {"DriverNumber": "1", "Position": 1, "ClassifiedPosition": "1",
         "Status": "Finished", "Abbreviation": "AAA", "FullName": "AAA Driver"},
        {"DriverNumber": "4", "Position": 2, "ClassifiedPosition": "2",
         "Status": "Finished", "Abbreviation": "BBB", "FullName": "BBB Driver"},
        {"DriverNumber": "14", "Position": 3, "ClassifiedPosition": "3",
         "Status": "+1 Lap", "Abbreviation": "CCC", "FullName": "CCC Driver"},
        {"DriverNumber": "18", "Position": 4, "ClassifiedPosition": "4",
         "Status": "Retired", "Abbreviation": "DDD", "FullName": "DDD Driver"},
    ]


def _fastf1_laps():
    return [
        {"DriverNumber": "1", "LapNumber": 57},
        {"DriverNumber": "4", "LapNumber": 57},
        {"DriverNumber": "14", "LapNumber": 56},
        {"DriverNumber": "18", "LapNumber": 32},
    ]


def _three_provider_rows():
    jolpica, _ = normalize_jolpica_results(_jolpica_payload())
    openf1 = normalize_openf1_results(_openf1_rows(), _openf1_drivers())
    fastf1 = normalize_fastf1_results(pd.DataFrame(_fastf1_results()), pd.DataFrame(_fastf1_laps()))
    return {"Jolpica": jolpica, "OpenF1": openf1, "FastF1": fastf1}


def test_three_providers_agree_on_completed_race_classification():
    report = reconcile_results(_three_provider_rows())
    assert report["passed"] is True
    assert report["hard_mismatch_count"] == 0
    assert report["warning_count"] == 0
    assert report["row_counts"] == {"FastF1": 4, "Jolpica": 4, "OpenF1": 4}
    openf1 = {row["driver_number"]: row for row in report["normalized"]["OpenF1"]}
    assert openf1[14]["status_class"] == "classified_lapped"
    assert openf1[18]["status_class"] == "dnf"


def test_one_provider_position_disagreement_is_hard_and_never_majority_repaired():
    providers = _three_provider_rows()
    changed = []
    for row in providers["OpenF1"]:
        position = row.position
        if row.driver_number == 1:
            position = 2
        elif row.driver_number == 4:
            position = 1
        changed.append(ResultRow(**{**row.__dict__, "position": position}))
    providers["OpenF1"] = changed

    report = reconcile_results(providers)
    assert report["passed"] is False
    assert report["hard_mismatch_count"] >= 4
    position_mismatches = [row for row in report["mismatches"] if row["field"] == "position"]
    assert position_mismatches
    assert report["policy"]["majority_vote"] is False
    assert report["policy"]["repair_disagreements"] is False


def test_missing_classified_laps_are_reported_as_evidence_gap():
    providers = _three_provider_rows()
    rows = list(providers["FastF1"])
    target = rows[0]
    rows[0] = ResultRow(**{**target.__dict__, "laps": None})
    providers["FastF1"] = rows

    report = reconcile_results(providers)
    assert report["passed"] is True
    assert report["verification_status"] == "PASS_WITH_GAPS"
    assert report["hard_mismatch_count"] == 0
    assert any(
        row.get("provider") == "FastF1"
        and row["field"] == "laps"
        and "classified" in row["reason"]
        for row in report["insufficient_hard_evidence"]
    )


def test_driver_code_difference_is_warning_only_when_numeric_truth_agrees():
    providers = _three_provider_rows()
    rows = list(providers["FastF1"])
    target = rows[0]
    rows[0] = ResultRow(**{**target.__dict__, "driver_code": "ZZZ"})
    providers["FastF1"] = rows

    report = reconcile_results(providers)
    assert report["passed"] is True
    assert report["hard_mismatch_count"] == 0
    assert report["warning_count"] >= 2  # FastF1 is compared with both other providers.
    assert all(
        row["severity"] == "warning"
        for row in report["mismatches"]
        if row["field"] == "driver_code"
    )


def test_jolpica_multi_race_payload_is_rejected_instead_of_implicitly_merged():
    payload = _jolpica_payload()
    payload["MRData"]["RaceTable"]["Races"].append(
        copy.deepcopy(payload["MRData"]["RaceTable"]["Races"][0])
    )
    with pytest.raises(ValueError, match="exactly one race"):
        normalize_jolpica_results(payload)


def test_openf1_dns_and_lapped_statuses_are_explicit():
    rows = _openf1_rows()
    rows.append({
        "driver_number": 22,
        "position": 5,
        "number_of_laps": 0,
        "dnf": False,
        "dns": True,
        "dsq": False,
        "gap_to_leader": None,
    })
    normalized = normalize_openf1_results(rows)
    by_number = {row.driver_number: row for row in normalized}
    assert by_number[14].status_class == "classified_lapped"
    assert by_number[18].status_class == "dnf"
    assert by_number[22].status_class == "dns"
    assert by_number[22].laps == 0


def test_fastf1_lap_count_is_derived_without_inventing_missing_laps():
    rows = _fastf1_results()
    normalized = normalize_fastf1_results(rows, _fastf1_laps())
    by_number = {row.driver_number: row for row in normalized}
    assert by_number[1].laps == 57
    assert by_number[14].laps == 56
    assert by_number[18].laps == 32

    missing_laps = [row for row in _fastf1_laps() if row["DriverNumber"] != "18"]
    normalized = normalize_fastf1_results(rows, missing_laps)
    by_number = {row.driver_number: row for row in normalized}
    assert by_number[18].laps is None


def test_openf1_starting_grid_is_observed_without_zero_filling_missing_driver():
    grid = [
        {"driver_number": 1, "position": 1},
        {"driver_number": 4, "position": 2},
    ]
    normalized = normalize_openf1_results(_openf1_rows(), _openf1_drivers(), None, grid)
    by_number = {row.driver_number: row for row in normalized}
    assert by_number[1].grid_position == 1
    assert by_number[4].grid_position == 2
    assert by_number[14].grid_position is None


def test_openf1_duplicate_starting_grid_driver_is_rejected():
    grid = [
        {"driver_number": 1, "position": 1},
        {"driver_number": 1, "position": 2},
    ]
    with pytest.raises(ValueError, match="duplicate driver_number"):
        normalize_openf1_results(_openf1_rows(), _openf1_drivers(), None, grid)


def test_collect_openf1_raw_keeps_missing_starting_grid_as_unknown(monkeypatch):
    import requests

    import f1_research.provider_reconciliation as module

    class Response:
        status_code = 404

    class FakeClient:
        def __init__(self):
            self.provenance = []

        def get(self, endpoint, **filters):
            if endpoint == "sessions":
                return [{"session_key": 9693, "meeting_key": 1254, "session_name": "Race"}]
            if endpoint == "meetings":
                return [{"meeting_key": 1254}]
            if endpoint == "starting_grid":
                raise requests.HTTPError("404 Client Error", response=Response())
            return []

    monkeypatch.setattr(module, "OpenF1Client", FakeClient)
    raw = module.collect_openf1_raw(9693)

    assert raw["starting_grid"] is None
    assert raw["optional_collection_errors"]["starting_grid"]["http_status"] == 404



def test_dns_vs_zero_lap_retired_is_semantic_gap_not_false_hard_mismatch():
    support = ResultRow("Jolpica", 1, 1, 57, "finished")
    providers = {
        "Jolpica": [support, ResultRow("Jolpica", 6, 2, 0, "dnf")],
        "FastF1": [
            ResultRow("FastF1", 1, 1, 57, "finished"),
            ResultRow("FastF1", 6, 2, 0, "dnf"),
        ],
        "OpenF1": [
            ResultRow("OpenF1", 1, 1, 57, "finished"),
            ResultRow("OpenF1", 6, None, 0, "dns"),
        ],
    }
    report = reconcile_results(providers)
    assert report["passed"] is True
    assert report["verification_status"] == "PASS_WITH_GAPS"
    assert report["hard_mismatch_count"] == 0
    assert not [
        mismatch
        for mismatch in report["mismatches"]
        if mismatch["driver_number"] == 6 and mismatch["field"] == "status_class"
    ]
    assert any(
        gap["driver_number"] == 6 and gap["field"] == "start_status"
        for gap in report["insufficient_hard_evidence"]
    )
    normalized = {
        provider: {row["driver_number"]: row for row in rows}
        for provider, rows in report["normalized"].items()
    }
    assert normalized["OpenF1"][6]["result_class"] == "non_finisher"
    assert normalized["OpenF1"][6]["start_status"] == "dns"
    assert normalized["Jolpica"][6]["result_class"] == "non_finisher"
    assert normalized["Jolpica"][6]["start_status"] is None


def test_genuine_result_class_disagreement_remains_hard_failure():
    providers = {
        "Jolpica": [
            ResultRow("Jolpica", 1, 1, 57, "finished"),
            ResultRow("Jolpica", 6, 2, 0, "dnf"),
        ],
        "OpenF1": [
            ResultRow("OpenF1", 1, 1, 57, "finished"),
            ResultRow("OpenF1", 6, 2, 0, "finished"),
        ],
    }
    report = reconcile_results(providers)
    assert report["passed"] is False
    assert any(
        row["driver_number"] == 6
        and row["field"] == "result_class"
        and row["severity"] == "hard"
        for row in report["mismatches"]
    )
