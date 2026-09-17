import pandas as pd
import pytest

from f1_research.reconciliation import (
    ProviderDriverRecord,
    normalize_fastf1,
    normalize_jolpica,
    normalize_openf1,
    reconcile,
)


def _jolpica_result():
    return {
        "MRData": {
            "RaceTable": {
                "Races": [{
                    "date": "2025-03-16",
                    "Results": [
                        {
                            "position": "1", "grid": "2", "laps": "57", "points": "25",
                            "status": "Finished",
                            "Driver": {"driverId": "alpha", "permanentNumber": "1"},
                            "Constructor": {"constructorId": "team_a"},
                        },
                        {
                            "position": "2", "grid": "1", "laps": "57", "points": "18",
                            "status": "+4.2 Seconds",
                            "Driver": {"driverId": "beta", "permanentNumber": "2"},
                            "Constructor": {"constructorId": "team_b"},
                        },
                        {
                            "position": "3", "grid": "3", "laps": "40", "points": "15",
                            "status": "Engine",
                            "Driver": {"driverId": "gamma", "permanentNumber": "3"},
                            "Constructor": {"constructorId": "team_c"},
                        },
                    ],
                }]
            }
        }
    }


def _jolpica_pits():
    return {
        "MRData": {
            "RaceTable": {
                "Races": [{
                    "PitStops": [
                        {"driverId": "alpha", "lap": "20"},
                        {"driverId": "alpha", "lap": "40"},
                        {"driverId": "beta", "lap": "22"},
                    ]
                }]
            }
        }
    }


def _openf1_results():
    return [
        {"driver_number": 1, "position": 1, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False},
        {"driver_number": 2, "position": 2, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False},
    ]


def _fastf1_results():
    return pd.DataFrame([
        {"DriverNumber": "1", "Abbreviation": "AAA", "TeamName": "A",
         "Position": 1.0, "GridPosition": 2.0, "Status": "Finished", "Points": 25.0},
        {"DriverNumber": "2", "Abbreviation": "BBB", "TeamName": "B",
         "Position": 2.0, "GridPosition": 1.0, "Status": "+4.2 Seconds", "Points": 18.0},
    ])


def test_jolpica_normalization_preserves_status_semantics_and_pit_counts():
    rows = normalize_jolpica(_jolpica_result(), _jolpica_pits())
    by_number = {row.driver_number: row for row in rows}
    assert by_number[1].finish_position == 1
    assert by_number[1].pit_stops == 2
    assert by_number[1].dnf is False
    assert by_number[2].dnf is False
    assert by_number[3].dnf is True
    assert by_number[3].status_raw == "Engine"


def test_jolpica_missing_pit_dataset_stays_unknown_not_zero():
    rows = normalize_jolpica(_jolpica_result(), None)
    assert all(row.pit_stops is None for row in rows)


def test_jolpica_explicit_empty_pit_dataset_means_zero_observed_stops():
    payload = {"MRData": {"RaceTable": {"Races": []}}}
    rows = normalize_jolpica(_jolpica_result(), payload)
    assert all(row.pit_stops == 0 for row in rows)


def test_openf1_normalization_deduplicates_identical_pit_rows():
    drivers = [
        {"driver_number": 1, "name_acronym": "AAA", "team_name": "A"},
        {"driver_number": 2, "name_acronym": "BBB", "team_name": "B"},
    ]
    pits = [
        {"driver_number": 1, "lap_number": 20, "date": "2025-03-16T10:20:00Z"},
        {"driver_number": 1, "lap_number": 20, "date": "2025-03-16T10:20:00Z"},
        {"driver_number": 1, "lap_number": 40, "date": "2025-03-16T10:40:00Z"},
    ]
    rows = normalize_openf1(_openf1_results(), drivers, pits)
    by_number = {row.driver_number: row for row in rows}
    assert by_number[1].pit_stops == 2
    assert by_number[2].pit_stops == 0
    assert by_number[1].team == "A"


def test_openf1_missing_pit_dataset_stays_unknown_not_zero():
    rows = normalize_openf1(_openf1_results(), pit_rows=None)
    assert all(row.pit_stops is None for row in rows)


def test_openf1_explicit_false_strings_do_not_become_true_by_python_truthiness():
    rows = normalize_openf1([
        {"driver_number": 1, "position": 1, "number_of_laps": 57,
         "dnf": "false", "dns": "0", "dsq": "False"},
    ], pit_rows=[])
    assert rows[0].dnf is False
    assert rows[0].dns is False
    assert rows[0].dsq is False


def test_openf1_malformed_explicit_boolean_is_rejected():
    with pytest.raises(ValueError, match="openf1.dnf"):
        normalize_openf1([
            {"driver_number": 1, "position": 1, "number_of_laps": 57,
             "dnf": "maybe", "dns": False, "dsq": False},
        ], pit_rows=[])


def test_fastf1_normalization_uses_lap_and_pit_evidence():
    laps = pd.DataFrame([
        {"DriverNumber": "1", "LapNumber": 1.0, "PitInTime": pd.NaT},
        {"DriverNumber": "1", "LapNumber": 2.0, "PitInTime": pd.Timedelta(seconds=100)},
        {"DriverNumber": "2", "LapNumber": 1.0, "PitInTime": pd.NaT},
        {"DriverNumber": "2", "LapNumber": 2.0, "PitInTime": pd.NaT},
    ])
    rows = normalize_fastf1(_fastf1_results(), laps)
    by_number = {row.driver_number: row for row in rows}
    assert by_number[1].completed_laps == 2
    assert by_number[1].pit_stops == 1
    assert by_number[1].dnf is False
    assert by_number[2].grid_position == 1


def test_fastf1_missing_lap_or_pit_evidence_stays_unknown():
    no_laps = normalize_fastf1(_fastf1_results(), None)
    assert all(row.completed_laps is None for row in no_laps)
    assert all(row.pit_stops is None for row in no_laps)

    no_pit_column = pd.DataFrame([
        {"DriverNumber": "1", "LapNumber": 1.0},
        {"DriverNumber": "2", "LapNumber": 1.0},
    ])
    rows = normalize_fastf1(_fastf1_results(), no_pit_column)
    assert all(row.completed_laps == 1 for row in rows)
    assert all(row.pit_stops is None for row in rows)


def test_fastf1_nullable_status_does_not_raise_ambiguous_truth_error():
    results = _fastf1_results()
    results.loc[0, "Status"] = pd.NA
    rows = normalize_fastf1(results, None)
    first = next(row for row in rows if row.driver_number == 1)
    assert first.status_raw is None
    assert first.dnf is None


def _record(provider, number, position, laps, *, pits=1, dnf=False):
    return ProviderDriverRecord(
        provider=provider,
        driver_number=number,
        finish_position=position,
        completed_laps=laps,
        pit_stops=pits,
        dnf=dnf,
    )


def test_reconciliation_passes_only_when_comparable_values_agree():
    records = []
    for provider in ("jolpica", "openf1", "fastf1"):
        records.extend([
            _record(provider, 1, 1, 57, pits=2),
            _record(provider, 2, 2, 57, pits=1),
        ])
    report = reconcile(records)
    assert report["summary"]["critical_failures"] == 0
    assert report["summary"]["secondary_failures"] == 0
    assert report["summary"]["truth_elected_on_disagreement"] is False
    assert "FastF1 maximum observed LapNumber" in report["field_semantics"]["completed_laps"]
    assert all(item["status"] == "PASS" for item in report["presence"])


def test_finish_position_disagreement_is_critical_and_has_no_consensus():
    records = [
        _record("jolpica", 1, 1, 57),
        _record("openf1", 1, 1, 57),
        _record("fastf1", 1, 2, 57),
        _record("jolpica", 2, 2, 57),
        _record("openf1", 2, 2, 57),
        _record("fastf1", 2, 1, 57),
    ]
    report = reconcile(records)
    failures = [
        row for row in report["comparisons"]
        if row["field"] == "finish_position" and row["status"] == "FAIL"
    ]
    assert len(failures) == 2
    assert all(row["severity"] == "critical" for row in failures)
    assert all(row["consensus"] is None for row in failures)
    assert report["summary"]["critical_failures"] >= 2


def test_missing_provider_driver_is_critical_presence_failure():
    records = [
        _record("jolpica", 1, 1, 57),
        _record("openf1", 1, 1, 57),
        _record("fastf1", 1, 1, 57),
        _record("jolpica", 2, 2, 56),
        _record("openf1", 2, 2, 56),
    ]
    report = reconcile(records)
    driver_two = next(row for row in report["presence"] if row["driver_number"] == 2)
    assert driver_two["status"] == "FAIL"
    assert driver_two["missing_providers"] == ["fastf1"]
    assert report["summary"]["critical_failures"] >= 1


def test_secondary_pit_disagreement_is_visible_but_does_not_invent_truth():
    records = [
        _record("jolpica", 1, 1, 57, pits=1),
        _record("openf1", 1, 1, 57, pits=2),
        _record("fastf1", 1, 1, 57, pits=1),
    ]
    report = reconcile(records)
    pit = next(row for row in report["comparisons"] if row["field"] == "pit_stops")
    assert pit["status"] == "FAIL"
    assert pit["severity"] == "secondary"
    assert pit["consensus"] is None
    assert report["summary"]["critical_failures"] == 0
    assert report["summary"]["secondary_failures"] == 1


def test_insufficient_field_evidence_is_not_converted_to_pass_or_fail():
    report = reconcile([
        ProviderDriverRecord("jolpica", 1, finish_position=1, completed_laps=57, points=25),
        ProviderDriverRecord("openf1", 1, finish_position=1, completed_laps=57),
        ProviderDriverRecord("fastf1", 1, finish_position=1, completed_laps=57),
    ])
    points = next(row for row in report["comparisons"] if row["field"] == "points")
    assert points["status"] == "INSUFFICIENT"
    assert points["consensus"] is None
    assert points["values"] == {"jolpica": 25}


def test_duplicate_provider_record_is_rejected():
    records = [
        _record("jolpica", 1, 1, 57),
        _record("jolpica", 1, 1, 57),
    ]
    with pytest.raises(ValueError, match="Duplicate jolpica"):
        reconcile(records)
