import pandas as pd
import pytest

from f1_research.provider_reconciliation import (
    ResultRow,
    normalize_fastf1_results,
    normalize_jolpica_results,
    normalize_openf1_results,
    reconcile_results,
)


def _jolpica_result():
    return {
        "MRData": {
            "total": "2",
            "RaceTable": {
                "Races": [{
                    "season": "2025",
                    "round": "1",
                    "raceName": "Truth GP",
                    "date": "2025-03-16",
                    "Circuit": {"circuitId": "truth_ring"},
                    "Results": [
                        {
                            "number": "1", "position": "1", "positionText": "1",
                            "grid": "2", "laps": "57", "points": "25", "status": "Finished",
                            "Driver": {"driverId": "alpha", "code": "AAA"},
                            "Constructor": {"constructorId": "team_a"},
                        },
                        {
                            "number": "2", "position": "2", "positionText": "2",
                            "grid": "1", "laps": "57", "points": "18", "status": "Finished",
                            "Driver": {"driverId": "beta", "code": "BBB"},
                            "Constructor": {"constructorId": "team_b"},
                        },
                    ],
                }]
            },
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
                        {"driverId": "beta", "lap": "25"},
                    ]
                }]
            }
        }
    }


def _openf1_results():
    return [
        {"driver_number": 1, "position": 1, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 0},
        {"driver_number": 2, "position": 2, "number_of_laps": 57,
         "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 4.2},
    ]


def _fastf1_results():
    return pd.DataFrame([
        {"DriverNumber": "1", "Position": 1.0, "ClassifiedPosition": "1",
         "GridPosition": 2.0, "Laps": 57.0, "Points": 25.0,
         "Status": "Finished", "Abbreviation": "AAA"},
        {"DriverNumber": "2", "Position": 2.0, "ClassifiedPosition": "2",
         "GridPosition": 1.0, "Laps": 57.0, "Points": 18.0,
         "Status": "Finished", "Abbreviation": "BBB"},
    ])


def test_missing_pit_dataset_is_unknown_for_jolpica_and_openf1():
    jolpica, _ = normalize_jolpica_results(_jolpica_result(), None)
    openf1 = normalize_openf1_results(_openf1_results(), pit_rows=None)
    assert all(row.pit_stops is None for row in jolpica)
    assert all(row.pit_stops is None for row in openf1)


def test_explicit_pit_evidence_can_prove_zero_or_nonzero_counts():
    jolpica, _ = normalize_jolpica_results(_jolpica_result(), _jolpica_pits())
    jolpica_by_number = {row.driver_number: row for row in jolpica}
    assert jolpica_by_number[1].pit_stops == 2
    assert jolpica_by_number[2].pit_stops == 1

    openf1 = normalize_openf1_results(
        _openf1_results(),
        pit_rows=[
            {"driver_number": 1, "lap_number": 20, "date": "2025-03-16T10:20:00Z"},
            {"driver_number": 1, "lap_number": 20, "date": "2025-03-16T10:20:00Z"},
        ],
    )
    openf1_by_number = {row.driver_number: row for row in openf1}
    assert openf1_by_number[1].pit_stops == 1
    assert openf1_by_number[2].pit_stops == 0


def test_openf1_false_strings_do_not_become_true_by_python_truthiness():
    rows = normalize_openf1_results([
        {"driver_number": 1, "position": 1, "number_of_laps": 57,
         "dnf": "false", "dns": "0", "dsq": "False", "gap_to_leader": 0},
        {"driver_number": 2, "position": 2, "number_of_laps": 57,
         "dnf": "0", "dns": "false", "dsq": 0, "gap_to_leader": 2.0},
    ])
    assert all(row.status_class == "finished" for row in rows)


def test_openf1_malformed_boolean_is_rejected_instead_of_coerced():
    with pytest.raises(ValueError, match="openf1.dnf"):
        normalize_openf1_results([
            {"driver_number": 1, "position": 1, "number_of_laps": 57,
             "dnf": "maybe", "dns": False, "dsq": False, "gap_to_leader": 0},
            {"driver_number": 2, "position": 2, "number_of_laps": 57,
             "dnf": False, "dns": False, "dsq": False, "gap_to_leader": 1.0},
        ])


def test_fastf1_missing_pit_column_does_not_invent_zero_stops():
    laps = pd.DataFrame([
        {"DriverNumber": "1", "LapNumber": 57.0},
        {"DriverNumber": "2", "LapNumber": 57.0},
    ])
    rows = normalize_fastf1_results(_fastf1_results(), laps)
    assert all(row.pit_stops is None for row in rows)


def test_fastf1_pit_column_makes_zero_a_real_observation():
    laps = pd.DataFrame([
        {"DriverNumber": "1", "LapNumber": 1.0, "PitInTime": pd.NaT},
        {"DriverNumber": "1", "LapNumber": 57.0, "PitInTime": pd.Timedelta(seconds=100)},
        {"DriverNumber": "2", "LapNumber": 1.0, "PitInTime": pd.NaT},
        {"DriverNumber": "2", "LapNumber": 57.0, "PitInTime": pd.NaT},
    ])
    rows = normalize_fastf1_results(_fastf1_results(), laps)
    by_number = {row.driver_number: row for row in rows}
    assert by_number[1].pit_stops == 1
    assert by_number[2].pit_stops == 0


def test_fastf1_nullable_status_remains_unknown_without_ambiguous_truth_error():
    results = _fastf1_results()
    results.loc[0, "Status"] = pd.NA
    rows = normalize_fastf1_results(results, None)
    first = next(row for row in rows if row.driver_number == 1)
    assert first.status_raw is None
    assert first.status_class is None


def _row(provider, number, *, position=1, laps=57, grid=None, pits=None, points=None):
    return ResultRow(
        provider=provider,
        driver_number=number,
        position=position,
        laps=laps,
        status_class="finished",
        grid_position=grid,
        pit_stops=pits,
        points=points,
    )


def test_secondary_disagreement_is_warning_and_never_majority_repaired():
    report = reconcile_results({
        "Jolpica": [_row("Jolpica", 1, grid=2, pits=1, points=25)],
        "OpenF1": [_row("OpenF1", 1, pits=2)],
        "FastF1": [_row("FastF1", 1, grid=2, pits=1, points=25)],
    })
    assert report["passed"] is True
    pit_warnings = [row for row in report["mismatches"] if row["field"] == "pit_stops"]
    assert pit_warnings
    assert all(row["severity"] == "warning" for row in pit_warnings)
    assert report["policy"]["majority_vote"] is False
    assert report["policy"]["repair_disagreements"] is False


def test_missing_secondary_evidence_is_reported_as_insufficient_not_zero():
    report = reconcile_results({
        "Jolpica": [_row("Jolpica", 1, grid=2, points=25)],
        "OpenF1": [_row("OpenF1", 1)],
        "FastF1": [_row("FastF1", 1, grid=2, points=25)],
    })
    assert report["passed"] is True
    assert report["insufficient_secondary_count"] > 0
    points_missing = [
        row for row in report["insufficient_secondary"]
        if row["field"] == "points" and "OpenF1" in {row["provider_a"], row["provider_b"]}
    ]
    assert points_missing
    assert report["field_semantics"]["pit_stops"].startswith("Count of provider pit-stop")
