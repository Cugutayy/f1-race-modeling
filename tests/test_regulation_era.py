import pandas as pd
import pytest

from f1_research.features import build_features, regulation_era


def _row(event_id, date, year, driver, team, finish, points):
    return {
        "event_id": event_id,
        "date": date,
        "year": year,
        "round": 1,
        "driver": driver,
        "team": team,
        "circuit": "test-track",
        "quali_position": finish,
        "quali_seconds": 90.0 + finish / 10,
        "finish_position": finish,
        "points": points,
        "dnf": 0,
    }


def test_regulation_era_boundaries():
    assert regulation_era(2013) == "pre_2014"
    assert regulation_era(2014) == "2014_2016_hybrid"
    assert regulation_era(2017) == "2017_2021_wide_car"
    assert regulation_era(2022) == "2022_2025_ground_effect"
    assert regulation_era(2026) == "2026_plus"


def test_2026_resets_team_form_but_preserves_driver_history():
    frame = pd.DataFrame([
        _row("2025-24", "2025-12-07", 2025, "driver_a", "team_a", 1, 25),
        _row("2025-24", "2025-12-07", 2025, "driver_b", "team_b", 2, 18),
        _row("2026-01", "2026-03-08", 2026, "driver_a", "team_a", 2, 18),
        _row("2026-01", "2026-03-08", 2026, "driver_b", "team_b", 1, 25),
    ])
    features = build_features(frame)
    first_2026 = features[features.event_id == "2026-01"].set_index("driver")

    assert first_2026.loc["driver_a", "regulation_era"] == "2026_plus"
    assert first_2026.loc["driver_a", "team_form"] == pytest.approx(0.5)
    assert first_2026.loc["driver_a", "team_era_history_count"] == 0
    assert first_2026.loc["driver_a", "history_count"] == 1
    assert first_2026.loc["driver_a", "driver_form"] == pytest.approx(0.0)


def test_team_form_accumulates_again_inside_the_new_era():
    frame = pd.DataFrame([
        _row("2026-01", "2026-03-08", 2026, "driver_a", "team_a", 1, 25),
        _row("2026-01", "2026-03-08", 2026, "driver_b", "team_b", 2, 18),
        _row("2026-02", "2026-03-22", 2026, "driver_a", "team_a", 2, 18),
        _row("2026-02", "2026-03-22", 2026, "driver_b", "team_b", 1, 25),
    ])
    features = build_features(frame)
    second = features[features.event_id == "2026-02"].set_index("driver")
    assert second.loc["driver_a", "team_era_history_count"] == 1
    assert second.loc["driver_a", "team_form"] == pytest.approx(0.0)
