from datetime import UTC, datetime, timedelta

from f1_research.live_state import RaceStateStore
from f1_research.pace_history import (
    pace_duration_is_plausible,
    rain_state,
    same_rain_regime,
    track_state_is_pace_eligible,
)


def _lap(lap: int, duration: float, at: datetime):
    return {
        "session_key": 123,
        "driver_number": 1,
        "lap_number": lap,
        "date_start": at.isoformat(),
        "lap_duration": duration,
        "is_pit_out_lap": False,
    }


def test_shared_pace_policy_rejects_only_extreme_slow_outlier():
    assert pace_duration_is_plausible(90.5, [90.0, 90.2, 89.9])
    assert pace_duration_is_plausible(82.0, [90.0, 90.2, 89.9])
    assert not pace_duration_is_plausible(166.0, [89.0, 86.5])
    assert rain_state("true") is True
    assert rain_state(0) is False
    assert same_rain_regime(False, False)
    assert not same_rain_regime(False, True)
    assert track_state_is_pace_eligible("GREEN")
    assert not track_state_is_pace_eligible("RESTART")
    assert not track_state_is_pace_eligible("SC")


def test_live_state_keeps_raw_laps_but_filters_model_pace_buffer_and_late_pit():
    store = RaceStateStore(123)
    base = datetime(2026, 9, 19, 12, tzinfo=UTC)
    for lap, duration in ((1, 90.0), (2, 90.2), (3, 89.9), (4, 170.0), (5, 90.1)):
        assert store.ingest("laps", _lap(lap, duration, base + timedelta(seconds=100 * lap)))

    driver = store.state.driver(1)
    assert driver.recent_laps_s == [90.0, 90.2, 89.9, 170.0, 90.1]
    assert driver.pace_laps_s == [90.0, 90.2, 89.9, 90.1]
    assert driver.pace_lap_numbers == [1, 2, 3, 5]

    # Provider topic ordering can deliver pit evidence after the lap row. The raw lap
    # remains observable, while model pace history is corrected.
    assert store.ingest(
        "pit",
        {
            "session_key": 123,
            "driver_number": 1,
            "lap_number": 5,
            "date": (base + timedelta(seconds=505)).isoformat(),
        },
    )
    assert driver.recent_laps_s[-1] == 90.1
    assert driver.pace_lap_numbers == [1, 2, 3]
    assert driver.pace_laps_s == [90.0, 90.2, 89.9]


def test_live_pace_buffer_resets_when_known_rain_regime_changes():
    store = RaceStateStore(123)
    base = datetime(2026, 9, 19, 12, tzinfo=UTC)
    assert store.ingest("weather", {
        "date": base.isoformat(),
        "rainfall": False,
    })
    for lap, duration in ((1, 90.0), (2, 90.1), (3, 89.9)):
        assert store.ingest("laps", _lap(lap, duration, base + timedelta(seconds=100 * lap)))
    driver = store.state.driver(1)
    assert len(driver.pace_laps_s) == 3

    assert store.ingest("weather", {
        "date": (base + timedelta(seconds=350)).isoformat(),
        "rainfall": True,
    })
    assert store.ingest("laps", _lap(4, 104.0, base + timedelta(seconds=400)))
    assert driver.pace_rainfall is True
    assert driver.pace_laps_s == [104.0]
    assert driver.pace_lap_numbers == [4]
