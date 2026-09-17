import numpy as np
import pandas as pd
import pytest

from f1_research.telemetry import clean_laps, summarize_samples


def test_summary_weights_time_and_never_bridges_dropouts():
    samples = pd.DataFrame({"time_s": [0, .2, 1, 4, 4.2],
                            "speed_kmh": [100, 200, 999, 300, 300],
                            "throttle_pct": [0, 100, 100, 50, 50],
                            "brake": [1, 0, 0, 0, 0]})
    result = summarize_samples(samples)
    assert result["gaps_excluded"] == 1
    assert result["interval_coverage"] == pytest.approx(1.2 / 4.2)
    assert result["braking_time_fraction"] == pytest.approx(.2 / 1.2)
    # Speed intervals touching the out-of-range 999 sample are excluded.
    assert result["mean_speed_kmh"] == pytest.approx(200)
    assert result["speed_kmh_coverage"] == pytest.approx(.4 / 4.2)


def test_duplicate_timestamp_is_not_silently_averaged():
    frame = pd.DataFrame({"time_s": [0, 0], "speed_kmh": [1, 2],
                          "throttle_pct": [0, 0], "brake": [0, 1]})
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_samples(frame)


def test_unknown_channels_remain_missing_not_zero():
    result = summarize_samples(pd.DataFrame({"time_s": [0, .2], "speed_kmh": [1, 2],
                                            "throttle_pct": [np.nan, np.nan], "brake": [5, 5]}))
    assert result["mean_throttle_pct"] is None
    assert result["braking_time_fraction"] is None
    assert result["brake_coverage"] == 0


def test_lap_quality_preserves_excluded_rows_and_reasons():
    frame = pd.DataFrame({"Driver": ["AAA"] * 4, "LapNumber": [1, 2, 3, 4],
                          "LapTime": pd.to_timedelta([90, 91, 92, 93], unit="s"),
                          "IsAccurate": [True, True, True, False],
                          "TrackStatus": ["1", "1", "14", "1"],
                          "PitInTime": [pd.NaT] * 4, "PitOutTime": [pd.NaT] * 4,
                          "Deleted": [False, False, False, False]})
    result = clean_laps(frame)
    assert len(result) == 4
    assert result["clean"].tolist() == [False, True, False, False]
    assert result.iloc[2]["exclusion_reason"] == "non_green_track"


def test_openf1_provider_normalizes_brake_and_encodes_time_filter(tmp_path, monkeypatch):
    from f1_research.telemetry import collect_openf1

    urls = []

    class Cache:
        provenance = []

        def __init__(self, *args, **kwargs):
            pass

        def get(self, url):
            urls.append(url)
            if "/sessions?" in url:
                return [{"year": 2024, "date_end": "2024-03-01T17:00:00Z",
                         "session_name": "Qualifying", "location": "Synthetic Source Test"}]
            if "/drivers?" in url:
                return [{"driver_number": 1, "name_acronym": "AAA"}]
            if "/laps?" in url:
                return [{"lap_number": 2, "lap_duration": 90., "is_pit_out_lap": False,
                         "date_start": "2024-03-01T16:00:00Z"}]
            assert "date%3E=" in url and "date%3C=" in url
            return [{"date": f"2024-03-01T16:00:00.{ms:03d}Z", "speed": 100,
                     "throttle": 0, "brake": 100, "rpm": 9000, "n_gear": 3, "drs": 0}
                    for ms in [0, 200, 400]]

    monkeypatch.setattr("f1_research.data.JsonCache", Cache)
    result = collect_openf1(1, [1], tmp_path / "cache", tmp_path / "output")
    assert result["clean_laps"] is None
    assert result["eligible_laps"] == 1
    assert result["telemetry"][0]["braking_time_fraction"] == 1
    assert result["telemetry"][0]["lap_interval_coverage"] == pytest.approx(.4 / 90)
    assert len(urls) == 4
