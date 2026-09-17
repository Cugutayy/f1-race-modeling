from datetime import UTC, datetime, timedelta

import pytest

from f1_research.openf1_doctor import run_doctor


class FakeClient:
    token = "test-token"

    def __init__(self):
        start = datetime(2026, 9, 17, 12, tzinfo=UTC)
        self.start = start
        self.calls = []

    def get(self, endpoint, **filters):
        self.calls.append((endpoint, filters))
        if endpoint == "sessions":
            return [{
                "session_key": 777,
                "session_name": "Race",
                "date_start": self.start.isoformat(),
            }]
        if endpoint == "drivers":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "name_acronym": "AAA",
                "team_name": "Team A",
            }]
        if endpoint == "position":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "position": 1,
                "date": (self.start + timedelta(minutes=2)).isoformat(),
            }]
        if endpoint == "intervals":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "date": (self.start + timedelta(minutes=2)).isoformat(),
                "interval": 0.0,
            }]
        if endpoint == "laps":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "lap_number": 1,
                "date_start": (self.start + timedelta(minutes=1)).isoformat(),
                "lap_duration": 90.0,
            }]
        if endpoint == "stints":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "stint_number": 1,
                "lap_start": 1,
                "compound": "MEDIUM",
            }]
        if endpoint == "pit":
            return []
        if endpoint == "weather":
            return [{
                "session_key": 777,
                "date": self.start.isoformat(),
                "air_temperature": 24.0,
                "track_temperature": 35.0,
            }]
        if endpoint == "race_control":
            return [{
                "session_key": 777,
                "date": self.start.isoformat(),
                "category": "Flag",
            }]
        if endpoint == "car_data":
            return [
                {
                    "session_key": 777,
                    "driver_number": 1,
                    "date": (self.start + timedelta(seconds=i * 0.25)).isoformat(),
                    "speed": 280 + i,
                    "throttle": 80,
                    "brake": 0,
                    "rpm": 11000,
                    "n_gear": 7,
                    "drs": 12,
                }
                for i in range(20)
            ]
        raise AssertionError(endpoint)


def test_provider_doctor_resolves_latest_session_and_checks_telemetry_contract():
    client = FakeClient()
    now = client.start + timedelta(minutes=5)
    report = run_doctor(client, "latest", include_telemetry=True, now=now)
    assert report["ok"] is True
    assert report["authenticated"] is True
    assert report["resolved_session_key"] == 777
    assert report["endpoints"]["laps"]["ok"] is True
    assert report["endpoints"]["drivers"]["required_field_coverage"]["team_name"]["fraction"] == 1.0
    assert report["telemetry_driver"] == 1
    assert report["telemetry"]["rows"] == 20
    assert report["telemetry"]["median_observed_spacing_ms"] == pytest.approx(250.0)
    assert report["telemetry"]["field_coverage"]["speed"] == pytest.approx(1.0)
    driver_call = next(filters for endpoint, filters in client.calls if endpoint == "drivers")
    assert driver_call["session_key"] == 777


class SchemaDriftClient(FakeClient):
    def get(self, endpoint, **filters):
        if endpoint == "laps":
            return [{
                "session_key": 777,
                "driver_number": 1,
                "date_start": self.start.isoformat(),
                # lap_number deliberately missing
            }]
        return super().get(endpoint, **filters)


def test_provider_doctor_marks_required_schema_drift_unhealthy():
    client = SchemaDriftClient()
    report = run_doctor(client, "latest", now=client.start + timedelta(minutes=5))
    assert report["ok"] is False
    laps = report["endpoints"]["laps"]
    assert laps["ok"] is False
    assert "lap_number" in laps["missing_or_sparse_required_fields"]
