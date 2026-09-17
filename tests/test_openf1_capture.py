import hashlib
import json
from datetime import UTC, datetime

from f1_research.live_state import RaceStateStore
from f1_research.openf1_live import CaptureWriter, bootstrap, replay_jsonl


class FakeOpenF1Client:
    def __init__(self):
        self.rows = {
            "sessions": [
                {
                    "session_key": 123,
                    "meeting_key": 456,
                    "session_name": "Race",
                    "location": "Test Circuit",
                    "date_start": "2026-09-17T10:00:00Z",
                }
            ],
            "drivers": [
                {"session_key": 123, "driver_number": 1, "name_acronym": "AAA"},
                {"session_key": 123, "driver_number": 2, "name_acronym": "BBB"},
            ],
            "position": [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "position": 1,
                    "date": "2026-09-17T10:03:00Z",
                },
                {
                    "session_key": 123,
                    "driver_number": 2,
                    "position": 2,
                    "date": "2026-09-17T10:03:00Z",
                },
            ],
            "intervals": [
                {
                    "session_key": 123,
                    "driver_number": 2,
                    "gap_to_leader": 2.4,
                    "interval": 2.4,
                    "date": "2026-09-17T10:03:00Z",
                },
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "gap_to_leader": 0.0,
                    "interval": 0.0,
                    "date": "2026-09-17T10:03:00Z",
                },
            ],
            # Deliberately newest-first: capture must canonicalize before writing.
            "laps": [
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "lap_number": 2,
                    "lap_duration": 89.5,
                    "date_start": "2026-09-17T10:02:00Z",
                },
                {
                    "session_key": 123,
                    "driver_number": 1,
                    "lap_number": 1,
                    "lap_duration": 90.0,
                    "date_start": "2026-09-17T10:00:30Z",
                },
            ],
            "stints": [],
            "pit": [],
            "weather": [],
            "race_control": [],
        }

    def get(self, endpoint, **_filters):
        return list(self.rows[endpoint])


def _semantic_state(snapshot):
    output = dict(snapshot)
    output.pop("snapshot_at", None)
    output.pop("data_age_s", None)
    return output


def test_rest_bootstrap_capture_replays_to_same_semantic_state(tmp_path):
    output = tmp_path / "capture"
    writer = CaptureWriter(output)
    store = RaceStateStore()
    snapshot = bootstrap(FakeOpenF1Client(), store, writer, 123)

    raw_rows = [json.loads(line) for line in writer.raw_path.read_text(encoding="utf-8").splitlines()]
    captured_laps = [row["payload"]["lap_number"] for row in raw_rows if row["topic"] == "laps"]
    assert captured_laps == [1, 2]

    replayed = replay_jsonl(writer.raw_path)
    assert _semantic_state(replayed) == _semantic_state(snapshot)
    driver = replayed["drivers"][0]
    assert driver["recent_lap_numbers"] == [1, 2]
    assert driver["recent_laps_s"] == [90.0, 89.5]


def test_capture_writer_incremental_hash_survives_restart(tmp_path):
    output = tmp_path / "capture"
    writer = CaptureWriter(output)
    writer.append(
        "position",
        {"session_key": 1, "driver_number": 1, "position": 1},
        datetime(2026, 9, 17, 10, 0, tzinfo=UTC),
    )
    writer.publish({"session_key": 1})

    first_manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["captured_rows"] == 1
    assert first_manifest["events_sha256"] == hashlib.sha256(writer.raw_path.read_bytes()).hexdigest()

    resumed = CaptureWriter(output)
    resumed.append(
        "position",
        {"session_key": 1, "driver_number": 2, "position": 2},
        datetime(2026, 9, 17, 10, 0, 1, tzinfo=UTC),
    )
    resumed.publish({"session_key": 1})

    manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    raw = resumed.raw_path.read_bytes()
    assert manifest["captured_rows"] == 2
    assert manifest["capture_bytes"] == len(raw)
    assert manifest["events_sha256"] == hashlib.sha256(raw).hexdigest()
