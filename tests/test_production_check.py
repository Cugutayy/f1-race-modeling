import hashlib
import json

from f1_research.model_registry import ModelManifest, write_manifest
from f1_research.production_check import run_checks


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _files(tmp_path, *, status="PASS_WITH_GAPS", test_events=12):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"matrix_schema_version": 1, "events": [{"verification_status": status} for _ in range(12)]}))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({
        "schema_version": 2, "data_kind": "historical", "run_id": "sealed-1", "test_events": test_events,
        "predictions": [{"event_id": f"T{i}", "driver": "VER"} for i in range(test_events)], "metrics": [{"position_mae": 1.0}],
        "audit": {"test_updates_model": False, "split": {
            "fit": ["E1"], "tuning": ["E2"], "calibration": ["E3"],
            "test": [f"T{i}" for i in range(test_events)],
        }},
    }))
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    calibration = tmp_path / "calibration.json"
    calibration.write_bytes(b"calibration")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ModelManifest(
        1, "rank-v1", _sha(b"model"), _sha(b"features"), _sha(b"data"),
        _sha(b"calibration"), "2026-03-01T00:00:00Z", "sealed-1", "abc123",
    ))
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({
        "topic": "position",
        "received_at": "2026-03-08T05:00:00Z",
        "payload": {
            "date": "2026-03-08T05:00:00Z", "session_key": 1,
            "driver_number": 1, "position": 1,
        },
    }) + "\n")
    return truth, benchmark, manifest, model, calibration, replay


def test_production_gate_can_pass_complete_evidence(tmp_path):
    truth, benchmark, manifest, model, calibration, replay = _files(tmp_path)
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration, replay_capture=replay)
    assert result["production_ready"] is True


def test_production_gate_rejects_failed_real_audit(tmp_path):
    truth, benchmark, manifest, model, calibration, replay = _files(tmp_path, status="FAIL")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration, replay_capture=replay)
    assert result["production_ready"] is False
    assert result["checks"]["data_truth"]["passed"] is False


def test_production_gate_rejects_tampered_model(tmp_path):
    truth, benchmark, manifest, model, calibration, replay = _files(tmp_path)
    model.write_bytes(b"tampered")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration, replay_capture=replay)
    assert result["production_ready"] is False
    assert result["checks"]["model_integrity"]["passed"] is False


def test_production_gate_rejects_overlapping_benchmark_blocks(tmp_path):
    truth, benchmark, manifest, model, calibration, replay = _files(tmp_path)
    payload = json.loads(benchmark.read_text())
    payload["audit"]["split"]["calibration"] = ["T0"]
    benchmark.write_text(json.dumps(payload))
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration, replay_capture=replay)
    assert result["production_ready"] is False
    assert "overlap" in result["checks"]["benchmark"]["detail"]


def test_production_gate_rejects_predictions_outside_sealed_test(tmp_path):
    truth, benchmark, manifest, model, calibration, replay = _files(tmp_path)
    payload = json.loads(benchmark.read_text())
    payload["predictions"][0]["event_id"] = "LEAKED"
    benchmark.write_text(json.dumps(payload))
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration, replay_capture=replay)
    assert result["production_ready"] is False
    assert "sealed test" in result["checks"]["benchmark"]["detail"]
