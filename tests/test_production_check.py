import hashlib
import json

from f1_research.model_registry import ModelManifest, write_manifest
from f1_research.production_check import run_checks

GIT_SHA = "a" * 40


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _files(tmp_path, *, status="PASS_WITH_GAPS", test_events=12):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({
        "matrix_schema_version": 1,
        "producer_git_sha": GIT_SHA,
        "events": [
            {
                "verification_status": status,
                "passed": status in {"PASS", "PASS_WITH_GAPS"},
                "event_identity_verified": status != "FAIL",
                "reconciliation_performed": status != "FAIL",
                "hard_mismatch_count": 0 if status != "FAIL" else 1,
                "provider_error_count": 0,
                "source_sha256": "b" * 64,
                "producer_git_sha": GIT_SHA,
            }
            for _ in range(12)
        ],
    }))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({
        "schema_version": 2, "data_kind": "historical", "run_id": "sealed-1", "test_events": test_events,
        "predictions": [{"event_id": f"T{i}", "driver": "VER"} for i in range(test_events)], "metrics": [{"position_mae": 1.0}],
        "winner_calibration": [{"model": "rank-v1", "winner_ece": 0.05, "reliability": []}],
        "audit": {"test_updates_model": False, "split": {
            "fit": ["E1"], "tuning": ["E2"], "calibration": ["E3"],
            "test": [f"T{i}" for i in range(test_events)],
        }},
    }))
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    calibration = tmp_path / "calibration.json"
    calibration.write_bytes(b"calibration")
    feature_schema = tmp_path / "feature_schema.json"
    feature_schema.write_bytes(b"features")
    training_data = tmp_path / "training_features.csv"
    training_data.write_bytes(b"data")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ModelManifest(
        1, "rank-v1", _sha(b"model"), _sha(b"features"), _sha(b"data"),
        _sha(b"calibration"), "2026-03-01T00:00:00Z", "sealed-1", GIT_SHA,
    ))
    replay = tmp_path / "replay.jsonl"
    replay_rows = [
        ("position", {"driver_number": 1, "position": 1}),
        ("laps", {"driver_number": 1, "lap_number": 1, "lap_duration": 90.0}),
        ("weather", {"track_temperature": 30.0, "air_temperature": 20.0, "rainfall": 0}),
        ("race_control", {"category": "Flag", "flag": "GREEN", "message": "GREEN LIGHT"}),
    ]
    replay.write_text("".join(
        json.dumps({
            "topic": topic,
            "received_at": f"2026-03-08T05:00:0{index}Z",
            "payload": {
                "date": f"2026-03-08T05:00:0{index}Z",
                "session_key": 1,
                **payload,
            },
        }) + "\n"
        for index, (topic, payload) in enumerate(replay_rows)
    ))
    return truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay


def test_production_gate_can_pass_complete_evidence(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is True


def test_production_gate_rejects_failed_real_audit(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path, status="FAIL")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert result["checks"]["data_truth"]["passed"] is False


def test_production_gate_rejects_tampered_model(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    model.write_bytes(b"tampered")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model, calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert result["checks"]["model_integrity"]["passed"] is False


def test_production_gate_rejects_overlapping_benchmark_blocks(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    payload = json.loads(benchmark.read_text())
    payload["audit"]["split"]["calibration"] = ["T0"]
    benchmark.write_text(json.dumps(payload))
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert "overlap" in result["checks"]["benchmark"]["detail"]


def test_production_gate_rejects_predictions_outside_sealed_test(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    payload = json.loads(benchmark.read_text())
    payload["predictions"][0]["event_id"] = "LEAKED"
    benchmark.write_text(json.dumps(payload))
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert "sealed test" in result["checks"]["benchmark"]["detail"]


def test_production_gate_rejects_benchmark_without_calibration_evidence(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    payload = json.loads(benchmark.read_text())
    payload.pop("winner_calibration")
    benchmark.write_text(json.dumps(payload))
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert "calibration" in result["checks"]["benchmark"]["detail"]


def test_production_gate_rejects_single_topic_replay(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    replay.write_text(json.dumps({
        "topic": "position", "received_at": "2026-03-08T05:00:00Z",
        "payload": {"date": "2026-03-08T05:00:00Z", "session_key": 1,
                    "driver_number": 1, "position": 1},
    }) + "\n")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model,
                        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay)
    assert result["production_ready"] is False
    assert "missing topics" in result["checks"]["replay"]["detail"]


def test_production_gate_rejects_stale_data_truth_revision(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    result = run_checks(
        data_truth=truth,
        benchmark=benchmark,
        manifest=manifest,
        model=model,
        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay,
        expected_git_sha="b" * 40,
    )
    assert result["production_ready"] is False
    assert "does not match release revision" in result["checks"]["data_truth"]["detail"]


def test_production_gate_accepts_matching_source_revision(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    result = run_checks(
        data_truth=truth,
        benchmark=benchmark,
        manifest=manifest,
        model=model,
        calibration=calibration,
        feature_schema=feature_schema, training_data=training_data,
        replay_capture=replay,
        expected_git_sha=GIT_SHA,
    )
    assert result["production_ready"] is True



def test_production_gate_rejects_tampered_training_snapshot(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    training_data.write_bytes(b"tampered-data")
    result = run_checks(
        data_truth=truth,
        benchmark=benchmark,
        manifest=manifest,
        model=model,
        calibration=calibration,
        feature_schema=feature_schema,
        training_data=training_data,
        replay_capture=replay,
    )
    assert result["production_ready"] is False
    assert "training data" in result["checks"]["model_integrity"]["detail"].lower()


def test_production_gate_rejects_tampered_feature_schema(tmp_path):
    truth, benchmark, manifest, model, calibration, feature_schema, training_data, replay = _files(tmp_path)
    feature_schema.write_bytes(b"tampered-features")
    result = run_checks(
        data_truth=truth,
        benchmark=benchmark,
        manifest=manifest,
        model=model,
        calibration=calibration,
        feature_schema=feature_schema,
        training_data=training_data,
        replay_capture=replay,
    )
    assert result["production_ready"] is False
    assert "feature schema" in result["checks"]["model_integrity"]["detail"].lower()
