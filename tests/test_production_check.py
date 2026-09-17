import hashlib
import json

from f1_research.model_registry import ModelManifest, write_manifest
from f1_research.production_check import run_checks


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _files(tmp_path, *, status="PASS_WITH_GAPS", test_events=12):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"matrix_schema_version": 1, "events": [
        {"verification_status": status}
    ]}))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({
        "data_kind": "historical", "test_events": test_events,
        "predictions": [{"driver": "VER"}], "metrics": [{"position_mae": 1.0}],
    }))
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ModelManifest(
        1, "rank-v1", _sha(b"model"), _sha(b"features"), _sha(b"data"),
        _sha(b"calibration"), "2026-03-01T00:00:00Z", "sealed-1", "abc123",
    ))
    return truth, benchmark, manifest, model


def test_production_gate_can_pass_complete_evidence(tmp_path):
    truth, benchmark, manifest, model = _files(tmp_path)
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model)
    assert result["production_ready"] is True


def test_production_gate_rejects_failed_real_audit(tmp_path):
    truth, benchmark, manifest, model = _files(tmp_path, status="FAIL")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model)
    assert result["production_ready"] is False
    assert result["checks"]["data_truth"]["passed"] is False


def test_production_gate_rejects_tampered_model(tmp_path):
    truth, benchmark, manifest, model = _files(tmp_path)
    model.write_bytes(b"tampered")
    result = run_checks(data_truth=truth, benchmark=benchmark, manifest=manifest, model=model)
    assert result["production_ready"] is False
    assert result["checks"]["model_integrity"]["passed"] is False
