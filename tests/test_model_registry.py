import hashlib

import pytest

from f1_research.model_registry import ModelManifest, load_manifest, write_manifest


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _manifest(model_sha):
    return ModelManifest(
        schema_version=1, model_id="rank-v1", model_sha256=model_sha,
        feature_schema_sha256=_sha("features"), training_data_sha256=_sha("data"),
        calibration_sha256=_sha("calibration"), trained_until="2026-03-01T00:00:00Z",
        benchmark_run_id="sealed-001", git_sha="abc123",
    )


def test_manifest_verifies_model_bytes(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, _manifest(hashlib.sha256(b"model").hexdigest()))
    assert load_manifest(manifest, model_path=model).model_id == "rank-v1"


def test_manifest_rejects_mutated_model(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, _manifest(hashlib.sha256(b"model").hexdigest()))
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        load_manifest(manifest, model_path=model)
