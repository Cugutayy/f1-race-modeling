import hashlib

from f1_research.release_manifest import build_release_manifest


def test_release_manifest_hashes_exact_artifacts(tmp_path):
    files = {}
    for name, raw in {"model": b"m", "truth": b"t", "benchmark": b"b", "replay": b"r"}.items():
        files[name] = tmp_path / name
        files[name].write_bytes(raw)
    manifest = build_release_manifest(
        git_sha="abc123", model_manifest=files["model"], data_truth=files["truth"],
        benchmark=files["benchmark"], replay=files["replay"],
    )
    assert manifest.data_truth_sha256 == hashlib.sha256(b"t").hexdigest()
    assert manifest.replay_sha256 == hashlib.sha256(b"r").hexdigest()
