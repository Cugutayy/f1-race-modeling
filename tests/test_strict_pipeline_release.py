import hashlib
import json

import pandas as pd

from f1_research import strict_pipeline
from f1_research.lap_pipeline import ENDPOINTS


def _manifest(session_key: int, *, cached: bool = False, retrieved_at: str | None = None):
    sources = {}
    for index, endpoint in enumerate(ENDPOINTS):
        sources[endpoint] = {
            "path": f"/tmp/{session_key}/{endpoint}.json",
            "bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
            "cached": cached,
        }
        if retrieved_at is not None:
            sources[endpoint]["retrieved_at"] = retrieved_at
    return {
        "session_key": session_key,
        "meeting_key": 9000 + session_key,
        "country_name": "Test",
        "location": "Circuit",
        "date_start": "2026-01-01T00:00:00+00:00",
        "date_end": "2026-01-01T02:00:00+00:00",
        "rows": 120,
        "valid_targets": 100,
        "dataset_sha256": f"{session_key:064x}",
        "sources": sources,
    }


def test_source_content_hash_is_independent_of_cache_and_local_paths():
    fresh = _manifest(1001, cached=False, retrieved_at="2026-09-19T00:00:00+00:00")
    cached = _manifest(1001, cached=True)
    for endpoint in ENDPOINTS:
        cached["sources"][endpoint]["path"] = f"/different/cache/{endpoint}.json"

    fresh_rows, fresh_digest = strict_pipeline._source_evidence([fresh])
    cached_rows, cached_digest = strict_pipeline._source_evidence([cached])

    assert fresh_digest == cached_digest
    assert fresh_rows[0]["source_content_sha256"] == cached_rows[0]["source_content_sha256"]
    assert fresh_rows[0]["sources"]["laps"]["retrieved_at"] is not None
    assert cached_rows[0]["sources"]["laps"]["retrieved_at"] is None


def test_strict_release_manifest_binds_model_priors_report_source_and_git(monkeypatch, tmp_path):
    datasets = [pd.DataFrame({"session_key": [1001 + index]}) for index in range(6)]
    manifests = [_manifest(1001 + index) for index in range(6)]
    monkeypatch.setattr(
        strict_pipeline,
        "collect_recent",
        lambda *args, **kwargs: (datasets, manifests),
    )

    artifact = {
        "schema_version": 3,
        "task": "next_lap_strict_mixture",
        "features": strict_pipeline.STRICT_FEATURES,
        "selected_regressor": "extra_trees",
        "calibration_sessions": [1003, 1004, 1005],
        "sealed_test_session": 1006,
        "conformal_nominal_coverage": 0.90,
        "conformal_radii_s": {
            "0.50": 0.5,
            "0.80": 0.8,
            "0.90": 1.0,
            "0.95": 1.2,
        },
        "retrospective_stint_features_used": False,
    }
    metrics = pd.DataFrame([{
        "test_session": 1006,
        "selected_regressor": "extra_trees",
        "green_rows": 50,
        "calibration_events": 3,
        "green_mae_s": 0.4,
        "green_rmse_s": 0.5,
        "recent_median_mae_s": 0.6,
        "last_lap_mae_s": 0.7,
        "interval_coverage": 0.9,
        "interval_mean_width_s": 2.0,
        "coverage_50": 0.5,
        "coverage_80": 0.8,
        "coverage_90": 0.9,
        "coverage_95": 0.96,
        "interval_width_50_s": 1.0,
        "interval_width_80_s": 1.6,
        "interval_width_90_s": 2.0,
        "interval_width_95_s": 2.4,
        "regime_accuracy": 0.9,
        "regime_log_loss": 0.2,
    }])
    monkeypatch.setattr(
        strict_pipeline,
        "fit_strict_mixture",
        lambda *args, **kwargs: (artifact, metrics, {"test_updates_model": False}),
    )
    monkeypatch.setattr(
        strict_pipeline,
        "calibrate_strategy_priors",
        lambda *args, **kwargs: ({"pit_loss": 20.0}, {"source": "test"}),
    )

    def fake_save(priors, audit, path):
        payload = {"schema_version": 1, "priors": priors, "audit": audit}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(strict_pipeline, "save_strategy_priors", fake_save)
    git_sha = "a" * 40
    monkeypatch.setattr(strict_pipeline, "current_git_sha", lambda required=True: git_sha)

    result = strict_pipeline.run(2026, 6, tmp_path)
    release = json.loads((tmp_path / "strict_release_manifest.json").read_text())

    assert release["git_sha"] == git_sha
    assert release["evidence_kind"] == "strict_live_pace_release"
    assert release["calibration_sessions"] == [1003, 1004, 1005]
    assert release["sealed_test_session"] == 1006
    assert release["sealed_test_session"] not in release["calibration_sessions"]
    assert release["retrospective_stint_features_used"] is False
    assert release["model_sha256"] == hashlib.sha256(
        (tmp_path / "next_lap_strict.joblib").read_bytes()
    ).hexdigest()
    assert release["strategy_priors_sha256"] == hashlib.sha256(
        (tmp_path / "strategy_priors.json").read_bytes()
    ).hexdigest()
    assert release["strict_report_sha256"] == hashlib.sha256(
        (tmp_path / "strict_report.json").read_bytes()
    ).hexdigest()
    _, expected_source_digest = strict_pipeline._source_evidence(manifests)
    assert release["source_evidence_sha256"] == expected_source_digest
    assert len(release["source_sessions"]) == 6
    assert result["release_manifest"] == str(tmp_path / "strict_release_manifest.json")
