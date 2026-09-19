import hashlib
import json

import pytest

from f1_research import live_api


def test_readyz_requires_installed_production_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.setattr(live_api, "_read_state", lambda: {"session_key": 1, "current_lap": 2})
    monkeypatch.setattr(live_api, "_trusted_live_audit", lambda state: {"verified": True})
    monkeypatch.setattr(live_api, "_load_artifact", lambda: None)
    monkeypatch.setattr(live_api, "_priors_path", lambda: tmp_path / "missing-priors.json")
    monkeypatch.setattr(
        live_api, "_read_model_evidence",
        lambda: (_ for _ in ()).throw(live_api.HTTPException(status_code=404, detail="missing")),
    )
    with pytest.raises(live_api.HTTPException) as exc:
        live_api.readyz(None)
    assert exc.value.status_code == 503
    assert set(exc.value.detail["missing"]) == {
        "strict_model", "strategy_priors", "model_evidence",
    }


def test_modelz_exposes_sealed_evidence(monkeypatch):
    monkeypatch.delenv("F1_API_TOKEN", raising=False)
    monkeypatch.setattr(
        live_api,
        "_load_artifact",
        lambda: {"schema_version": 7, "selected_regressor": "extra_trees"},
    )
    monkeypatch.setattr(
        live_api,
        "_strict_release_metadata",
        lambda: {
            "verified": True,
            "production_eligible": True,
            "git_sha": "a" * 40,
            "source_evidence_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(live_api, "_read_model_evidence", lambda: {
        "evidence_kind": "retrospective_sealed_historical_benchmark",
        "sealed_test_events": 12,
        "benchmark_run_id": "sealed-abc",
        "source_provenance_sha256": "c" * 64,
    })
    response = live_api.modelz(None)
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["live_pace_model"]["artifact_schema_version"] == 7
    assert payload["live_pace_model"]["selected_regressor"] == "extra_trees"
    assert payload["live_pace_model"]["release"]["verified"] is True
    assert payload["race_outcome_model_evidence"]["sealed_test_events"] == 12
    assert payload["race_outcome_model_evidence"]["benchmark_run_id"] == "sealed-abc"
    assert payload["race_outcome_model_evidence"]["source_provenance_sha256"] == "c" * 64


def _strict_release_fixture(monkeypatch, tmp_path):
    model = tmp_path / "next_lap_strict.joblib"
    priors = tmp_path / "strategy_priors.json"
    release = tmp_path / "strict_release_manifest.json"
    model.write_bytes(b"strict-model-bytes")
    priors.write_bytes(b'{"schema_version":1,"priors":{"pit_loss":20.0}}')
    baseline_guard = {
        "baseline": "recent_median_5_baseline",
        "minimum_relative_improvement": 0.01,
        "tuning_baseline_mae_s": 0.8,
        "tuning_selected_mae_s": 0.8,
        "challenger_selected": False,
    }
    artifact = {
        "schema_version": 4,
        "task": "next_lap_strict_mixture",
        "features": ["lap_number", "driver_number"],
        "selected_regressor": "recent_median_5_baseline",
        "pace_prediction_mode": "recent_median_5_baseline",
        "pace_regressor": None,
        "baseline_guard": baseline_guard,
        "calibration_sessions": [1001, 1002, 1003],
        "sealed_test_session": 1004,
        "conformal_radii_s": {
            "0.50": 0.5,
            "0.80": 0.8,
            "0.90": 1.0,
            "0.95": 1.2,
        },
        "retrospective_stint_features_used": False,
    }
    payload = {
        "schema_version": 1,
        "evidence_kind": "strict_live_pace_release",
        "git_sha": "a" * 40,
        "artifact_schema_version": 4,
        "pace_prediction_mode": "recent_median_5_baseline",
        "baseline_guard": baseline_guard,
        "validation_scope": "retrospective_historical_posthoc_diagnostic",
        "prospective_validation": False,
        "feature_policy": "strict_asof_only",
        "feature_schema_sha256": live_api.sha256_json(artifact["features"]),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "strategy_priors_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
        "source_evidence_sha256": "b" * 64,
        "calibration_sessions": artifact["calibration_sessions"],
        "sealed_test_session": artifact["sealed_test_session"],
        "conformal_radii_s": artifact["conformal_radii_s"],
        "retrospective_stint_features_used": False,
    }
    release.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(live_api, "_model_path", lambda: model)
    monkeypatch.setattr(live_api, "_priors_path", lambda: priors)
    monkeypatch.setattr(live_api, "_strict_release_manifest_path", lambda: release)
    monkeypatch.setattr(live_api, "load_strict_artifact", lambda path: artifact)
    monkeypatch.delenv("F1_ALLOW_UNVERIFIED_STRICT_MODEL", raising=False)
    live_api._artifact_cache.update({"key": None, "value": None, "release": None})
    return model, priors, release, artifact


def test_strict_runtime_rejects_missing_release_manifest(monkeypatch, tmp_path):
    model, priors, release, _ = _strict_release_fixture(monkeypatch, tmp_path)
    release.unlink()
    with pytest.raises(live_api.HTTPException) as exc:
        live_api._load_artifact()
    assert exc.value.status_code == 503
    assert "release manifest" in str(exc.value.detail).lower()
    assert model.exists() and priors.exists()


def test_strict_runtime_verifies_hashes_and_invalidates_cache(monkeypatch, tmp_path):
    _, priors, _, artifact = _strict_release_fixture(monkeypatch, tmp_path)
    loaded = live_api._load_artifact()
    assert loaded is artifact
    release = live_api._strict_release_metadata()
    assert release["verified"] is True
    assert release["production_eligible"] is True
    assert release["calibration_sessions"] == [1001, 1002, 1003]
    assert release["artifact_schema_version"] == 4
    assert release["pace_prediction_mode"] == "recent_median_5_baseline"
    assert release["baseline_guard"]["challenger_selected"] is False

    priors.write_bytes(b'{"schema_version":1,"priors":{"pit_loss":21.0},"changed":true}')
    with pytest.raises(live_api.HTTPException) as exc:
        live_api._load_artifact()
    assert exc.value.status_code == 503
    assert "strategy-prior sha-256" in str(exc.value.detail).lower()


def test_strict_runtime_rejects_legacy_artifact_schema(monkeypatch, tmp_path):
    _, _, release_path, artifact = _strict_release_fixture(monkeypatch, tmp_path)
    payload = json.loads(release_path.read_text())
    payload["artifact_schema_version"] = 3
    release_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact["schema_version"] = 3
    live_api._artifact_cache.update({"key": None, "value": None, "release": None})

    with pytest.raises(live_api.HTTPException) as exc:
        live_api._load_artifact()
    assert exc.value.status_code == 503
    assert "schema v4" in str(exc.value.detail).lower()


def test_strict_runtime_override_is_explicitly_non_production(monkeypatch, tmp_path):
    model = tmp_path / "next_lap_strict.joblib"
    model.write_bytes(b"model")
    artifact = {"schema_version": 4}
    monkeypatch.setattr(live_api, "_model_path", lambda: model)
    monkeypatch.setattr(live_api, "load_strict_artifact", lambda path: artifact)
    monkeypatch.setenv("F1_ALLOW_UNVERIFIED_STRICT_MODEL", "1")
    live_api._artifact_cache.update({"key": None, "value": None, "release": None})

    assert live_api._load_artifact() is artifact
    release = live_api._strict_release_metadata()
    assert release["verified"] is False
    assert release["production_eligible"] is False
    assert release["override"] == "F1_ALLOW_UNVERIFIED_STRICT_MODEL"


def test_strict_live_prediction_is_written_to_immutable_ledger(monkeypatch, tmp_path):
    model = tmp_path / "pace.joblib"
    model.write_bytes(b"strict-model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"captured_rows": 10}')
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(live_api, "_model_path", lambda: model)
    monkeypatch.setattr(live_api, "_manifest_path", lambda: manifest)
    monkeypatch.setattr(live_api, "_prediction_ledger_path", lambda: ledger)
    state = {
        "session_key": 9693, "current_lap": 12,
        "updated_at": "2025-03-16T04:30:00Z",
        "latest_provider_event_at": "2025-03-16T04:29:59Z",
    }
    report = {
        "analysis_kind": "live_race_monte_carlo",
        "predictions": [{"driver_number": 1, "win_probability": 0.5}],
        "pace_predictions": [{"driver_number": 1, "predicted_green_lap_s": 90.0}],
        "strategy_prior_source": {"source": "test"},
    }
    prediction_id = live_api._record_live_prediction(state, report, "strict_model")
    assert prediction_id
    assert ledger.exists()
    import json
    row = json.loads(ledger.read_text().strip())
    assert row["prediction_id"] == prediction_id
    assert row["event_id"] == "9693"
    assert len(row["model_sha256"]) == 64
    assert len(row["evidence_sha256"]) == 64


def test_fallback_prediction_is_not_recorded_as_strict_model_evidence(monkeypatch, tmp_path):
    ledger = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(live_api, "_prediction_ledger_path", lambda: ledger)
    prediction_id = live_api._record_live_prediction(
        {"session_key": 1, "updated_at": "2026-01-01T00:00:00Z"},
        {"predictions": []},
        "fallback_recent_laps",
    )
    assert prediction_id is None
    assert not ledger.exists()
