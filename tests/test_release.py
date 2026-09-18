import json

import pandas as pd
from sklearn.dummy import DummyRegressor

from f1_research import release as rel
from f1_research.model_registry import load_manifest, sha256_file


def test_release_builder_writes_verified_model_bundle(monkeypatch, tmp_path):
    rows = []
    for event in range(8):
        for driver, finish in (("AAA", 1), ("BBB", 2)):
            rows.append({
                "event_id": f"E{event}", "date": f"2025-{event + 1:02d}-01",
                "driver": driver, "team": "T1" if driver == "AAA" else "T2",
                "finish_position": finish, "grid_position": finish,
                "quali_position": finish, "quali_seconds": 80.0 + finish, "quali_gap": float(finish - 1),
                "driver_form": float(finish), "team_form": float(finish),
                "circuit_driver_form": float(finish), "circuit_team_form": float(finish),
                "driver_dnf_rate": 0.0, "team_dnf_rate": 0.0,
                "round": event + 1, "circuit": "X", "year": 2025, "season": 2025,
                "points": 25.0 if finish == 1 else 18.0, "dnf": 0,
            })
    frame = pd.DataFrame(rows)

    fitted_event_ids = []

    def fake_fit(train, spec):
        fitted_event_ids.extend(sorted(train.event_id.unique().tolist()))
        return DummyRegressor(strategy="mean").fit([[0.0]] * len(train), [0.5] * len(train))
    monkeypatch.setattr(rel, "fit_selected", fake_fit)
    monkeypatch.setattr(rel, "benchmark_v2", lambda clean, **kwargs: (
        pd.DataFrame([{"event_id": "E7", "model": "modern::extra_trees",
                       "position_mae": 0.0, "winner_log_loss": 0.1,
                       "winner_brier": 0.01, "winner_accuracy": 1.0, "podium_recall": 1.0,
                       "spearman_rank": 1.0, "kendall_rank": 1.0, "ndcg": 1.0}]),
        pd.DataFrame([{"event_id": "E7", "driver": "AAA", "model": "modern::extra_trees",
                       "actual_position": 1, "predicted_position": 1, "win_probability": 1.0}]),
        {"split": {"fit": ["E0", "E1"], "tuning": ["E2", "E3"],
                   "calibration": ["E4", "E5"], "test": ["E6", "E7"]},
         "selected_modern": {"name": "extra_trees", "params": {}},
         "temperatures": {"modern::extra_trees": 1.0},
         "test_updates_model": False},
    ))
    out = tmp_path / "release"
    result = rel.build_release(frame, out, test_events=2, tuning_events=2,
                               calibration_events=2, min_fit_events=2)
    manifest = load_manifest(out / "model_manifest.json", model_path=out / "model.joblib")
    assert result["model_id"] == manifest.model_id
    assert fitted_event_ids == ["E0", "E1", "E2", "E3"]
    assert result["model_training_blocks"] == ["fit", "tuning"]
    assert result["calibration_used_for_model_fit"] is False
    assert manifest.calibration_sha256 == sha256_file(out / "calibration.json")
    assert manifest.feature_schema_sha256 == sha256_file(out / "feature_schema.json")
    assert manifest.training_data_sha256 == sha256_file(out / "training_features.csv")
    assert result["feature_schema"] == str(out / "feature_schema.json")
    assert result["training_data"] == str(out / "training_features.csv")
    assert json.loads((out / "benchmark" / "report.json").read_text())["test_events"] == 2
