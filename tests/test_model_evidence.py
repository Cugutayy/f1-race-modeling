import json

import pytest

from f1_research.model_evidence import build_model_evidence, generate_model_evidence


def _report():
    return {
        "schema_version": 2,
        "run_id": "abc123",
        "summary": [
            {
                "model": "rank_ensemble",
                "position_mae": 3.2,
                "winner_log_loss": 1.08,
                "winner_brier": 0.45,
                "winner_accuracy": 0.75,
                "podium_recall": 0.58,
            },
            {
                "model": "qualifying_order",
                "position_mae": 3.1,
                "winner_log_loss": 1.30,
                "winner_brier": 0.57,
                "winner_accuracy": 0.58,
                "podium_recall": 0.58,
            },
        ],
        "provenance": {
            "provider": "Jolpica",
            "years": [2022, 2023, 2024, 2025, 2026],
            "qualifying_time_basis": "Q1",
            "publication_timestamps_available": False,
            "source_csv_sha256": "c" * 64,
            "provenance_sidecar_sha256": "d" * 64,
            "requests": [{
                "url": "https://api.jolpi.ca/ergast/f1/2026/results/",
                "retrieved_at": "2026-09-19T00:00:00+00:00",
                "sha256": "e" * 64,
            }],
        },
    }


def _selection():
    return {
        "protocol": "fit -> tuning -> calibration -> sealed test",
        "split": {"fit": ["A"], "tuning": ["B"], "calibration": ["C"], "test": ["D", "E"]},
        "selected_modern": {"name": "catboost", "params": {"depth": 4}},
        "ensemble_weights": {"modern": 0.75, "qualifying": 0.25, "pl": 0.0},
    }


def _uncertainty():
    return {
        "baseline": "qualifying_order",
        "bootstrap_samples": 10000,
        "unit": "whole race event",
        "paired_vs_baseline": {
            "rank_ensemble": {
                "winner_log_loss": {
                    "mean_difference_model_minus_baseline": -0.22,
                    "interval_95": [-0.59, 0.11],
                    "events": 2,
                    "model_better_events": 1,
                    "baseline_better_events": 1,
                    "ties": 0,
                    "bootstrap_fraction_favorable": 0.9,
                    "direction": "lower",
                },
                "position_mae": {
                    "mean_difference_model_minus_baseline": 0.05,
                    "interval_95": [-0.10, 0.21],
                    "events": 2,
                    "model_better_events": 1,
                    "baseline_better_events": 1,
                    "ties": 0,
                    "bootstrap_fraction_favorable": 0.3,
                    "direction": "lower",
                },
            }
        },
        "interpretation": "paired whole-event bootstrap",
    }


def test_build_model_evidence_keeps_sealed_metrics_selection_and_uncertainty():
    payload = build_model_evidence(_report(), _selection(), _uncertainty())
    assert payload["sealed_test_events"] == 2
    assert payload["provider"] == "Jolpica"
    assert payload["years"] == [2022, 2023, 2024, 2025, 2026]
    assert payload["source_request_count"] == 1
    assert payload["source_csv_sha256"] == "c" * 64
    assert payload["provenance_sidecar_sha256"] == "d" * 64
    assert payload["publication_timestamps_available"] is False
    assert payload["qualifying_time_basis"] == "Q1"
    assert len(payload["source_provenance_sha256"]) == 64
    assert payload["selected_modern"]["name"] == "catboost"
    assert payload["ensemble_weights"] == {"modern": 0.75, "qualifying": 0.25, "pl": 0.0}
    models = {row["model"]: row for row in payload["models"]}
    assert models["rank_ensemble"]["winner_log_loss"] == pytest.approx(1.08)
    paired = payload["uncertainty"]["paired_vs_baseline"]["rank_ensemble"]["winner_log_loss"]
    assert paired["interval_95"] == pytest.approx([-0.59, 0.11])


def test_generator_does_not_invent_missing_uncertainty(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    (benchmark / "selection.json").write_text(json.dumps(_selection()), encoding="utf-8")
    output = tmp_path / "model_evidence.json"
    payload = generate_model_evidence(benchmark, output)
    assert payload["uncertainty"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["benchmark_run_id"] == "abc123"


def test_evidence_requires_explicit_sealed_test_split():
    selection = _selection()
    selection["split"] = {"fit": ["A"]}
    with pytest.raises(ValueError, match="sealed test"):
        build_model_evidence(_report(), selection, None)
