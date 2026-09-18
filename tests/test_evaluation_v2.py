import numpy as np
import pytest

from f1_research.demo import synthetic_history
from f1_research.evaluation_v2 import _winner_calibration, benchmark_v2


def test_v2_blocks_are_disjoint_and_probability_distributions_are_coherent():
    frame = synthetic_history(events=12, drivers=6)
    metrics, predictions, audit = benchmark_v2(
        frame,
        test_events=2,
        tuning_events=2,
        calibration_events=2,
        min_fit_events=6,
        modern_names=("hist_gradient_boosting",),
        max_specs_per_model=1,
    )
    split = audit["split"]
    blocks = [set(split[name]) for name in ("fit", "tuning", "calibration", "test")]
    assert all(a.isdisjoint(b) for i, a in enumerate(blocks) for b in blocks[i + 1:])
    assert metrics.event_id.nunique() == 2
    assert set(metrics.event_id) == set(split["test"])
    assert metrics.model.nunique() == 6
    assert "rank_ensemble" in set(metrics.model)
    assert np.isfinite(metrics[["position_mae", "winner_log_loss", "winner_brier", "spearman_rank", "kendall_rank", "ndcg"]]).all().all()

    grouped = predictions.groupby(["event_id", "model"])
    for _, group in grouped:
        assert group.win_probability.sum() == pytest.approx(1)
        assert group.podium_probability.sum() == pytest.approx(3, abs=0.08)
        assert group.top10_probability.sum() == pytest.approx(6, abs=0.08)

    assert audit["modern_selection_metric"] == "winner_log_loss"
    assert audit["modern_tuning_temperature_is_final"] is False
    assert audit["pl_selection_metric"] == "winner_log_loss"
    assert audit["pl_tuning_temperature_is_final"] is False
    assert audit["ensemble_enabled"] is True
    assert audit["ensemble_selection_metric"] == "winner_log_loss"
    assert audit["ensemble_tuning_temperature_is_final"] is False
    assert sum(audit["ensemble_weights"].values()) == pytest.approx(1.0)
    assert sum(value > 0 for value in audit["ensemble_weights"].values()) >= 2
    modern_rows = [row for row in audit["modern_tuning"] if row["error"] is None]
    pl_rows = [row for row in audit["pl_tuning"] if row["error"] is None]
    assert modern_rows and pl_rows and audit["ensemble_tuning"]
    assert all(np.isfinite(row["mean_winner_log_loss"]) for row in modern_rows)
    assert all(np.isfinite(row["mean_position_mae"]) for row in modern_rows)
    assert all(row["tuning_temperature"] > 0 for row in modern_rows)
    assert all(np.isfinite(row["mean_winner_log_loss"]) for row in pl_rows)
    assert all(row["tuning_temperature"] > 0 for row in pl_rows)
    assert audit["test_updates_model"] is False


def test_mutating_sealed_test_outcomes_cannot_change_model_or_temperature_selection():
    frame = synthetic_history(events=11, drivers=5)
    kwargs = dict(
        test_events=2,
        tuning_events=2,
        calibration_events=1,
        min_fit_events=6,
        modern_names=("hist_gradient_boosting",),
        max_specs_per_model=1,
    )
    _, _, original = benchmark_v2(frame, **kwargs)
    changed = frame.copy()
    test_ids = set(original["split"]["test"])
    for event_id in test_ids:
        mask = changed.event_id.eq(event_id)
        maximum = changed.loc[mask, "finish_position"].max()
        changed.loc[mask, "finish_position"] = maximum + 1 - changed.loc[mask, "finish_position"]

    _, _, mutated = benchmark_v2(changed, **kwargs)
    assert original["split"] == mutated["split"]
    assert original["selected_modern"] == mutated["selected_modern"]
    assert original["modern_tuning"] == mutated["modern_tuning"]
    assert original["selected_pl_l2"] == mutated["selected_pl_l2"]
    assert original["pl_tuning"] == mutated["pl_tuning"]
    assert original["ensemble_weights"] == mutated["ensemble_weights"]
    assert original["ensemble_tuning"] == mutated["ensemble_tuning"]
    assert original["temperatures"] == mutated["temperatures"]


def test_ensemble_can_be_disabled_without_changing_individual_model_protocol():
    frame = synthetic_history(events=11, drivers=5)
    metrics, _, audit = benchmark_v2(
        frame,
        test_events=2,
        tuning_events=2,
        calibration_events=1,
        min_fit_events=6,
        modern_names=("hist_gradient_boosting",),
        max_specs_per_model=1,
        include_ensemble=False,
    )
    assert "rank_ensemble" not in set(metrics.model)
    assert metrics.model.nunique() == 5
    assert audit["ensemble_enabled"] is False
    assert audit["ensemble_weights"] is None
    assert audit["ensemble_tuning"] == []


def test_winner_calibration_ece_is_zero_for_perfect_extreme_forecasts():
    import pandas as pd

    predictions = pd.DataFrame([
        {"model": "m", "actual_position": 1, "win_probability": 1.0},
        {"model": "m", "actual_position": 2, "win_probability": 0.0},
        {"model": "m", "actual_position": 1, "win_probability": 1.0},
        {"model": "m", "actual_position": 3, "win_probability": 0.0},
    ])
    result = _winner_calibration(predictions, bins=5)
    assert result[0]["winner_ece"] == pytest.approx(0.0)
    assert sum(row["count"] for row in result[0]["reliability"]) == 4
