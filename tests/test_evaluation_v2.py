import numpy as np
import pytest

from f1_research.demo import synthetic_history
from f1_research.evaluation_v2 import benchmark_v2


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
    assert metrics.model.nunique() == 5
    assert np.isfinite(metrics[["position_mae", "winner_log_loss", "winner_brier"]]).all().all()

    grouped = predictions.groupby(["event_id", "model"])
    for _, group in grouped:
        assert group.win_probability.sum() == pytest.approx(1)
        assert group.podium_probability.sum() == pytest.approx(3, abs=0.08)
        assert group.top10_probability.sum() == pytest.approx(6, abs=0.08)
    assert audit["test_updates_model"] is False
