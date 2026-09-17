import numpy as np
import pandas as pd
import pytest

from f1_research.rank_ensemble import (
    blend_scores,
    normalized_event_scores,
    simplex_weights,
    tune_rank_ensemble,
)


def _event(event_id: str, truth=(1, 2, 3, 4)):
    return pd.DataFrame({
        "event_id": [event_id] * 4,
        "finish_position": list(truth),
    })


def test_score_normalization_is_scale_invariant_and_preserves_order():
    raw = np.asarray([10.0, 30.0, 20.0, 40.0])
    first = normalized_event_scores(raw)
    second = normalized_event_scores(raw * 1000.0 + 7.0)
    np.testing.assert_allclose(first, second)
    assert first[0] == pytest.approx(0.0)
    assert first[3] == pytest.approx(1.0)


def test_simplex_grid_requires_real_blends_not_one_hot_duplicates():
    weights = simplex_weights(("modern", "pl", "qualifying"), step=0.5)
    assert weights
    assert all(sum(value > 0 for value in row.values()) >= 2 for row in weights)
    assert all(sum(row.values()) == pytest.approx(1.0) for row in weights)


def test_ensemble_tuning_uses_probability_loss_and_returns_valid_blend():
    tuning = []
    for event_id in ("A", "B", "C"):
        event = _event(event_id)
        parts = {
            "modern": np.asarray([0.0, 0.8, 1.8, 3.0]),
            "pl": np.asarray([0.1, 0.7, 2.0, 2.8]),
            # Slightly wrong front-row ordering: useful prior, not sufficient alone.
            "qualifying": np.asarray([0.3, 0.0, 2.0, 3.0]),
        }
        tuning.append((event, parts))

    best, table = tune_rank_ensemble(tuning, step=0.25)
    assert set(best) == {"modern", "pl", "qualifying"}
    assert sum(best.values()) == pytest.approx(1.0)
    assert sum(value > 0 for value in best.values()) >= 2
    assert np.isfinite(table.iloc[0].mean_winner_log_loss)
    assert table.iloc[0].tuning_temperature > 0

    event, components = tuning[0]
    blended = blend_scores(components, best)
    assert len(blended) == len(event)
    assert np.isfinite(blended).all()


def test_ensemble_tuning_is_invariant_to_component_score_scale():
    event = _event("A")
    base = {
        "modern": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "pl": np.asarray([0.1, 0.8, 2.1, 3.1]),
    }
    scaled = {
        "modern": base["modern"] * 500.0 + 99.0,
        "pl": base["pl"] * 0.01 - 12.0,
    }
    best_a, table_a = tune_rank_ensemble([(event, base)], step=0.5)
    best_b, table_b = tune_rank_ensemble([(event, scaled)], step=0.5)
    assert best_a == best_b
    assert table_a.iloc[0].mean_winner_log_loss == pytest.approx(
        table_b.iloc[0].mean_winner_log_loss
    )
