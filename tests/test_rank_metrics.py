import numpy as np
import pandas as pd
import pytest

from f1_research.model import score_event


def test_perfect_order_has_perfect_rank_metrics():
    event = pd.DataFrame({"finish_position": [1, 2, 3, 4]})
    metrics, ranks = score_event(event, np.array([0.0, 1.0, 2.0, 3.0]), 1.0)
    assert ranks.tolist() == [1, 2, 3, 4]
    assert metrics["position_mae"] == 0.0
    assert metrics["spearman_rank"] == 1.0
    assert metrics["kendall_rank"] == 1.0
    assert metrics["ndcg"] == pytest.approx(1.0)


def test_reversed_order_degrades_full_rank_metrics():
    event = pd.DataFrame({"finish_position": [1, 2, 3, 4]})
    metrics, _ = score_event(event, np.array([3.0, 2.0, 1.0, 0.0]), 1.0)
    assert metrics["spearman_rank"] < 0
    assert metrics["kendall_rank"] < 0
    assert metrics["ndcg"] < 1.0
