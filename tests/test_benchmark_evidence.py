import pandas as pd
import pytest

from f1_research.benchmark_evidence import benchmark_uncertainty


def _metrics():
    rows = []
    for event_id, qualifying, challenger in [
        ("A", 1.0, 0.8),
        ("B", 1.2, 1.0),
        ("C", 0.9, 0.9),
        ("D", 1.1, 0.7),
    ]:
        rows.extend([
            {
                "event_id": event_id,
                "model": "qualifying_order",
                "position_mae": qualifying,
                "winner_log_loss": qualifying,
                "winner_brier": qualifying / 2,
                "winner_accuracy": 0.5,
                "podium_recall": 0.5,
            },
            {
                "event_id": event_id,
                "model": "challenger",
                "position_mae": challenger,
                "winner_log_loss": challenger,
                "winner_brier": challenger / 2,
                "winner_accuracy": 0.75,
                "podium_recall": 0.75,
            },
        ])
    return pd.DataFrame(rows)


def test_paired_bootstrap_uses_whole_events_and_correct_metric_direction():
    table, report = benchmark_uncertainty(_metrics(), samples=2000, seed=7)
    logloss = report["paired_vs_baseline"]["challenger"]["winner_log_loss"]
    assert logloss["events"] == 4
    assert logloss["mean_difference_model_minus_baseline"] == pytest.approx(-0.2)
    assert logloss["model_better_events"] == 3
    assert logloss["baseline_better_events"] == 0
    assert logloss["ties"] == 1
    assert logloss["bootstrap_fraction_favorable"] > 0.95

    accuracy = report["paired_vs_baseline"]["challenger"]["winner_accuracy"]
    assert accuracy["mean_difference_model_minus_baseline"] == pytest.approx(0.25)
    assert accuracy["model_better_events"] == 4
    assert set(table.metric) == {
        "position_mae", "winner_log_loss", "winner_brier", "winner_accuracy", "podium_recall"
    }


def test_uncertainty_rejects_duplicate_event_model_rows():
    frame = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one metric row"):
        benchmark_uncertainty(frame)
