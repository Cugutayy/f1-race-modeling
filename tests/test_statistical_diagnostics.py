import numpy as np
import pandas as pd

from f1_research.statistical_diagnostics import (
    moving_block_bootstrap_mean,
    paired_randomization_test,
    winner_calibration_diagnostics,
)


def test_paired_randomization_detects_consistently_lower_metric():
    diff = np.array([-0.3, -0.2, -0.4, -0.1, -0.25, -0.35])
    result = paired_randomization_test(diff, direction="lower", samples=2000, seed=7)
    assert result["observed_favors_model"] is True
    assert result["observed_mean_difference"] < 0
    assert result["p_one_sided_favorable"] <= 0.05
    assert result["method"] == "exact_sign_flip"


def test_moving_block_bootstrap_is_deterministic_and_preserves_mean_scale():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = moving_block_bootstrap_mean(values, samples=500, block_length=2, seed=11)
    b = moving_block_bootstrap_mean(values, samples=500, block_length=2, seed=11)
    assert np.array_equal(a, b)
    assert len(a) == 500
    assert np.isfinite(a).all()
    assert abs(a.mean() - values.mean()) < 0.5


def _predictions():
    rows = []
    for event, probs, winner_index in [
        ("e1", [0.7, 0.2, 0.1], 0),
        ("e2", [0.2, 0.6, 0.2], 1),
        ("e3", [0.55, 0.25, 0.20], 2),
        ("e4", [0.4, 0.35, 0.25], 0),
    ]:
        for idx, probability in enumerate(probs):
            rows.append({
                "event_id": event,
                "model": "m",
                "driver": f"d{idx}",
                "actual_position": 1 if idx == winner_index else idx + 2,
                "win_probability": probability,
            })
    return pd.DataFrame(rows)


def test_winner_calibration_checks_coherent_event_probabilities():
    report = winner_calibration_diagnostics(_predictions(), bins=4)
    model = report["models"]["m"]
    assert model["events"] == 4
    assert model["driver_rows"] == 12
    assert model["event_probability_sum_max_abs_error"] < 1e-12
    assert 0 <= model["ece_equal_width"] <= 1
    assert 0 <= model["ece_adaptive"] <= 1
    assert np.isfinite(model["brier_decomposition"]["brier_binary_driver_row"])
    assert model["calibration_logistic"]["converged"] is True
    assert 0 <= model["top_choice"]["accuracy"] <= 1


def test_winner_calibration_rejects_noncoherent_probability_sum():
    frame = _predictions()
    frame.loc[0, "win_probability"] += 0.1
    try:
        winner_calibration_diagnostics(frame, bins=4)
    except ValueError as exc:
        assert "do not sum to one" in str(exc)
    else:
        raise AssertionError("Expected non-coherent winner probabilities to fail")
