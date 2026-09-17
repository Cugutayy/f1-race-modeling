from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from f1_research.live_replay import OnlineRidge, metrics, normalize_laps, replay, select_sessions


def laps():
    start = pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime()
    rows = []
    for driver, offset in [(1, 0), (2, 35)]:
        for lap in range(1, 12):
            rows.append({"session_key": 1, "driver_number": driver, "lap_number": lap,
                         "date_start": (start + timedelta(seconds=(lap - 1) * 90 + offset)).isoformat(),
                         "lap_duration": 89.0 + lap / 100})
    return rows


def test_every_feature_and_training_label_is_available_at_forecast():
    predictions = replay(normalize_laps(laps()), OnlineRidge(10))
    assert len(predictions) > 10
    assert (pd.to_datetime(predictions.feature_available_at) <= pd.to_datetime(predictions.forecast_at)).all()
    labels = pd.to_datetime(predictions.max_training_label_at)
    assert (labels.dropna() <= pd.to_datetime(predictions.loc[labels.notna(), "forecast_at"])).all()
    assert (pd.to_datetime(predictions.target_available_at) > pd.to_datetime(predictions.forecast_at)).all()
    assert predictions.training_labels.iloc[-1] > predictions.training_labels.iloc[0]


def test_future_target_mutation_cannot_change_existing_forecasts():
    original = laps()
    modified = [dict(row) for row in original]
    cutoff = pd.Timestamp("2024-01-01T00:12:00Z")
    for row in modified:
        if pd.Timestamp(row["date_start"]) >= cutoff:
            row["lap_duration"] = 700.0
    before = replay(normalize_laps(original), OnlineRidge(10))
    after = replay(normalize_laps(modified), OnlineRidge(10))
    a = before[pd.to_datetime(before.forecast_at) <= cutoff]
    b = after[pd.to_datetime(after.forecast_at) <= cutoff]
    np.testing.assert_allclose(a.ridge, b.ridge)
    assert a.features.tolist() == b.features.tolist()


def test_latency_delays_observation_and_training():
    fast = replay(normalize_laps(laps(), 0), OnlineRidge(10))
    slow = replay(normalize_laps(laps(), 100), OnlineRidge(10))
    assert pd.Timestamp(slow.forecast_at.iloc[0]) > pd.Timestamp(fast.forecast_at.iloc[0])
    assert len(slow) < len(fast)


def test_duplicate_and_missing_target_behavior():
    rows = laps()
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_laps(rows + [rows[0]])
    rows[-1]["lap_duration"] = None
    predictions = replay(normalize_laps(rows), OnlineRidge(10))
    assert predictions.actual.isna().sum() == 1
    assert metrics(predictions)[0]["n"] == len(predictions) - 1


def test_online_ridge_matches_batch_normal_equations():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    model = OnlineRidge(7)
    for a, b in zip(x, y):
        model.update(a, b, pd.Timestamp("2024-01-01T00:00:00Z"))
    expected = x[-1] @ np.linalg.solve(x.T @ x + 7 * np.eye(5), x.T @ y)
    assert model.predict(x[-1]) == pytest.approx(expected)


def test_model_trained_on_future_is_rejected():
    model = OnlineRidge(1)
    model.update(np.ones(5), 1, pd.Timestamp("2025-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="crosses forecast cutoff"):
        replay(normalize_laps(laps()), model)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_nonfinite_or_negative_latency_and_regularization_rejected(invalid):
    with pytest.raises(ValueError, match="latency_s"):
        normalize_laps(laps(), invalid)
    with pytest.raises(ValueError, match="alpha"):
        OnlineRidge(invalid)


def test_invalid_labels_do_not_enter_history_or_training():
    rows = laps()
    rows[-1]["lap_duration"] = np.inf
    frame = normalize_laps(rows)
    assert not frame.valid.iloc[-1]
    assert pd.isna(frame.available.iloc[-1])
    prediction = replay(frame, OnlineRidge(10))
    assert prediction.actual.isna().sum() == 1
    assert np.isfinite(prediction.ridge).all()


def test_input_order_and_dataframe_index_do_not_change_replay():
    frame = normalize_laps(laps())
    original = replay(frame, OnlineRidge(10))
    shuffled = frame.sample(frac=1, random_state=42)
    shuffled.index = [0] * len(shuffled)
    result = replay(shuffled, OnlineRidge(10))
    pd.testing.assert_frame_equal(original, result)


def test_frozen_model_can_observe_features_without_learning_test_labels():
    model = OnlineRidge(10)
    result = replay(normalize_laps(laps()), model, update=False)
    assert model.n == 0
    assert result.training_labels.eq(0).all()
    np.testing.assert_allclose(result.ridge, result.recent_median)


def test_invalid_identity_rejected():
    rows = laps()
    rows[0]["driver_number"] = "not-a-driver"
    with pytest.raises(ValueError, match="identity: driver_number"):
        normalize_laps(rows)


def test_empty_metrics_and_nonfinite_prediction_rejected():
    with pytest.raises(ValueError, match="No scored"):
        metrics(pd.DataFrame())
    predictions = replay(normalize_laps(laps()), OnlineRidge(10))
    predictions.loc[0, "ridge"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics(predictions)


def test_latest_completed_filters_future_cancelled_and_unknown_end():
    rows = [{"session_key": i, "date_start": f"2026-01-0{i}T12:00:00Z",
             "date_end": f"2026-01-0{i}T14:00:00Z", "is_cancelled": i == 3}
            for i in range(1, 8)]
    rows[3]["date_end"] = None
    selected = select_sessions(rows[::-1], 2, True, pd.Timestamp("2026-01-06T14:00:00Z"))
    assert [row["session_key"] for row in selected] == [2, 5]
    # Exactly at scheduled end is still excluded, as are sessions underway.
    assert select_sessions(rows, 2, False)[-1]["session_key"] == 2
    with pytest.raises(ValueError, match="Insufficient"):
        select_sessions(rows, 4, True, pd.Timestamp("2026-01-06T14:00:00Z"))


def test_iso_timestamps_with_mixed_precision_preserve_valid_laps():
    rows = laps()
    rows[0]["date_start"] = "2024-01-01T00:00:00.123+00:00"
    frame = normalize_laps(rows)
    assert frame.valid.all()
    assert frame.start.notna().all()
