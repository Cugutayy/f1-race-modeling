"""Behavioral guards against the leakage and probability failures in the old script."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f1_research.cli import main
from f1_research.data import JsonCache, races, validate
from f1_research.evaluation import backtest, predict_entries, save_report
from f1_research.features import FEATURES, build_features
from f1_research.model import distribution, probability


@pytest.fixture
def history():
    rows = []
    for rnd in range(1, 10):
        for i in range(6):
            position = (i + rnd) % 6 + 1
            rows.append({
                "event_id": f"2024-{rnd:02d}", "date": f"2024-{rnd:02d}-10",
                "year": 2024, "round": rnd, "driver": f"driver_{i}", "team": f"team_{i // 2}",
                "circuit": f"track_{rnd % 2}", "quali_position": i + 1,
                "quali_seconds": 90 + i / 10, "finish_position": position,
                "points": 7 - position, "dnf": int(position == 6),
            })
    return validate(pd.DataFrame(rows))


def test_features_ignore_current_and_future_results(history):
    original = build_features(history)
    changed = history.copy()
    mask = changed["round"] >= 7
    changed.loc[mask, "finish_position"] = 7 - changed.loc[mask, "finish_position"]
    changed.loc[mask, "points"] = 100
    changed.loc[mask, "dnf"] = 1
    recomputed = build_features(changed)
    assert_frame_equal(original.loc[original["round"] <= 7, FEATURES],
                       recomputed.loc[recomputed["round"] <= 7, FEATURES])


def test_teammate_order_does_not_change_features(history):
    expected = build_features(history)
    actual = build_features(history.sample(frac=1, random_state=100))
    assert_frame_equal(expected, actual)
    first = actual[actual["round"] == 1]
    assert first["team_points_before"].eq(0).all()
    assert first["team_form"].eq(0.5).all()


def test_simultaneous_events_cannot_leak(history):
    history.loc[history["round"] == 2, "date"] = history.loc[history["round"] == 1, "date"].iloc[0]
    features = build_features(history)
    assert features.loc[features["round"] <= 2, "history_count"].eq(0).all()


def test_distribution_is_coherent_and_deterministic():
    scores = np.linspace(0, 1, 20)
    result = distribution(scores, 0.15)
    repeated = distribution(scores, 0.15)
    for key in result:
        np.testing.assert_array_equal(result[key], repeated[key])
        assert np.isfinite(result[key]).all()
    assert result["win_probability"].sum() == pytest.approx(1)
    assert result["podium_probability"].sum() == pytest.approx(3)
    assert result["top10_probability"].sum() == pytest.approx(10)
    assert result["expected_position"].sum() == pytest.approx(210)
    assert ((result["expected_position"] >= 1) & (result["expected_position"] <= 20)).all()
    np.testing.assert_allclose(probability(np.zeros(20), 1), 0.05)
    with pytest.raises(ValueError):
        probability([0, np.nan], 1)
    with pytest.raises(ValueError):
        probability([0, 1], 0)


def test_backtest_parity_and_earlier_fit_boundaries(history):
    metrics, predictions = backtest(history, min_history=7)
    assert metrics["event_id"].nunique() == 2
    assert (pd.to_datetime(metrics["calibration_end"]) < pd.to_datetime(metrics["date"])).all()
    assert (pd.to_datetime(metrics["train_end"]) < pd.to_datetime(metrics["calibration_end"])).all()
    event = history[history["round"] == 8].copy()
    # Passing later results to public inference cannot affect this target.
    inferred, _ = predict_entries(history, event)
    expected = predictions[predictions["event_id"] == "2024-08"]
    cols = ["model", "driver", "predicted_position", "win_probability", "expected_position"]
    assert_frame_equal(expected[cols].reset_index(drop=True), inferred[cols].reset_index(drop=True))
    changed = history.copy()
    changed.loc[changed["round"] >= 8, "finish_position"] = (
        7 - changed.loc[changed["round"] >= 8, "finish_position"])
    _, changed_predictions = backtest(changed, min_history=7)
    np.testing.assert_array_equal(expected["win_probability"],
                                  changed_predictions.loc[changed_predictions.event_id == "2024-08",
                                                          "win_probability"])


def test_unseen_driver_team_and_missing_quali_are_finite(history):
    entries = history[history["round"] == 9].copy()
    entries.loc[entries.index[0], ["driver", "team"]] = ["new_driver", "new_team"]
    entries["quali_seconds"] = np.nan
    predictions, _ = predict_entries(history, entries)
    assert np.isfinite(predictions["win_probability"]).all()
    np.testing.assert_allclose(predictions.groupby("model")["win_probability"].sum(), 1)


def test_validation_rejects_partial_duplicates_and_bad_numeric(history):
    with pytest.raises(ValueError):
        validate(pd.concat([history, history.iloc[:1]]))
    with pytest.raises(ValueError):
        validate(history.drop(index=history.index[0]))
    history.loc[0, "quali_seconds"] = np.inf
    with pytest.raises(ValueError):
        validate(history)


def test_pagination_merges_split_race():
    class Fake:
        def get(self, url):
            offset = int(url.rsplit("=", 1)[-1])
            return {"MRData": {"total": "3", "RaceTable": {"Races": [
                {"round": "1", "Results": [{"driver": i} for i in (
                    [0, 1] if offset == 0 else [2])]}]}}}
    result = races(Fake(), 2024, "results")
    assert len(result[1]["Results"]) == 3


def test_cache_provenance_offline_and_checksum(tmp_path):
    client = JsonCache(tmp_path, delay=0)
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hello": "world"}
    client.session.get = lambda *args, **kwargs: Response()
    assert client.get("https://example.test/data") == {"hello": "world"}
    offline = JsonCache(tmp_path, offline=True)
    assert offline.get("https://example.test/data") == {"hello": "world"}
    assert offline.provenance == client.provenance
    with pytest.raises(FileNotFoundError):
        offline.get("https://example.test/missing")
    cached = next(tmp_path.glob("*.json"))
    body = json.loads(cached.read_text())
    body["data"]["hello"] = "changed"
    cached.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="checksum"):
        offline.get("https://example.test/data")


def test_report_and_cli_contract(history, tmp_path):
    path = tmp_path / "fixture.csv"
    history.to_csv(path, index=False)
    main(["backtest", "--input", str(path), "--output", str(tmp_path / "report"),
          "--min-history", "8", "--data-kind", "synthetic"])
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["schema_version"] == 1 and report["series"] == "f1"
    assert report["data_kind"] == "synthetic" and report["test_events"] == 1
    assert len(report["summary"]) == 5 and len(report["predictions"]) == 30
    metrics, predictions = backtest(history, min_history=8)
    repeated = save_report(history, metrics, predictions, tmp_path / "repeat", data_kind="synthetic")
    assert repeated["run_id"] == report["run_id"]
