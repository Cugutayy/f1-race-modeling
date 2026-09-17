from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from f1_research.demo import synthetic_history
from f1_research.features import FEATURES, build_features
from f1_research.live_state import RaceStateStore
from f1_research.modern_models import (
    CandidateSpec,
    build_estimator,
    normalized_rank_target,
    tune_forward_events,
)
from f1_research.plackett_luce import PlackettLuceRanker
from f1_research.strategy import (
    DriverInput,
    PaceOverride,
    SimulationConfig,
    compare_pit_windows,
    drivers_from_state,
    predict_from_state,
    simulate,
)


def test_live_state_rejects_stale_driver_topic_and_keeps_recent_laps():
    store = RaceStateStore(123)
    now = datetime(2026, 9, 17, 9, tzinfo=UTC)
    assert store.ingest("drivers", {"session_key": 123, "driver_number": 1,
                                     "name_acronym": "AAA", "team_name": "A"}, now)
    assert store.ingest("position", {"driver_number": 1, "position": 2, "date": now.isoformat()}, now)
    stale = now - timedelta(seconds=2)
    assert not store.ingest("position", {"driver_number": 1, "position": 1,
                                          "date": stale.isoformat()}, now)
    for lap, duration in [(5, 90.2), (6, 90.1), (7, 90.0)]:
        at = now + timedelta(seconds=lap)
        assert store.ingest("laps", {"driver_number": 1, "lap_number": lap,
                                     "lap_duration": duration, "date_start": at.isoformat()}, at)
    state = store.snapshot(now + timedelta(seconds=20))
    assert state["drivers"][0]["position"] == 2
    assert state["drivers"][0]["recent_laps_s"] == [90.2, 90.1, 90.0]
    assert state["current_lap"] == 7
    assert state["rejected_stale_messages"] == 1


def _live_state():
    drivers = []
    for i, pace in enumerate((89.8, 90.0, 90.3, 90.6), start=1):
        drivers.append({
            "driver_number": i, "acronym": f"D{i}", "position": i,
            "gap_to_leader_s": 0.0 if i == 1 else (i - 1) * 2.5,
            "recent_laps_s": [pace + 0.08, pace + 0.04, pace, pace + 0.02, pace + 0.05],
            "last_lap_s": pace + 0.05, "compound": "MEDIUM", "tyre_age": 10 + i,
            "pit_stops": 0,
        })
    return {"session_key": 99, "current_lap": 20, "updated_at": "2026-09-17T09:00:00+00:00",
            "drivers": drivers}


def test_live_simulation_is_coherent_deterministic_and_supports_pit_counterfactuals():
    state = _live_state()
    config = SimulationConfig(samples=2000, seed=7, safety_car_hazard_per_lap=0.0,
                              dnf_hazard_per_lap=0.0)
    first = predict_from_state(state, 30, config=config)
    second = predict_from_state(state, 30, config=config)
    assert first == second
    predictions = first["predictions"]
    assert sum(row["win_probability"] for row in predictions) == pytest.approx(1)
    assert sum(row["podium_probability"] for row in predictions) == pytest.approx(3)
    assert sum(row["top10_probability"] for row in predictions) == pytest.approx(4)
    assert all(1 <= row["position_p10"] <= row["position_p90"] <= 4 for row in predictions)
    scenarios = compare_pit_windows(state, 30, 1, offsets=(1, 3), compounds=("SOFT", "HARD"), config=config)
    assert len(scenarios) == 4
    assert {(row["pit_in_laps"], row["compound"]) for row in scenarios} == {
        (1, "SOFT"), (1, "HARD"), (3, "SOFT"), (3, "HARD")}


def test_learned_pace_override_replaces_recent_pace_and_is_audited():
    state = _live_state()
    config = SimulationConfig(samples=1000, seed=3, safety_car_hazard_per_lap=0.0,
                              dnf_hazard_per_lap=0.0)
    overrides = {1: PaceOverride(88.25, 0.22, "strict_next_lap")}
    drivers = drivers_from_state(state, config, overrides)
    first = next(driver for driver in drivers if driver.driver_number == 1)
    assert first.pace_s == pytest.approx(88.25)
    assert first.pace_uncertainty_s == pytest.approx(0.22)
    assert first.pace_source == "strict_next_lap"
    report = predict_from_state(state, 24, config=config, pace_overrides=overrides)
    assert report["audit"]["pace_sources"]["1"] == "strict_next_lap"
    assert report["audit"]["pace_sources"]["2"] == "recent_laps"


def test_current_compound_and_existing_tyre_age_are_not_double_counted():
    drivers = [
        DriverInput(1, "A", 1, 0.0, 90.0, 0.0, 0.10, 10, "SOFT", pace_source="test"),
        DriverInput(2, "B", 2, 0.0, 90.0, 0.0, 0.10, 10, "HARD", pace_source="test"),
    ]
    config = SimulationConfig(
        samples=1000,
        seed=1,
        lap_noise_s=0.0,
        safety_car_hazard_per_lap=0.0,
        dnf_hazard_per_lap=0.0,
    )
    results, _ = simulate(drivers, 1, config=config)
    assert results[0].mean_remaining_time_s == pytest.approx(90.0)
    assert results[1].mean_remaining_time_s == pytest.approx(90.0)


def test_direct_plackett_luce_and_modern_core_models_produce_finite_scores():
    frame = build_features(synthetic_history(events=16, drivers=8))
    train = frame[frame.event_id.isin(frame.event_id.drop_duplicates().iloc[:12])]
    test = frame[frame.event_id.isin(frame.event_id.drop_duplicates().iloc[12:])]

    pl = PlackettLuceRanker(l2=5, max_iter=150).fit(train)
    assert np.isfinite(pl.predict(test)).all()
    assert pl.optimization_["events"] == 12

    spec = CandidateSpec("hist_gradient_boosting", {"max_iter": 30, "max_leaf_nodes": 7})
    model = build_estimator(spec).fit(train[FEATURES], normalized_rank_target(train))
    assert np.isfinite(model.predict(test[FEATURES])).all()

    extra = CandidateSpec("extra_trees", {"min_samples_leaf": 3, "max_features": 0.6,
                                           "n_estimators": 20})
    extra_model = build_estimator(extra).fit(train[FEATURES], normalized_rank_target(train))
    assert np.isfinite(extra_model.predict(test[FEATURES])).all()


def test_forward_tuning_never_uses_future_event_labels():
    frame = build_features(synthetic_history(events=14, drivers=6))
    specs = [CandidateSpec("hist_gradient_boosting", {"max_iter": 20, "max_leaf_nodes": 7})]
    best, table = tune_forward_events(frame, specs=specs, tuning_events=3, min_fit_events=6)
    assert best.name == "hist_gradient_boosting"
    assert table.iloc[0].events == 3
    assert np.isfinite(table.iloc[0].mean_position_mae)

    changed = frame.copy()
    last = changed.event_id.drop_duplicates().iloc[-1]
    changed.loc[changed.event_id == last, "finish_position"] = (
        changed.loc[changed.event_id == last, "finish_position"].max() + 1
        - changed.loc[changed.event_id == last, "finish_position"])
    # Earlier tuning events are unaffected if the mutated final event is excluded.
    earlier = frame[frame.event_id != last]
    changed_earlier = changed[changed.event_id != last]
    best_a, table_a = tune_forward_events(earlier, specs=specs, tuning_events=2, min_fit_events=6)
    best_b, table_b = tune_forward_events(changed_earlier, specs=specs, tuning_events=2, min_fit_events=6)
    assert best_a == best_b
    pd.testing.assert_frame_equal(table_a, table_b)
