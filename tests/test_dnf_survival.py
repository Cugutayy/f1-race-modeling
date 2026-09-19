import numpy as np
import pandas as pd
import pytest

from f1_research.dnf_survival import AFTTargets, build_aft_frame, concordance_index


def _history():
    rows = []
    for event_index, date in enumerate(("2026-03-01", "2026-03-08"), start=1):
        for position, driver, dnf, laps in [
            (1, "a", 0, 60),
            (2, "b", 0, 60),
            (3, "c", 1, 31 + event_index),
        ]:
            rows.append(
                {
                    "event_id": f"2026-{event_index:02d}",
                    "date": date,
                    "year": 2026,
                    "round": event_index,
                    "driver": driver,
                    "team": f"team-{driver}",
                    "circuit": f"circuit-{event_index}",
                    "quali_position": float(position),
                    "quali_seconds": 80.0 + position,
                    "finish_position": float(position),
                    "points": 25.0 if position == 1 else 0.0,
                    "dnf": dnf,
                    "status": "Finished" if not dnf else "Accident",
                    "starter_count": 3,
                    "laps_completed": laps,
                    "race_total_laps": 60,
                }
            )
    return pd.DataFrame(rows)


def test_aft_targets_keep_finishers_as_right_censored_not_dropped():
    featured, target = build_aft_frame(_history())
    assert len(featured) == 6
    assert target.observed.sum() == 2
    assert np.isinf(target.upper[~target.observed]).all()
    assert np.all(target.lower[~target.observed] == 1.0)
    assert np.isfinite(target.upper[target.observed]).all()
    assert np.all(target.upper[target.observed] < 1.0)


def test_concordance_rewards_higher_risk_for_earlier_retirement():
    targets = AFTTargets(
        lower=np.array([0.2, 0.6, 1.0]),
        upper=np.array([0.2, 0.6, np.inf]),
        observed=np.array([True, True, False]),
        progress=np.array([0.2, 0.6, 1.0]),
    )
    assert concordance_index(targets, np.array([3.0, 2.0, 1.0])) == pytest.approx(1.0)
    assert concordance_index(targets, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)
