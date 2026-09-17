"""Deterministic fictional events for an offline software demonstration only."""

import numpy as np
import pandas as pd


def synthetic_history(seed=42, events=16, drivers=10):
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=drivers)
    rows = []
    for event in range(events):
        date = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=14 * event)
        qualifying = np.argsort(ability + rng.normal(0, .5, drivers))
        result = np.argsort(ability + rng.normal(0, .8, drivers))
        for driver in range(drivers):
            position = int(np.flatnonzero(result == driver)[0]) + 1
            qpos = int(np.flatnonzero(qualifying == driver)[0]) + 1
            rows.append({"event_id": f"synthetic-{event:02d}", "date": date.isoformat(),
                         "year": 2021, "round": event + 1,
                         "driver": f"Synthetic Driver {driver + 1:02d}",
                         "team": f"Synthetic Team {driver // 2 + 1}",
                         "circuit": f"Synthetic Circuit {event % 3}",
                         "quali_position": qpos, "quali_seconds": 90 + .2 * qpos,
                         "quali_time_basis": "Q1", "finish_position": position,
                         "points": float(max(0, 11 - position)), "dnf": 0})
    return pd.DataFrame(rows)
