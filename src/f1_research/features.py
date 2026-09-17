"""Post-qualifying forecasts: historical outcomes update AFTER the whole event."""

from collections import defaultdict

import numpy as np
import pandas as pd

CATEGORICAL = ["team", "circuit"]
NUMERIC = ["quali_position", "quali_gap_pct", "driver_form", "team_form",
           "driver_dnf_rate", "driver_points_before", "team_points_before",
           "circuit_form", "history_count"]
FEATURES = CATEGORICAL + NUMERIC


def build_features(frame):
    df = frame.sort_values(["date", "event_id", "driver"]).copy()
    driver_history, team_history, circuit_history = (defaultdict(list) for _ in range(3))
    driver_points, team_points = defaultdict(float), defaultdict(float)
    records = []
    # Simultaneous events cannot see each other's outcomes.
    for _, date_group in df.groupby("date", sort=True):
        for _, event in date_group.groupby("event_id", sort=True):
            pole = event["quali_seconds"].where(event["quali_seconds"] > 0).min()
            for row in event.to_dict("records"):
                driver, team = row["driver"], row["team"]
                hist = driver_history[driver]
                team_hist = team_history[team]
                circuit_hist = circuit_history[(driver, row["circuit"])]
                row.update({
                    "quali_gap_pct": 100 * (row["quali_seconds"] / pole - 1)
                    if np.isfinite(pole) else np.nan,
                    "driver_form": np.mean([h[0] for h in hist[-5:]]) if hist else 0.5,
                    "team_form": np.mean(team_hist[-5:]) if team_hist else 0.5,
                    "driver_dnf_rate": (sum(h[1] for h in hist) + 1) / (len(hist) + 10),
                    "driver_points_before": driver_points[(row["year"], driver)],
                    "team_points_before": team_points[(row["year"], team)],
                    "circuit_form": np.mean(circuit_hist[-3:]) if circuit_hist else 0.5,
                    "history_count": len(hist),
                })
                records.append(row)
        for _, event in date_group.groupby("event_id"):
            n = len(event)
            team_event = defaultdict(list)
            for row in event.to_dict("records"):
                # Entries for inference deliberately have no results.
                if pd.isna(row.get("finish_position")):
                    continue
                result = (row["finish_position"] - 1) / max(n - 1, 1)
                driver_history[row["driver"]].append((result, row["dnf"]))
                circuit_history[(row["driver"], row["circuit"])].append(result)
                team_event[row["team"]].append(result)
                driver_points[(row["year"], row["driver"])] += row["points"]
                team_points[(row["year"], row["team"])] += row["points"]
            for team, values in team_event.items():
                team_history[team].append(float(np.mean(values)))
    return pd.DataFrame(records)
