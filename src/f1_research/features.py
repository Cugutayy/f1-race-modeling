"""Post-qualifying forecasts: historical outcomes update AFTER the whole event."""

from collections import defaultdict

import numpy as np
import pandas as pd

CATEGORICAL = ["team", "circuit", "regulation_era"]
NUMERIC = [
    "quali_position", "quali_gap_pct", "driver_form", "team_form",
    "driver_dnf_rate", "driver_points_before", "team_points_before",
    "circuit_form", "history_count", "team_era_history_count",
]
FEATURES = CATEGORICAL + NUMERIC


def regulation_era(year: int) -> str:
    """Coarse technical-regulation regimes used to prevent blind team-form carryover."""
    year = int(year)
    if year >= 2026:
        return "2026_plus"
    if year >= 2022:
        return "2022_2025_ground_effect"
    if year >= 2017:
        return "2017_2021_wide_car"
    if year >= 2014:
        return "2014_2016_hybrid"
    return "pre_2014"


def build_features(frame):
    df = frame.sort_values(["date", "event_id", "driver"]).copy()
    driver_history = defaultdict(list)
    team_era_history = defaultdict(list)
    circuit_history = defaultdict(list)
    driver_points, team_points = defaultdict(float), defaultdict(float)
    records = []
    # Simultaneous events cannot see each other's outcomes.
    for _, date_group in df.groupby("date", sort=True):
        for _, event in date_group.groupby("event_id", sort=True):
            pole = event["quali_seconds"].where(event["quali_seconds"] > 0).min()
            for row in event.to_dict("records"):
                driver, team = row["driver"], row["team"]
                era = regulation_era(row["year"])
                hist = driver_history[driver]
                team_hist = team_era_history[(era, team)]
                circuit_hist = circuit_history[(driver, row["circuit"])]
                row.update({
                    "regulation_era": era,
                    "quali_gap_pct": 100 * (row["quali_seconds"] / pole - 1)
                    if np.isfinite(pole) else np.nan,
                    "driver_form": np.mean([h[0] for h in hist[-5:]]) if hist else 0.5,
                    "team_form": np.mean(team_hist[-5:]) if team_hist else 0.5,
                    "driver_dnf_rate": (sum(h[1] for h in hist) + 1) / (len(hist) + 10),
                    "driver_points_before": driver_points[(row["year"], driver)],
                    "team_points_before": team_points[(row["year"], team)],
                    "circuit_form": np.mean(circuit_hist[-3:]) if circuit_hist else 0.5,
                    "history_count": len(hist),
                    "team_era_history_count": len(team_hist),
                })
                records.append(row)
        for _, event in date_group.groupby("event_id"):
            n = len(event)
            team_event = defaultdict(list)
            team_eras = {}
            for row in event.to_dict("records"):
                # Entries for inference deliberately have no results.
                if pd.isna(row.get("finish_position")):
                    continue
                result = (row["finish_position"] - 1) / max(n - 1, 1)
                driver_history[row["driver"]].append((result, row["dnf"]))
                circuit_history[(row["driver"], row["circuit"])].append(result)
                team_event[row["team"]].append(result)
                team_eras[row["team"]] = regulation_era(row["year"])
                driver_points[(row["year"], row["driver"])] += row["points"]
                team_points[(row["year"], row["team"])] += row["points"]
            for team, values in team_event.items():
                team_era_history[(team_eras[team], team)].append(float(np.mean(values)))
    return pd.DataFrame(records)
