"""Unified live F1 race intelligence view.

Combines captured OpenF1 state, the leakage-strict local next-lap model, calibrated
public-data strategy priors and transparent Monte Carlo outcome simulation.
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from f1_research.live_intelligence import (
    combined_live_report,
    combined_pit_windows,
    load_strict_artifact,
)
from f1_research.strategy import SimulationConfig, compare_pit_windows, predict_from_state
from f1_research.strategy_calibration import load_simulation_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"
DEFAULT_PRIORS = ROOT / "reports" / "local" / "lap-strict" / "strategy_priors.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def state_age_s(state: dict) -> float:
    updated = pd.to_datetime(state.get("updated_at"), utc=True, errors="coerce")
    if pd.isna(updated):
        return float("inf")
    return max(0.0, float((pd.Timestamp(datetime.now(UTC)) - updated).total_seconds()))


def simulation_config(path: Path, samples: int) -> tuple[SimulationConfig, dict]:
    if not path.exists():
        return SimulationConfig(samples=samples), {
            "source": "built_in_defaults",
            "warning": "No calibrated strategy prior artifact found.",
        }
    return load_simulation_config(path, samples=samples)


def telemetry_tail(path: Path, driver_number: int, max_rows: int = 1200) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = deque(handle, maxlen=20_000)
    rows = []
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("topic") != "car_data":
            continue
        payload = item.get("payload", {})
        if payload.get("driver_number") != driver_number:
            continue
        rows.append({
            "date": payload.get("date") or item.get("received_at"),
            "speed": payload.get("speed"),
            "throttle": payload.get("throttle"),
            "brake": payload.get("brake"),
            "rpm": payload.get("rpm"),
            "gear": payload.get("n_gear"),
            "drs": payload.get("drs"),
        })
        if len(rows) >= max_rows:
            break
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(reversed(rows))
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="coerce")
    for column in frame.columns.drop("date"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"])


def enrich_race(predictions: list[dict], state: dict) -> pd.DataFrame:
    race = pd.DataFrame(predictions)
    if race.empty:
        return race
    metadata = pd.DataFrame(state.get("drivers", []))
    if not metadata.empty:
        keep = [
            column for column in (
                "driver_number", "position", "team_name", "compound", "tyre_age",
                "gap_to_leader_s", "last_lap_s",
            ) if column in metadata
        ]
        race = race.merge(metadata[keep], on="driver_number", how="left")
    race["win_pct"] = race.win_probability * 100
    race["podium_pct"] = race.podium_probability * 100
    race["dnf_pct"] = race.dnf_probability * 100
    return race.sort_values(["expected_position", "driver_number"])


def enrich_pace(rows: list[dict], state: dict) -> pd.DataFrame:
    pace = pd.DataFrame(rows)
    if pace.empty:
        return pace
    metadata = {
        int(row["driver_number"]): row
        for row in state.get("drivers", [])
        if row.get("driver_number") is not None
    }
    pace["driver"] = [
        metadata.get(int(number), {}).get("acronym")
        or metadata.get(int(number), {}).get("full_name")
        or str(number)
        for number in pace.driver_number
    ]
    pace["position"] = [metadata.get(int(number), {}).get("position") for number in pace.driver_number]
    pace["compound_display"] = [metadata.get(int(number), {}).get("compound") for number in pace.driver_number]
    pace["tyre_age_display"] = [metadata.get(int(number), {}).get("tyre_age") for number in pace.driver_number]
    pace["delta_vs_median_s"] = pace.predicted_green_lap_s - pace.recent_median_5_s
    for column in ("p_green", "p_pit", "p_neutralized"):
        if column in pace:
            pace[f"{column}_pct"] = pace[column] * 100
    return pace.sort_values(["position", "predicted_green_lap_s"], na_position="last")


def probability_chart(frame: pd.DataFrame) -> alt.Chart:
    top = frame.sort_values("win_probability", ascending=False).head(12)
    return alt.Chart(top).mark_bar().encode(
        x=alt.X("win_probability:Q", title="Win probability", axis=alt.Axis(format="%")),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[
            "label:N",
            alt.Tooltip("win_probability:Q", format=".1%"),
            alt.Tooltip("podium_probability:Q", format=".1%"),
            alt.Tooltip("expected_position:Q", format=".2f"),
        ],
    ).properties(height=max(300, len(top) * 25))


def pace_chart(frame: pd.DataFrame) -> alt.Chart:
    long = frame[["driver", "predicted_green_lap_s", "recent_median_5_s", "last_lap_s"]].melt(
        "driver", var_name="signal", value_name="lap_time_s"
    )
    points = alt.Chart(long).mark_point(size=95).encode(
        x=alt.X("lap_time_s:Q", title="Lap time · seconds", scale=alt.Scale(zero=False)),
        y=alt.Y("driver:N", title=None),
        shape=alt.Shape("signal:N", title="Signal"),
        tooltip=["driver:N", "signal:N", alt.Tooltip("lap_time_s:Q", format=".3f")],
    )
    interval = alt.Chart(frame).mark_rule(size=3).encode(
        x="green_lap_lower_s:Q",
        x2="green_lap_upper_s:Q",
        y="driver:N",
    )
    return (interval + points).properties(height=max(340, len(frame) * 24))


def location_chart(state: dict) -> alt.Chart | None:
    frame = pd.DataFrame(state.get("drivers", []))
    if frame.empty or not {"x", "y"} <= set(frame):
        return None
    frame = frame.dropna(subset=["x", "y"])
    if frame.empty:
        return None
    frame["driver"] = frame.acronym.fillna(frame.driver_number.astype(str))
    return alt.Chart(frame).mark_circle(size=130).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        tooltip=["driver:N", "position:Q", "speed_kmh:Q", "compound:N"],
    ).properties(height=390)


def probability_history(predictions: list[dict], state_time: str | None) -> pd.DataFrame:
    history = st.session_state.setdefault("race_intel_probability_history", [])
    previous = st.session_state.get("race_intel_probability_state")
    if state_time and state_time != previous:
        history.extend({
            "time": state_time,
            "driver": row["label"],
            "win_probability": row["win_probability"],
        } for row in predictions)
        st.session_state.race_intel_probability_history = history[-1500:]
        st.session_state.race_intel_probability_state = state_time
    frame = pd.DataFrame(st.session_state.race_intel_probability_history)
    if not frame.empty:
        frame["time"] = pd.to_datetime(frame.time, utc=True, errors="coerce")
    return frame


def main() -> None:
    st.set_page_config(page_title="F1 · Race Intelligence Center", page_icon="🏎️", layout="wide")
    st.title("F1 · Race Intelligence Center")
    st.caption(
        "Live OpenF1 state → strict local pace AI → calibrated public-data priors → transparent race simulation"
    )

    with st.sidebar:
        state_path = Path(st.text_input(
            "Live state", os.environ.get("F1_LIVE_STATE_PATH", str(DEFAULT_STATE))))
        model_path = Path(st.text_input(
            "Strict pace artifact", os.environ.get("F1_STRICT_MODEL_PATH", str(DEFAULT_MODEL))))
        priors_path = Path(st.text_input(
            "Strategy priors", os.environ.get("F1_STRATEGY_PRIORS_PATH", str(DEFAULT_PRIORS))))
        total_laps = int(st.number_input("Race total laps", 2, 100, 57, 1))
        samples = int(st.select_slider(
            "Monte Carlo samples", options=[2000, 4000, 8000, 12000, 20000], value=4000))
        refresh = st.slider("Refresh seconds", 1.0, 10.0, 2.0, 0.5)
        st.code("f1-live capture --output reports/local/live")
        st.code("f1-laps-strict --year 2026 --race-count 8")

    events_path = state_path.with_name("events.jsonl")

    @st.fragment(run_every=refresh)
    def live_view():
        if not state_path.exists():
            st.warning("Live state is missing. Start the capture service first.")
            return
        try:
            state = read_json(state_path)
            sim_config, prior_audit = simulation_config(priors_path, samples)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Live state / strategy prior error: {exc}")
            return

        artifact = None
        pace_mode = "robust recent-lap fallback"
        try:
            if model_path.exists():
                artifact = load_strict_artifact(model_path)
                report = combined_live_report(state, total_laps, artifact, sim_config)
                pace_mode = f"strict AI · {artifact.get('selected_regressor', 'model')}"
            else:
                report = predict_from_state(state, total_laps, config=sim_config)
                report["pace_predictions"] = []
        except ValueError as exc:
            st.info(f"Simulation warm-up: {exc}")
            return

        race = enrich_race(report["predictions"], state)
        pace = enrich_pace(report.get("pace_predictions", []), state)
        age = state_age_s(state)
        a, b, c, d, e = st.columns(5)
        a.metric("Lap", f"{state.get('current_lap', '—')} / {total_laps}")
        b.metric("Pace source", pace_mode)
        c.metric("State age", "unknown" if age == float("inf") else f"{age:.1f}s")
        d.metric("Simulations", f"{samples:,}")
        e.metric("Messages", state.get("received_messages", 0))
        if age > max(10, refresh * 4):
            st.warning("State is stale; forecasts are frozen until new timing data arrives.")
        if artifact is None:
            st.warning("Strict pace artifact not found. Race simulation is using robust recent-lap fallback.")
        if prior_audit.get("source") == "built_in_defaults":
            st.warning("Calibrated strategy priors not found. Simulator is using documented defaults.")

        race_tab, pace_tab, strategy_tab, telemetry_tab, track_tab, audit_tab = st.tabs(
            ["Race", "Pace AI", "Strategy Lab", "Telemetry", "Track", "Evidence audit"])

        with race_tab:
            st.altair_chart(probability_chart(race), width="stretch")
            show = race[[
                "label", "position", "compound", "tyre_age", "gap_to_leader_s",
                "expected_position", "win_pct", "podium_pct", "position_p10", "position_p90", "dnf_pct",
            ]].copy()
            st.dataframe(show, hide_index=True, width="stretch")
            history = probability_history(report["predictions"], state.get("updated_at"))
            if not history.empty:
                leaders = set(race.nsmallest(6, "expected_position").label)
                history = history[history.driver.isin(leaders)]
                st.altair_chart(alt.Chart(history).mark_line().encode(
                    x=alt.X("time:T", title=None),
                    y=alt.Y("win_probability:Q", title="Win probability", axis=alt.Axis(format="%")),
                    color=alt.Color("driver:N", title="Driver"),
                    tooltip=["driver:N", alt.Tooltip("win_probability:Q", format=".1%")],
                ).properties(height=300), width="stretch")

        with pace_tab:
            if pace.empty:
                st.info("Strict pace model is unavailable or still warming up.")
            else:
                st.altair_chart(pace_chart(pace), width="stretch")
                columns = [
                    "position", "driver", "compound_display", "tyre_age_display", "lap_number",
                    "predicted_green_lap_s", "green_lap_lower_s", "green_lap_upper_s",
                    "recent_median_5_s", "last_lap_s", "delta_vs_median_s",
                    "p_green_pct", "p_pit_pct", "p_neutralized_pct",
                ]
                st.dataframe(pace[[column for column in columns if column in pace]],
                             hide_index=True, width="stretch")
                st.caption(
                    "Compound and tyre age are display-only for the strict historical model; they do not enter its features."
                )

        with strategy_tab:
            options = {
                (row.get("acronym") or str(row["driver_number"])): int(row["driver_number"])
                for row in state.get("drivers", []) if row.get("position")
            }
            if not options:
                st.info("No positioned drivers yet.")
            else:
                selected = st.selectbox("Driver", list(options), key="race_intel_strategy_driver")
                scenario_key = (
                    state.get("updated_at"), options[selected], total_laps, samples,
                    str(model_path), str(priors_path),
                )
                if st.button("Run pit-window scenarios", type="primary"):
                    with st.spinner("Running common-seed counterfactuals…"):
                        try:
                            if artifact is not None:
                                scenarios = combined_pit_windows(
                                    state, total_laps, options[selected], artifact, sim_config)
                            else:
                                scenarios = compare_pit_windows(
                                    state, total_laps, options[selected], config=sim_config)
                            st.session_state.race_intel_scenarios = scenarios
                            st.session_state.race_intel_scenario_key = scenario_key
                        except ValueError as exc:
                            st.error(str(exc))
                scenarios = (
                    st.session_state.get("race_intel_scenarios")
                    if st.session_state.get("race_intel_scenario_key") == scenario_key
                    else None
                )
                if scenarios:
                    table = pd.DataFrame(scenarios).sort_values(["expected_position", "pit_in_laps"])
                    table["win_pct"] = table.win_probability * 100
                    table["podium_pct"] = table.podium_probability * 100
                    st.dataframe(table[[
                        "pit_in_laps", "compound", "expected_position", "win_pct", "podium_pct",
                        "position_p10", "position_p90", "dnf_probability",
                    ]], hide_index=True, width="stretch")
                    st.altair_chart(alt.Chart(table).mark_line(point=True).encode(
                        x=alt.X("pit_in_laps:Q", title="Pit in laps from now"),
                        y=alt.Y("expected_position:Q", title="Expected finish", scale=alt.Scale(reverse=True)),
                        color=alt.Color("compound:N", title="Next compound"),
                        tooltip=[
                            "compound:N", "pit_in_laps:Q",
                            alt.Tooltip("expected_position:Q", format=".2f"),
                            alt.Tooltip("win_probability:Q", format=".1%"),
                        ],
                    ).properties(height=300), width="stretch")

        with telemetry_tab:
            options = {
                (row.get("acronym") or str(row["driver_number"])): int(row["driver_number"])
                for row in state.get("drivers", [])
            }
            if options:
                selected = st.selectbox("Telemetry driver", list(options), key="race_intel_telemetry_driver")
                telemetry = telemetry_tail(events_path, options[selected])
                if telemetry.empty:
                    st.info("No captured car_data samples for this driver yet.")
                else:
                    for column, title in (
                        ("speed", "Speed · km/h"),
                        ("throttle", "Throttle · %"),
                        ("brake", "Brake · 0/100"),
                    ):
                        valid = telemetry.dropna(subset=[column])
                        if not valid.empty:
                            st.altair_chart(alt.Chart(valid).mark_line().encode(
                                x=alt.X("date:T", title=None),
                                y=alt.Y(f"{column}:Q", title=title),
                            ).properties(height=145), width="stretch")
                    st.caption("Public broadcast telemetry is not the complete team sensor feed.")

        with track_tab:
            chart = location_chart(state)
            if chart is None:
                st.info("Approximate x/y location samples are not available yet.")
            else:
                st.altair_chart(chart, width="stretch")
                st.caption("OpenF1 x/y is an approximate track view, not precision racing-line GPS.")

        with audit_tab:
            st.json({
                "state": {
                    "session_key": state.get("session_key"),
                    "updated_at": state.get("updated_at"),
                    "data_age_s": age,
                    "received_messages": state.get("received_messages"),
                    "rejected_stale_messages": state.get("rejected_stale_messages"),
                },
                "race_simulation": report.get("audit"),
                "pace_model": report.get("pace_model"),
                "strategy_priors": prior_audit,
            }, expanded=False)
            st.warning(
                "Public-data research only: fuel load, setup, internal tyre temperatures and many team engineering channels are unavailable."
            )

    live_view()


if __name__ == "__main__":
    main()
