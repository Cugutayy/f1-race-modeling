"""Live Pit Wall: read-only Streamlit view over captured OpenF1 state."""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from f1_research.strategy import SimulationConfig, compare_pit_windows, predict_from_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Live state file is unexpectedly large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Live state must be a JSON object")
    return value


def state_age_seconds(state: dict) -> float:
    updated = pd.to_datetime(state.get("updated_at"), utc=True, errors="coerce")
    if pd.isna(updated):
        return float("inf")
    now = pd.Timestamp(datetime.now(UTC))
    return max(0.0, float((now - updated).total_seconds()))


def recent_telemetry(path: Path, driver_number: int, max_rows: int = 1400) -> pd.DataFrame:
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
            "speed_kmh": payload.get("speed"),
            "throttle_pct": payload.get("throttle"),
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


def leaderboard(state: dict, predictions: list[dict] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(state.get("drivers", []))
    if frame.empty:
        return frame
    keep = [
        column
        for column in (
            "position", "acronym", "driver_number", "team_name", "gap_to_leader_s",
            "interval_s", "compound", "tyre_age", "last_lap_s", "pit_stops", "speed_kmh",
        )
        if column in frame
    ]
    frame = frame[keep].copy()
    if predictions:
        probability = pd.DataFrame(predictions)[
            ["driver_number", "win_probability", "podium_probability", "expected_position",
             "position_p10", "position_p90"]
        ]
        frame = frame.merge(probability, on="driver_number", how="left")
    return frame.sort_values("position", na_position="last")


def prediction_chart(predictions: pd.DataFrame) -> alt.Chart:
    chart = predictions.sort_values("win_probability", ascending=False).head(12)
    return alt.Chart(chart).mark_bar().encode(
        x=alt.X("win_probability:Q", title="Win probability", axis=alt.Axis(format="%")),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[
            "label:N",
            alt.Tooltip("win_probability:Q", format=".1%"),
            alt.Tooltip("podium_probability:Q", format=".1%"),
            alt.Tooltip("expected_position:Q", format=".2f"),
        ],
    ).properties(height=max(280, 24 * min(12, len(chart))))


def location_chart(state: dict) -> alt.Chart | None:
    frame = pd.DataFrame(state.get("drivers", []))
    if frame.empty or not {"x", "y"} <= set(frame):
        return None
    frame = frame.dropna(subset=["x", "y"])
    if frame.empty:
        return None
    frame["label"] = frame["acronym"].fillna(frame["driver_number"].astype(str))
    points = alt.Chart(frame).mark_circle(size=130).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        tooltip=["label:N", "position:Q", "speed_kmh:Q", "compound:N"],
    )
    labels = alt.Chart(frame).mark_text(dx=12, fontSize=11).encode(
        x="x:Q", y="y:Q", text="label:N"
    )
    return (points + labels).properties(height=420)


def telemetry_charts(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Seçili sürücü için capture dosyasında henüz car_data örneği yok.")
        return
    for column, title in (
        ("speed_kmh", "Speed · km/h"),
        ("throttle_pct", "Throttle · %"),
        ("brake", "Brake · 0/100"),
    ):
        valid = frame.dropna(subset=[column])
        if valid.empty:
            continue
        chart = alt.Chart(valid).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(f"{column}:Q", title=title),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{column}:Q", format=".1f")],
        ).properties(height=145)
        st.altair_chart(chart, width="stretch")


def append_probability_history(predictions: list[dict], state_time: str | None) -> pd.DataFrame:
    if "probability_history" not in st.session_state:
        st.session_state.probability_history = []
        st.session_state.last_probability_state = None
    if state_time and state_time != st.session_state.last_probability_state:
        for row in predictions:
            st.session_state.probability_history.append({
                "time": state_time,
                "driver": row["label"],
                "win_probability": row["win_probability"],
            })
        st.session_state.last_probability_state = state_time
        st.session_state.probability_history = st.session_state.probability_history[-1200:]
    return pd.DataFrame(st.session_state.probability_history)


def main() -> None:
    st.set_page_config(page_title="F1 · Live Pit Wall", page_icon="🏎️", layout="wide")
    st.title("F1 · Live Pit Wall")
    st.caption("OpenF1 event-time state → transparent Monte Carlo → continuously refreshed research view")

    with st.sidebar:
        state_path = Path(st.text_input(
            "State file", os.environ.get("F1_LIVE_STATE_PATH", str(DEFAULT_STATE))))
        total_laps = st.number_input("Race total laps", min_value=2, max_value=100, value=57, step=1)
        samples = st.select_slider(
            "Live simulation samples", options=[2000, 4000, 8000, 12000, 20000], value=4000)
        refresh = st.slider("Refresh seconds", 1.0, 10.0, 2.0, 0.5)
        st.caption("Capture service writes the state file. This page never polls OpenF1 directly.")

    events_path = state_path.with_name("events.jsonl")

    @st.fragment(run_every=refresh)
    def live_view():
        try:
            state = read_json(state_path)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
            st.warning(f"Live state unavailable: {exc}")
            st.code("python -m f1_research.openf1_live capture --output reports/local/live")
            return

        age = state_age_seconds(state)
        current_lap = state.get("current_lap")
        a, b, c, d = st.columns(4)
        a.metric("Session", state.get("session_name") or state.get("session_key") or "—")
        b.metric("Lap", f"{current_lap or '—'} / {int(total_laps)}")
        c.metric("State age", "unknown" if age == float("inf") else f"{age:.1f}s")
        d.metric("Messages", state.get("received_messages", 0))
        if age > max(10, refresh * 4):
            st.warning("State is stale. Predictions below are frozen until the capture receives new data.")

        predictions = None
        report = None
        try:
            report = predict_from_state(
                state, int(total_laps), config=SimulationConfig(samples=int(samples)))
            predictions = report["predictions"]
        except ValueError as exc:
            st.info(f"Simulation warm-up: {exc}")

        race, probability, track, telemetry, strategy, audit = st.tabs(
            ["Race", "Probabilities", "Track", "Telemetry", "Strategy Lab", "Model / data audit"])

        with race:
            st.dataframe(leaderboard(state, predictions), hide_index=True, width="stretch")
            weather = state.get("weather") or {}
            columns = st.columns(4)
            columns[0].metric("Track °C", weather.get("track_temperature_c", "—"))
            columns[1].metric("Air °C", weather.get("air_temperature_c", "—"))
            columns[2].metric("Rain", "YES" if weather.get("rainfall") else "No")
            columns[3].metric(
                "SC / flag", state.get("safety_car") or state.get("flag") or "Green/unknown")

        with probability:
            if not predictions:
                st.info("Pace + position observations are needed before probability simulation starts.")
            else:
                pred = pd.DataFrame(predictions)
                st.altair_chart(prediction_chart(pred), width="stretch")
                history = append_probability_history(predictions, state.get("updated_at"))
                if not history.empty:
                    history["time"] = pd.to_datetime(history.time, utc=True, errors="coerce")
                    leaders = pred.nsmallest(6, "expected_position").label.tolist()
                    history = history[history.driver.isin(leaders)]
                    probability_history = alt.Chart(history).mark_line(point=False).encode(
                        x=alt.X("time:T", title=None),
                        y=alt.Y(
                            "win_probability:Q",
                            title="Win probability",
                            axis=alt.Axis(format="%"),
                        ),
                        color=alt.Color("driver:N", title="Driver"),
                        tooltip=[
                            "driver:N",
                            alt.Tooltip("time:T"),
                            alt.Tooltip("win_probability:Q", format=".1%"),
                        ],
                    ).properties(height=320)
                    st.altair_chart(probability_history, width="stretch")
                st.caption(
                    "Probabilities are coherent Monte Carlo outcomes from the current public-data state; "
                    "they are not team probabilities or betting odds.")

        with track:
            chart = location_chart(state)
            if chart is None:
                st.info("Location samples have not reached the state yet.")
            else:
                st.altair_chart(chart, width="stretch")
                st.caption(
                    "OpenF1 x/y location is approximate track position, not precision GPS or "
                    "lateral racing-line data.")

        with telemetry:
            drivers = [
                (driver.get("acronym") or str(driver["driver_number"]), int(driver["driver_number"]))
                for driver in state.get("drivers", [])
            ]
            if not drivers:
                st.info("No drivers in current state.")
            else:
                mapping = dict(drivers)
                label = st.selectbox("Telemetry driver", list(mapping), key="live_telemetry_driver")
                frame = recent_telemetry(events_path, mapping[label])
                telemetry_charts(frame)
                st.caption(
                    "Public car_data is ~3.7 Hz broadcast telemetry. Brake is binary 0/100; "
                    "this is not the team's full sensor feed.")

        with strategy:
            drivers = [
                (driver.get("acronym") or str(driver["driver_number"]), int(driver["driver_number"]))
                for driver in state.get("drivers", [])
                if driver.get("position")
            ]
            if not predictions or not drivers:
                st.info("Strategy Lab becomes available after live simulation warm-up.")
            else:
                mapping = dict(drivers)
                chosen = st.selectbox("Driver", list(mapping), key="strategy_driver")
                scenario_key = (state.get("updated_at"), mapping[chosen], int(total_laps), int(samples))
                if st.button("Run pit-window comparison", type="primary"):
                    with st.spinner("Running common-seed strategy scenarios…"):
                        try:
                            scenarios = compare_pit_windows(
                                state,
                                int(total_laps),
                                mapping[chosen],
                                config=SimulationConfig(samples=int(samples)),
                            )
                            st.session_state.strategy_result = scenarios
                            st.session_state.strategy_result_key = scenario_key
                        except ValueError as exc:
                            st.error(str(exc))
                scenarios = (
                    st.session_state.get("strategy_result")
                    if st.session_state.get("strategy_result_key") == scenario_key
                    else None
                )
                if scenarios:
                    table = pd.DataFrame(scenarios).sort_values(["expected_position", "pit_in_laps"])
                    st.dataframe(
                        table[[
                            "pit_in_laps", "compound", "expected_position", "win_probability",
                            "podium_probability", "position_p10", "position_p90", "dnf_probability",
                        ]],
                        hide_index=True,
                        width="stretch",
                    )
                    scenario_chart = alt.Chart(table).mark_line(point=True).encode(
                        x=alt.X("pit_in_laps:Q", title="Pit in laps from now"),
                        y=alt.Y(
                            "expected_position:Q",
                            title="Expected finish",
                            scale=alt.Scale(reverse=True),
                        ),
                        color="compound:N",
                        tooltip=[
                            "compound:N",
                            "pit_in_laps:Q",
                            alt.Tooltip("expected_position:Q", format=".2f"),
                            alt.Tooltip("win_probability:Q", format=".1%"),
                        ],
                    ).properties(height=300)
                    st.altair_chart(scenario_chart, width="stretch")
                    st.caption(
                        "Counterfactual sensitivity analysis: lower expected position is better. "
                        "Defaults must be validated circuit-by-circuit before stronger claims.")

        with audit:
            st.json({
                "state": {
                    "session_key": state.get("session_key"),
                    "meeting_key": state.get("meeting_key"),
                    "updated_at": state.get("updated_at"),
                    "computed_data_age_s": None if age == float("inf") else age,
                    "received_messages": state.get("received_messages"),
                    "rejected_stale_messages": state.get("rejected_stale_messages"),
                },
                "simulation": report.get("audit") if report else None,
            }, expanded=False)
            st.warning(
                "Missing team-only variables (fuel, setup, carcass/internal tyre temperatures, full sensor "
                "channels) are never imputed as if observed.")

    live_view()


if __name__ == "__main__":
    main()
