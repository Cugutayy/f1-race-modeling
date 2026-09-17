"""Evidence-grade live pace view using the leakage-strict next-lap artifact."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from f1_research.lap_strict import STRICT_FEATURES, predict_live_strict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-strict" / "next_lap_strict.joblib"


def read_state(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Live state must be a JSON object")
    return value


@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime_ns: int) -> dict:
    del mtime_ns
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Strict model artifact must be a dictionary")
    if artifact.get("task") != "next_lap_strict_mixture":
        raise ValueError("Artifact is not a strict next-lap model")
    if artifact.get("features") != STRICT_FEATURES:
        raise ValueError("Strict model feature schema mismatch")
    if artifact.get("retrospective_stint_features_used") is not False:
        raise ValueError("Artifact does not prove strict stint-feature exclusion")
    return artifact


def state_age_s(state: dict) -> float:
    updated = pd.to_datetime(state.get("updated_at"), utc=True, errors="coerce")
    if pd.isna(updated):
        return float("inf")
    return max(0.0, (pd.Timestamp(datetime.now(UTC)) - updated).total_seconds())


def enrich(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    lookup = {str(int(row["driver_number"])): row for row in state.get("drivers", [])}
    output = frame.copy()
    output["driver"] = [
        lookup.get(str(number), {}).get("acronym")
        or lookup.get(str(number), {}).get("full_name")
        or str(number)
        for number in output.driver_number
    ]
    output["team"] = [lookup.get(str(number), {}).get("team_name") for number in output.driver_number]
    output["position"] = [lookup.get(str(number), {}).get("position") for number in output.driver_number]
    output["compound_display"] = [lookup.get(str(number), {}).get("compound") for number in output.driver_number]
    output["tyre_age_display"] = [lookup.get(str(number), {}).get("tyre_age") for number in output.driver_number]
    output["delta_vs_median_s"] = output.predicted_green_lap_s - output.recent_median_5_s
    output["delta_vs_last_s"] = output.predicted_green_lap_s - output.last_lap_s
    return output.sort_values(["position", "predicted_green_lap_s"], na_position="last")


def pace_plot(frame: pd.DataFrame) -> alt.Chart:
    points = frame[["driver", "predicted_green_lap_s", "recent_median_5_s", "last_lap_s"]].melt(
        "driver", var_name="signal", value_name="lap_time_s"
    )
    base = alt.Chart(points).mark_point(size=95).encode(
        x=alt.X("lap_time_s:Q", title="Lap time · seconds", scale=alt.Scale(zero=False)),
        y=alt.Y("driver:N", sort=alt.SortField(field="lap_time_s", order="ascending"), title=None),
        shape=alt.Shape("signal:N", title="Signal"),
        tooltip=["driver:N", "signal:N", alt.Tooltip("lap_time_s:Q", format=".3f")],
    )
    interval = alt.Chart(frame).mark_rule(size=3).encode(
        x=alt.X("green_lap_lower_s:Q", title="Lap time · seconds"),
        x2="green_lap_upper_s:Q",
        y=alt.Y("driver:N", title=None),
        tooltip=[
            "driver:N",
            alt.Tooltip("green_lap_lower_s:Q", format=".3f"),
            alt.Tooltip("green_lap_upper_s:Q", format=".3f"),
        ],
    )
    return (interval + base).properties(height=max(360, len(frame) * 26))


def regime_plot(frame: pd.DataFrame) -> alt.Chart:
    long = frame[["driver", "p_green", "p_neutralized", "p_pit"]].melt(
        "driver", var_name="regime", value_name="probability"
    )
    long["regime"] = long.regime.str.removeprefix("p_")
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("probability:Q", title="Probability", axis=alt.Axis(format="%")),
        y=alt.Y("driver:N", title=None),
        row=alt.Row("regime:N", title=None),
        tooltip=["driver:N", "regime:N", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=max(100, len(frame) * 17))


def main() -> None:
    st.set_page_config(page_title="F1 · Strict Live Pace", page_icon="🏎️", layout="wide")
    st.title("F1 · Strict Live Pace")
    st.caption(
        "OpenF1 event-time state → leakage-strict local model → conformal pace interval + lap-regime probabilities"
    )

    with st.sidebar:
        state_path = Path(st.text_input("Live state", os.environ.get("F1_LIVE_STATE_PATH", str(DEFAULT_STATE))))
        model_path = Path(st.text_input("Strict artifact", os.environ.get("F1_STRICT_MODEL_PATH", str(DEFAULT_MODEL))))
        refresh = st.slider("Refresh seconds", 1.0, 10.0, 2.0, 0.5)
        st.code("f1-laps-strict --year 2026 --race-count 8")
        st.caption("Use --foundation to let local TabICLv2 compete under the exact same chronological validation.")

    @st.fragment(run_every=refresh)
    def live_view():
        if not state_path.exists():
            st.warning("Live state is missing. Start `f1-live capture` first.")
            return
        if not model_path.exists():
            st.warning("Strict model artifact is missing. Run the strict training command first.")
            return
        try:
            state = read_state(state_path)
            artifact = load_model(str(model_path), model_path.stat().st_mtime_ns)
            frame = enrich(predict_live_strict(artifact, state), state)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Strict live pace input error: {exc}")
            return
        if frame.empty:
            st.info("At least three completed laps per driver are required before inference begins.")
            return

        age = state_age_s(state)
        a, b, c, d = st.columns(4)
        a.metric("Selected model", artifact.get("selected_regressor", "—"))
        b.metric("Lap", state.get("current_lap") or "—")
        c.metric("State age", f"{age:.1f}s")
        d.metric("Predicted drivers", len(frame))
        if age > max(10.0, refresh * 4):
            st.warning("State is stale. Forecasts are frozen until new provider messages arrive.")

        next_lap, comparison, regimes, audit = st.tabs(
            ["Next lap", "Model vs baseline", "Lap regime", "Evidence audit"]
        )
        with next_lap:
            columns = [
                "position", "driver", "team", "compound_display", "tyre_age_display", "lap_number",
                "predicted_green_lap_s", "green_lap_lower_s", "green_lap_upper_s",
                "recent_median_5_s", "last_lap_s", "delta_vs_median_s",
                "p_green", "p_pit", "p_neutralized",
            ]
            st.dataframe(
                frame[[column for column in columns if column in frame]],
                hide_index=True,
                width="stretch",
                column_config={
                    "predicted_green_lap_s": st.column_config.NumberColumn("Green pace", format="%.3fs"),
                    "green_lap_lower_s": st.column_config.NumberColumn("Lower", format="%.3fs"),
                    "green_lap_upper_s": st.column_config.NumberColumn("Upper", format="%.3fs"),
                    "recent_median_5_s": st.column_config.NumberColumn("5-lap median", format="%.3fs"),
                    "last_lap_s": st.column_config.NumberColumn("Last lap", format="%.3fs"),
                    "delta_vs_median_s": st.column_config.NumberColumn("Δ vs median", format="%+.3fs"),
                    "p_green": st.column_config.NumberColumn("P(green)", format="percent"),
                    "p_pit": st.column_config.NumberColumn("P(pit)", format="percent"),
                    "p_neutralized": st.column_config.NumberColumn("P(neutralized)", format="percent"),
                },
            )
            st.altair_chart(pace_plot(frame), width="stretch")
            st.caption(
                "Compound and tyre age are display-only here. They are deliberately excluded from the strict historical model."
            )

        with comparison:
            delta = frame[["driver", "delta_vs_median_s", "delta_vs_last_s"]].melt(
                "driver", var_name="baseline", value_name="delta_s"
            )
            st.altair_chart(
                alt.Chart(delta).mark_bar().encode(
                    x=alt.X("delta_s:Q", title="Model minus baseline · seconds"),
                    y=alt.Y("driver:N", title=None),
                    column=alt.Column("baseline:N", title=None),
                    tooltip=["driver:N", "baseline:N", alt.Tooltip("delta_s:Q", format="+.3f")],
                ).properties(height=max(320, len(frame) * 22)),
                width="stretch",
            )
            st.caption("A negative delta means the model expects a faster lap than that baseline; it is not itself evidence of better accuracy.")

        with regimes:
            st.altair_chart(regime_plot(frame), width="stretch")
            st.caption(
                "Neutralized probability only reflects state known at the cutoff; an unexpected Safety Car beginning mid-lap cannot be known in advance."
            )

        with audit:
            st.json({
                "artifact": {
                    "task": artifact.get("task"),
                    "selected_regressor": artifact.get("selected_regressor"),
                    "trained_through_session": artifact.get("trained_through_session"),
                    "calibration_session": artifact.get("calibration_session"),
                    "sealed_test_session": artifact.get("sealed_test_session"),
                    "conformal_alpha": artifact.get("conformal_alpha"),
                    "conformal_radius_s": artifact.get("conformal_radius_s"),
                    "retrospective_stint_features_used": artifact.get("retrospective_stint_features_used"),
                    "features": artifact.get("features"),
                },
                "live_state": {
                    "session_key": state.get("session_key"),
                    "updated_at": state.get("updated_at"),
                    "data_age_s": age,
                    "messages": state.get("received_messages"),
                },
            }, expanded=False)
            st.warning(
                "This is public-data research, not team telemetry. Fuel, setup, internal tyre temperature and many engineering channels are unavailable."
            )

    live_view()


if __name__ == "__main__":
    main()
