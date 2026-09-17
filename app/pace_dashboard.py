"""Live next-lap prediction dashboard backed by a trained local artifact."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from f1_research.lap_intelligence import FEATURES, live_feature_rows
from f1_research.lap_pipeline import load_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "reports" / "local" / "live" / "state.json"
DEFAULT_MODEL = ROOT / "reports" / "local" / "lap-intelligence" / "next_lap_model.joblib"


def read_state(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Live state must be an object")
    return value


@st.cache_resource(show_spinner=False)
def cached_artifact(path: str, mtime_ns: int):
    del mtime_ns
    return load_artifact(Path(path))


def state_age_s(state: dict) -> float:
    raw = state.get("updated_at")
    if not raw:
        return float("inf")
    updated = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(updated):
        return float("inf")
    return max(0.0, (pd.Timestamp(datetime.now(UTC)) - updated).total_seconds())


def build_prediction_table(state: dict, artifact: dict) -> pd.DataFrame:
    features = live_feature_rows(state)
    if features.empty:
        return features
    pipeline = artifact["pipeline"]
    predicted = pipeline.predict(features[FEATURES])
    features = features.copy()
    features["predicted_next_lap_s"] = predicted
    features["model_vs_recent_median_s"] = features.predicted_next_lap_s - features.recent_median_5_s
    features["model_vs_last_lap_s"] = features.predicted_next_lap_s - features.last_lap_s
    driver_lookup = {str(int(row["driver_number"])): row for row in state.get("drivers", [])}
    features["driver"] = [
        (driver_lookup.get(str(number), {}).get("acronym")
         or driver_lookup.get(str(number), {}).get("full_name")
         or str(number))
        for number in features.driver_number
    ]
    features["team"] = [driver_lookup.get(str(number), {}).get("team_name") for number in features.driver_number]
    features["position"] = [driver_lookup.get(str(number), {}).get("position") for number in features.driver_number]
    features["recent_sigma_s"] = [
        float(pd.Series(driver_lookup.get(str(number), {}).get("recent_laps_s") or []).tail(5).std())
        for number in features.driver_number
    ]
    return features.sort_values(["position", "predicted_next_lap_s"], na_position="last")


def pace_chart(frame: pd.DataFrame) -> alt.Chart:
    long = frame[["driver", "predicted_next_lap_s", "recent_median_5_s", "last_lap_s"]].melt(
        "driver", var_name="series", value_name="lap_time_s")
    return alt.Chart(long).mark_point(size=90).encode(
        x=alt.X("lap_time_s:Q", title="Lap time · seconds", scale=alt.Scale(zero=False)),
        y=alt.Y("driver:N", sort=alt.SortField(field="lap_time_s", order="ascending"), title=None),
        shape=alt.Shape("series:N", title="Signal"),
        tooltip=["driver:N", "series:N", alt.Tooltip("lap_time_s:Q", format=".3f")],
    ).properties(height=max(340, len(frame) * 25))


def main() -> None:
    st.set_page_config(page_title="F1 · Live Pace AI", page_icon="🏁", layout="wide")
    st.title("F1 · Live Pace AI")
    st.caption("Captured OpenF1 state → local trained model → next-lap forecast. No model artifact, no invented prediction.")

    with st.sidebar:
        state_path = Path(st.text_input("Live state", os.environ.get("F1_LIVE_STATE_PATH", str(DEFAULT_STATE))))
        model_path = Path(st.text_input("Next-lap artifact", os.environ.get("F1_LAP_MODEL_PATH", str(DEFAULT_MODEL))))
        refresh = st.slider("Refresh seconds", 1.0, 10.0, 2.0, 0.5)
        st.code("python -m f1_research.lap_pipeline --year 2026 --race-count 8")
        st.caption("Add --foundation to evaluate local TabICLv2 on the same whole-session validation split.")

    @st.fragment(run_every=refresh)
    def live_pace():
        if not state_path.exists():
            st.warning("Live state is missing. Start the OpenF1 capture service first.")
            return
        if not model_path.exists():
            st.warning("Next-lap model artifact is missing. Train the lap pipeline first.")
            return
        try:
            state = read_state(state_path)
            artifact = cached_artifact(str(model_path), model_path.stat().st_mtime_ns)
            frame = build_prediction_table(state, artifact)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Live pace input error: {exc}")
            return
        if frame.empty:
            st.info("At least three completed laps per driver are required before next-lap inference starts.")
            return

        age = state_age_s(state)
        a, b, c, d = st.columns(4)
        a.metric("Model", artifact.get("model_name", "—"))
        b.metric("Current lap", state.get("current_lap") or "—")
        c.metric("State age", f"{age:.1f}s")
        d.metric("Drivers predicted", len(frame))
        if age > max(10.0, refresh * 4):
            st.warning("State is stale; forecasts are frozen until new timing data arrives.")

        overview, chart_tab, model_tab = st.tabs(["Next lap", "Pace comparison", "Model audit"])
        with overview:
            display = frame[["position", "driver", "team", "compound", "tyre_age", "lap_number",
                             "predicted_next_lap_s", "recent_median_5_s", "last_lap_s",
                             "model_vs_recent_median_s", "recent_sigma_s"]].copy()
            st.dataframe(display, hide_index=True, width="stretch", column_config={
                "predicted_next_lap_s": st.column_config.NumberColumn("Pred next lap", format="%.3fs"),
                "recent_median_5_s": st.column_config.NumberColumn("5-lap median", format="%.3fs"),
                "last_lap_s": st.column_config.NumberColumn("Last lap", format="%.3fs"),
                "model_vs_recent_median_s": st.column_config.NumberColumn("Δ vs median", format="%+.3fs"),
                "recent_sigma_s": st.column_config.NumberColumn("Recent σ", format="%.3fs"),
            })
            st.caption("Recent σ is an observed-pace stability diagnostic, not a calibrated predictive interval.")

        with chart_tab:
            st.altair_chart(pace_chart(frame), width="stretch")
            delta = frame[["driver", "model_vs_recent_median_s"]].copy()
            st.altair_chart(alt.Chart(delta).mark_bar().encode(
                x=alt.X("model_vs_recent_median_s:Q", title="Predicted delta vs 5-lap median · s"),
                y=alt.Y("driver:N", sort="x", title=None),
                tooltip=["driver:N", alt.Tooltip("model_vs_recent_median_s:Q", format="+.3f")],
            ).properties(height=max(320, len(delta) * 24)), width="stretch")

        with model_tab:
            st.json({
                "artifact": {key: artifact.get(key) for key in (
                    "schema_version", "task", "model_name", "trained_through_session",
                    "sealed_test_session", "features")},
                "live_state": {"session_key": state.get("session_key"), "updated_at": state.get("updated_at"),
                               "data_age_s": age, "messages": state.get("received_messages")},
            }, expanded=False)
            st.warning("Historical stint features lack as-published timestamps. Live captured state is stronger evidence than retrospective REST joins.")

    live_pace()


if __name__ == "__main__":
    main()
