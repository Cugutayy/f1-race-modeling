"""Build contextual lap datasets from OpenF1 without future-state joins.

Weather and race-control context are joined as-of each lap start. Stint metadata is
joined by lap membership. Future weather, future race-control messages and final
race outcomes are never copied backwards into a lap forecast row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .openf1_live import OpenF1Client
from .value_parsing import strict_optional_bool


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).copy()


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")


def _race_control_context(messages: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[str, str]:
    if messages.empty or pd.isna(timestamp):
        return "UNKNOWN", "UNKNOWN"
    prior = messages[messages["date"].le(timestamp)]
    if prior.empty:
        return "GREEN_OR_UNKNOWN", "NONE_OR_UNKNOWN"
    flag = "GREEN_OR_UNKNOWN"
    safety = "NONE_OR_UNKNOWN"
    for row in prior.itertuples(index=False):
        category = str(getattr(row, "category", "") or "").upper()
        message = str(getattr(row, "message", "") or "").upper()
        raw_flag = str(getattr(row, "flag", "") or "").upper()
        if category == "FLAG" and raw_flag:
            flag = raw_flag
        if "RED FLAG" in message:
            flag = "RED"
        if "VIRTUAL SAFETY CAR" in message or "VSC" in message:
            safety = "VSC"
        elif "SAFETY CAR" in message and "VIRTUAL" not in message:
            safety = "SC"
        if any(term in message for term in ("SAFETY CAR IN", "VSC END", "GREEN FLAG")):
            safety = "NONE"
            if "GREEN" in message:
                flag = "GREEN"
    return flag, safety


def build_contextual_laps(
    laps_rows: list[dict[str, Any]],
    stints_rows: list[dict[str, Any]],
    pit_rows: list[dict[str, Any]],
    weather_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    drivers_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    laps = _frame(laps_rows)
    required = {"session_key", "driver_number", "lap_number", "date_start", "lap_duration"}
    if required - set(laps):
        raise ValueError(f"Missing lap fields: {sorted(required - set(laps))}")
    _numeric(laps, ("session_key", "driver_number", "lap_number", "lap_duration",
                    "duration_sector_1", "duration_sector_2", "duration_sector_3"))
    _dates(laps, ("date_start",))
    laps = laps.dropna(subset=["session_key", "driver_number", "lap_number", "date_start"])
    if laps.duplicated(["session_key", "driver_number", "lap_number"]).any():
        raise ValueError("Duplicate OpenF1 lap identity")
    laps[["session_key", "driver_number", "lap_number"]] = laps[
        ["session_key", "driver_number", "lap_number"]].astype("int64")

    stints = _frame(stints_rows)
    if not stints.empty:
        _numeric(stints, ("driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"))
    pits = _frame(pit_rows)
    if not pits.empty:
        _numeric(pits, ("driver_number", "lap_number", "lane_duration", "stop_duration"))
    weather = _frame(weather_rows)
    if not weather.empty:
        _dates(weather, ("date",))
        _numeric(weather, ("air_temperature", "track_temperature", "humidity", "pressure",
                           "wind_speed", "wind_direction"))
        weather = weather.dropna(subset=["date"]).sort_values("date")
    control = _frame(control_rows)
    if not control.empty:
        _dates(control, ("date",))
        control = control.dropna(subset=["date"]).sort_values("date")
    drivers = _frame(drivers_rows or [])
    driver_lookup = {}
    if not drivers.empty and "driver_number" in drivers:
        _numeric(drivers, ("driver_number",))
        driver_lookup = {
            int(row.driver_number): {
                "driver": getattr(row, "name_acronym", None),
                "team": getattr(row, "team_name", None),
            }
            for row in drivers.dropna(subset=["driver_number"]).itertuples(index=False)
        }

    pit_keys = set()
    if not pits.empty and {"driver_number", "lap_number"} <= set(pits):
        pit_keys = {
            (int(driver), int(lap))
            for driver, lap in pits[["driver_number", "lap_number"]].dropna().itertuples(index=False, name=None)
        }

    records = []
    for row in laps.sort_values(["date_start", "driver_number", "lap_number"]).to_dict("records"):
        number = int(row["driver_number"])
        lap_number = int(row["lap_number"])
        timestamp = row["date_start"]
        stint = None
        if not stints.empty:
            candidates = stints[
                stints["driver_number"].eq(number)
                & stints["lap_start"].le(lap_number)
                & (stints["lap_end"].isna() | stints["lap_end"].ge(lap_number))
            ]
            if not candidates.empty:
                stint = candidates.sort_values("stint_number").iloc[-1]
        weather_row = None
        if not weather.empty:
            prior_weather = weather[weather["date"].le(timestamp)]
            if not prior_weather.empty:
                weather_row = prior_weather.iloc[-1]
        flag, safety = _race_control_context(control, timestamp)
        metadata = driver_lookup.get(number, {})
        start_age = None if stint is None else stint.get("tyre_age_at_start")
        lap_start = None if stint is None else stint.get("lap_start")
        tyre_age = (
            float(start_age + lap_number - lap_start)
            if start_age is not None and lap_start is not None and np.isfinite(start_age) and np.isfinite(lap_start)
            else np.nan
        )
        duration = row.get("lap_duration")
        pit_out = strict_optional_bool(
            row.get("is_pit_out_lap"),
            field="openf1.laps.is_pit_out_lap",
        )
        records.append({
            "session_key": int(row["session_key"]),
            "driver_number": number,
            "driver": metadata.get("driver"),
            "team": metadata.get("team"),
            "lap_number": lap_number,
            "date_start": timestamp,
            "lap_duration": float(duration) if duration is not None and np.isfinite(duration) else np.nan,
            "sector_1_s": row.get("duration_sector_1"),
            "sector_2_s": row.get("duration_sector_2"),
            "sector_3_s": row.get("duration_sector_3"),
            "is_pit_out_lap": pit_out,
            "is_pit_lap": (number, lap_number) in pit_keys,
            "stint_number": np.nan if stint is None else stint.get("stint_number"),
            "compound": None if stint is None else stint.get("compound"),
            "tyre_age": tyre_age,
            "air_temperature_c": np.nan if weather_row is None else weather_row.get("air_temperature"),
            "track_temperature_c": np.nan if weather_row is None else weather_row.get("track_temperature"),
            "humidity_pct": np.nan if weather_row is None else weather_row.get("humidity"),
            "rainfall": None if weather_row is None else weather_row.get("rainfall"),
            "track_flag": flag,
            "safety_state": safety,
        })
    result = pd.DataFrame(records)
    result["target_valid"] = (
        np.isfinite(result["lap_duration"])
        & result["lap_duration"].gt(0)
        & result["lap_number"].gt(1)
        & result["is_pit_out_lap"].eq(False)
        & ~result["is_pit_lap"]
    )
    return result.sort_values(["date_start", "driver_number", "lap_number"]).reset_index(drop=True)


def collect_contextual_session(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    session_key = int(session_key)
    endpoints = {
        "laps": client.get("laps", session_key=session_key),
        "stints": client.get("stints", session_key=session_key),
        "pit": client.get("pit", session_key=session_key),
        "weather": client.get("weather", session_key=session_key),
        "race_control": client.get("race_control", session_key=session_key),
        "drivers": client.get("drivers", session_key=session_key),
    }
    return build_contextual_laps(
        endpoints["laps"], endpoints["stints"], endpoints["pit"], endpoints["weather"],
        endpoints["race_control"], endpoints["drivers"])


def save_contextual_session(frame: pd.DataFrame, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
