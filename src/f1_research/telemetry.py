"""Retrospective timing/telemetry research, explicitly separate from pre-race features.

python -m f1_research.telemetry --year 2024 --round 1 --session Q
"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def clean_laps(laps):
    """Retain original rows with exclusion reasons; no silent replacement of missing data."""
    frame = pd.DataFrame(laps).copy()
    required = {"Driver", "LapNumber", "LapTime", "IsAccurate", "TrackStatus",
                "PitInTime", "PitOutTime", "Deleted"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing timing fields: {sorted(missing)}")
    frame["lap_time_s"] = pd.to_timedelta(frame["LapTime"]).dt.total_seconds()
    flags = {
        "missing_or_invalid_time": ~np.isfinite(frame["lap_time_s"]) | (frame["lap_time_s"] <= 0),
        "first_lap": frame["LapNumber"].le(1),
        "timing_inaccurate": ~frame["IsAccurate"].fillna(False).astype(bool),
        "non_green_track": frame["TrackStatus"].astype(str).ne("1"),
        "pit_in": frame["PitInTime"].notna(),
        "pit_out": frame["PitOutTime"].notna(),
        "deleted_or_unknown": frame["Deleted"].fillna(True).astype(bool),
    }
    frame["exclusion_reason"] = [
        ";".join(name for name, mask in flags.items() if bool(mask.iloc[i]))
        for i in range(len(frame))
    ]
    frame["clean"] = frame["exclusion_reason"].eq("")
    return frame


def summarize_samples(samples, max_gap_s=1.0):
    """Time-weighted summaries over observed intervals; never bridge a long dropout.

    Left-sample hold is explicit: brake is a boolean broadcast channel, not pressure.
    max_gap_s is a quality threshold, not an interpolation frequency.
    """
    required = {"time_s", "speed_kmh", "throttle_pct", "brake"}
    if required - set(samples):
        raise ValueError(f"Missing telemetry fields: {sorted(required - set(samples))}")
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    frame = samples.copy().sort_values("time_s", kind="stable")
    if frame["time_s"].duplicated().any():
        raise ValueError("Duplicate telemetry timestamps")
    if len(frame) < 2:
        raise ValueError("At least two telemetry samples required")
    time_s = pd.to_numeric(frame["time_s"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(time_s).all():
        raise ValueError("Nonfinite telemetry timestamps")
    delta = np.diff(time_s)
    valid_interval = (delta > 0) & (delta <= max_gap_s)
    total_duration = float(time_s[-1] - time_s[0])
    if not valid_interval.any() or total_duration <= 0:
        raise ValueError("No usable telemetry intervals")
    result = {"samples": len(frame), "span_s": total_duration,
              "observed_interval_s": float(delta[valid_interval].sum()),
              "interval_coverage": float(delta[valid_interval].sum() / total_duration),
              "gaps_excluded": int((~valid_interval).sum()),
              "weighting": "left_sample_hold; intervals exceeding max_gap_s excluded",
              "max_gap_s": max_gap_s}
    for column, output, bounds in [
        ("speed_kmh", "mean_speed_kmh", (0, 450)),
        ("throttle_pct", "mean_throttle_pct", (0, 100)),
        ("brake", "braking_time_fraction", (0, 1)),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        quality = np.isfinite(values) & (values >= bounds[0]) & (values <= bounds[1])
        if column == "brake":
            quality &= np.isin(values, [0, 1])
        mask = valid_interval & quality[:-1] & quality[1:]
        duration = float(delta[mask].sum())
        result[output] = float(np.average(values[:-1][mask], weights=delta[mask])) if duration else None
        result[column + "_coverage"] = duration / total_duration
        if column == "throttle_pct":
            result["full_throttle_time_fraction"] = (
                float(np.average((values[:-1][mask] >= 98).astype(float), weights=delta[mask]))
                if duration else None
            )
    return result


def pace_summary(laps):
    """Describe observed pace; slope is confounded by fuel, traffic and track evolution."""
    clean = laps[laps["clean"]].copy()
    records = []
    for driver, group in clean.groupby("Driver"):
        times = group["lap_time_s"]
        records.append({"driver": driver, "clean_laps": len(group),
                        "median_lap_s": float(times.median()),
                        "iqr_lap_s": float(times.quantile(.75) - times.quantile(.25)),
                        "fastest_clean_lap_s": float(times.min())})
    stints = []
    if {"Stint", "Compound"} <= set(clean):
        for (driver, stint, compound), group in clean.groupby(["Driver", "Stint", "Compound"]):
            if len(group) < 5:
                continue
            x = group["LapNumber"].to_numpy(dtype=float)
            y = group["lap_time_s"].to_numpy(dtype=float)
            slopes = [(y[j] - y[i]) / (x[j] - x[i])
                      for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
            if slopes:
                stints.append({"driver": driver, "stint": float(stint), "compound": compound,
                               "laps": len(group), "observed_pace_slope_s_per_lap": float(np.median(slopes))})
    return records, stints


def collect_session(year, round_number, session_code, cache, output, drivers=None):
    import fastf1  # Optional dependency; import core modeling without FastF1.

    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(year, round_number, session_code)
    session.load(telemetry=True, weather=True, messages=True)
    try:
        source_laps = session.laps
    except fastf1.exceptions.DataNotLoadedError as exc:
        raise ValueError("FastF1 timing unavailable. Use --provider openf1 with a historical "
                         "--session-key, or retry after source recovery.") from exc
    if source_laps.empty:
        raise ValueError("Source returned no laps")
    laps = clean_laps(session.laps)
    columns = [c for c in ["Driver", "LapNumber", "Stint", "Compound", "TyreLife",
                           "lap_time_s", "clean", "exclusion_reason"] if c in laps]
    laps[columns].to_csv(output / "laps.csv", index=False)
    pace, stints = pace_summary(laps)
    selected = list(drivers or sorted(laps["Driver"].dropna().unique()))
    telemetry, unavailable = [], []
    for driver in selected:
        eligible = laps[laps["Driver"].eq(driver) & laps["clean"]]
        if eligible.empty:
            unavailable.append({"driver": driver, "reason": "No eligible clean lap"})
            continue
        best = eligible.loc[eligible["lap_time_s"].idxmin()]
        original = session.laps[
            session.laps["Driver"].eq(driver) & session.laps["LapNumber"].eq(best["LapNumber"])
        ].iloc[0]
        try:
            car = original.get_car_data().add_distance()
            table = pd.DataFrame({
                "time_s": car["Time"].dt.total_seconds(),
                "distance_m": car["Distance"], "speed_kmh": car["Speed"],
                "throttle_pct": car["Throttle"], "brake": car["Brake"].astype(float),
                "rpm": car["RPM"], "gear": car["nGear"], "drs": car["DRS"],
            })
            summary = summarize_samples(table)
            file = f"telemetry_{driver}.csv"
            table.to_csv(output / file, index=False)
            telemetry.append({"driver": driver, "lap": int(best["LapNumber"]),
                              "lap_time_s": float(best["lap_time_s"]), "file": file,
                              "trace": json.loads(table.to_json(orient="records")), **summary})
        except (ValueError, KeyError, IndexError) as exc:
            unavailable.append({"driver": driver, "reason": str(exc)})
    manifest = {
        "schema_version": 1, "series": "f1", "data_kind": "historical",
        "analysis_kind": "retrospective_session_analysis", "year": year,
        "round": round_number, "session": session_code, "event": str(session.event["EventName"]),
        "source": "FastF1", "source_url": "https://github.com/theOehrly/Fast-F1",
        "provider_version": fastf1.__version__, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "laps": len(laps), "clean_laps": int(laps["clean"].sum()),
        "pace": pace, "stints": stints, "telemetry": telemetry, "unavailable": unavailable,
        "limitations": [
            "Retrospective analysis; target-race observations are never pre-race forecast features.",
            "Broadcast telemetry is not full team sensor data. Brake is boolean, not pressure.",
            "Derived distance integrates sampled speed; it is not independent GPS measurement.",
            "Pace slope is not identified tyre degradation: fuel, traffic and track evolution confound it.",
            "First, pit, deleted, inaccurate and non-green laps are excluded; remaining laps may still contain traffic.",
        ],
    }
    manifest["file_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in output.glob("*.csv")}
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def collect_openf1(session_key, driver_numbers, cache, output):
    """Explicit alternative provider, never claim equivalent lap-validity coverage."""
    from urllib.parse import urlencode

    from .data import JsonCache

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    client = JsonCache(Path(cache) / "openf1")

    def fetch(endpoint, **params):
        return client.get("https://api.openf1.org/v1/" + endpoint + "?" + urlencode(params))

    sessions = fetch("sessions", session_key=session_key)
    if len(sessions) != 1:
        raise ValueError("Expected one historical session")
    session = sessions[0]
    if pd.Timestamp(session["date_end"]) >= pd.Timestamp.now(tz="UTC"):
        raise ValueError("This collector supports completed historical sessions only")
    entries = {int(d["driver_number"]): d for d in fetch("drivers", session_key=session_key)}
    telemetry, unavailable, lap_rows, pace = [], [], [], []
    for number in driver_numbers:
        if number not in entries:
            raise ValueError(f"Driver {number} not in selected session")
        driver = entries[number]["name_acronym"]
        all_laps = fetch("laps", session_key=session_key, driver_number=number)
        eligible = []
        for lap in all_laps:
            duration = lap.get("lap_duration")
            good = (duration is not None and np.isfinite(duration) and duration > 0
                    and bool(lap.get("date_start")) and not lap.get("is_pit_out_lap", True))
            lap_rows.append({"Driver": driver, "LapNumber": lap["lap_number"],
                             "lap_time_s": duration, "eligible_timed_lap": good,
                             "validity": "not_verified_by_this_provider"})
            if good:
                eligible.append(lap)
        if not eligible:
            unavailable.append({"driver": driver, "reason": "No complete non-pit-out timed lap"})
            continue
        durations = [r["lap_duration"] for r in eligible]
        pace.append({"driver": driver, "eligible_laps": len(eligible),
                     "median_lap_s": float(np.median(durations)),
                     "iqr_lap_s": float(np.quantile(durations, .75) - np.quantile(durations, .25)),
                     "fastest_timed_lap_s": float(min(durations))})
        lap = min(eligible, key=lambda row: row["lap_duration"])
        start = pd.Timestamp(lap["date_start"])
        end = pd.Timestamp(start.to_pydatetime() + timedelta(seconds=float(lap["lap_duration"])))
        raw = fetch("car_data", session_key=session_key, driver_number=number,
                    **{"date>": start.isoformat(), "date<": end.isoformat()})
        if len(raw) < 2:
            unavailable.append({"driver": driver, "reason": "Fewer than two telemetry samples"})
            continue
        car = pd.DataFrame(raw).sort_values("date")
        t = (pd.to_datetime(car["date"], utc=True) - start).dt.total_seconds().to_numpy()
        speed = car["speed"].to_numpy(dtype=float)
        dt = np.diff(t)
        # Reject long gaps for a continuous distance plot rather than invent missing distance.
        if np.any(dt <= 0) or np.any(dt > 1):
            unavailable.append({"driver": driver, "reason": "Duplicate times or >1s dropout in lap trace"})
            continue
        distance = np.r_[0.0, np.cumsum((speed[:-1] + speed[1:]) / 2 / 3.6 * dt)]
        table = pd.DataFrame({"time_s": t, "distance_m": distance, "speed_kmh": speed,
                              "throttle_pct": car["throttle"].to_numpy(),
                              "brake": car["brake"].map({0: 0, 100: 1}).to_numpy(),
                              "rpm": car["rpm"].to_numpy(), "gear": car["n_gear"].to_numpy(),
                              "drs": car["drs"].to_numpy()})
        summary = summarize_samples(table)
        filename = f"telemetry_{driver}.csv"
        table.to_csv(output / filename, index=False)
        telemetry.append({"driver": driver, "lap": int(lap["lap_number"]),
                          "lap_time_s": float(lap["lap_duration"]), "file": filename,
                          "lap_interval_coverage": summary["observed_interval_s"] / lap["lap_duration"],
                          "trace": json.loads(table.to_json(orient="records")), **summary})
    pd.DataFrame(lap_rows).to_csv(output / "laps.csv", index=False)
    manifest = {
        "schema_version": 1, "series": "f1", "data_kind": "historical",
        "analysis_kind": "retrospective_session_analysis", "year": session["year"],
        "session": session["session_name"], "session_key": session_key,
        "event": session["location"], "source": "OpenF1", "source_url": "https://openf1.org/docs/",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "laps": len(lap_rows),
        "clean_laps": None, "eligible_laps": sum(x["eligible_timed_lap"] for x in lap_rows),
        "quality_scope": "Timed non-pit-out laps only; deleted, track-status and pit-in flags not verified",
        "pace": pace, "stints": [], "telemetry": telemetry, "unavailable": unavailable,
        "provenance": client.provenance,
        "limitations": [
            "Retrospective observed session, independent of pre-race forecasting benchmarks.",
            "Fastest timed lap selection is not verified for deletion, track status or pit-in.",
            "Brake is boolean (mapped from source 0/100); no brake pressure or steering measurement.",
            "Distance is a speed integral starting at the first sample, not GPS or a precision lap delta.",
            "Low-frequency broadcast data and missing lap boundaries limit comparison accuracy.",
        ],
    }
    manifest["file_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in output.glob("*.csv")}
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["fastf1", "openf1"], default="fastf1")
    parser.add_argument("--year", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--session-key", type=int)
    parser.add_argument("--driver-numbers", type=int, nargs="+")
    parser.add_argument("--session", choices=["FP1", "FP2", "FP3", "Q", "R", "S", "SQ"], default="Q")
    parser.add_argument("--drivers", nargs="+")
    parser.add_argument("--cache", default="data/fastf1")
    parser.add_argument("--output", default="reports/local/telemetry")
    args = parser.parse_args()
    if args.provider == "openf1":
        if not args.session_key or not args.driver_numbers:
            parser.error("OpenF1 requires --session-key and --driver-numbers")
        result = collect_openf1(args.session_key, args.driver_numbers, args.cache, args.output)
    else:
        if not args.year or not args.round:
            parser.error("FastF1 requires --year and --round")
        result = collect_session(args.year, args.round, args.session, args.cache, args.output, args.drivers)
    print(f"{result['event']}: {result.get('eligible_laps', result['clean_laps'])}/{result['laps']} eligible laps; "
          f"{len(result['telemetry'])} driver traces. Saved {args.output}/summary.json")


if __name__ == "__main__":
    main()
