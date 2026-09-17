"""Event-time replay of next-lap forecasts, NOT a tested live ingestion service.

Run: python -m f1_research.live_replay --output reports/local/replay
OpenF1 historical laps are fetched once with their exact bytes and SHA256 retained.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.openf1.org/v1/"
FEATURES = ["intercept", "last_minus_median", "previous_minus_median", "trend", "lap_number"]


def fetch(endpoint, params, cache, refresh=False):
    """Cache response bytes plus original URL/time; fail rather than invent observations."""
    url = requests.Request("GET", API + endpoint, params=params).prepare().url
    path = Path(cache) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".metadata.json")
    if path.exists() and metadata_path.exists() and not refresh:
        raw = path.read_bytes()
        metadata = json.loads(metadata_path.read_text())
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError("Cached source hash mismatch")
    else:
        for attempt in range(5):
            response = requests.get(url, timeout=90, headers={"User-Agent": "f1-race-research/0.1"})
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                break
            if attempt == 4:
                response.raise_for_status()
            time.sleep(min(2 ** (attempt + 1), 16))
        raw = response.content
        metadata = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        path.write_bytes(raw)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid source: {url}")
    return rows, metadata


def normalize_laps(rows, latency_s=1.0):
    """Availability is an explicit simulation assumption, never claimed as ingestion time."""
    if not np.isfinite(latency_s) or latency_s < 0:
        raise ValueError("latency_s must be finite and nonnegative")
    frame = pd.DataFrame(rows)
    required = {"session_key", "driver_number", "lap_number", "date_start", "lap_duration"}
    if required - set(frame):
        raise ValueError(f"Missing lap fields: {sorted(required - set(frame))}")
    for column in ("session_key", "driver_number", "lap_number"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not (np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"Invalid positive integer identity: {column}")
        frame[column] = values.astype("int64")
    if frame.duplicated(["session_key", "driver_number", "lap_number"]).any():
        raise ValueError("Duplicate lap identity")
    frame["start"] = pd.to_datetime(frame["date_start"], format="ISO8601", utc=True, errors="coerce")
    frame["duration"] = pd.to_numeric(frame["lap_duration"], errors="coerce")
    frame["valid"] = frame["start"].notna() & np.isfinite(frame["duration"]) & frame["duration"].gt(0)
    # No future-dependent pace, pit, safety-car or outlier exclusion: score every valid target.
    # Invalid labels must never create completion events (including infinity).
    frame["available"] = frame["start"] + pd.to_timedelta(frame["duration"].where(frame["valid"]), unit="s")
    frame["available"] += pd.to_timedelta(latency_s, unit="s")
    return frame


class OnlineRidge:
    """Exact cumulative ridge sufficient statistics, updated only after label availability."""

    def __init__(self, alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)
        self.xx = np.zeros((len(FEATURES), len(FEATURES)))
        self.xy = np.zeros(len(FEATURES))
        self.n = 0
        self.latest_label = None

    def update(self, x, residual, available):
        self.xx += np.outer(x, x)
        self.xy += x * residual
        self.n += 1
        self.latest_label = available if self.latest_label is None else max(self.latest_label, available)

    def predict(self, x):
        if not self.n:
            return 0.0
        # Fixed feature scales; no full-session scaler fit. Intercept is also regularized.
        coefficients = np.linalg.solve(self.xx + self.alpha * np.eye(len(FEATURES)), self.xy)
        return float(x @ coefficients)


def replay(frame, model, update=True):
    """Predict at lap start; finish events release only already completed labels.

    History resets between sessions. Fitted sufficient statistics persist. At exactly
    equal times, availability events precede forecast events, matching <= cutoff.
    Missing durations have no completion event; no guessed label is generated.
    """
    if frame["session_key"].nunique() != 1:
        raise ValueError("Replay one session at a time")
    # Unique positional indices make equal-time event ordering stable even after a
    # caller concatenates or shuffles dataframes with duplicate index labels.
    frame = frame.sort_values(["start", "driver_number", "lap_number"]).reset_index(drop=True)
    histories, pending, predictions = {}, {}, []
    events = []
    for index, row in frame.iterrows():
        if pd.notna(row["start"]):
            events.append((row["start"], 1, index))
        if row["valid"]:
            events.append((row["available"], 0, index))
    for timestamp, kind, index in sorted(events):
        row = frame.loc[index]
        driver = int(row["driver_number"])
        if kind == 0:
            histories.setdefault(driver, []).append((timestamp, float(row["duration"])))
            if index in pending and update:
                x, baseline = pending[index]
                model.update(x, float(row["duration"]) - baseline, timestamp)
            continue
        history = histories.get(driver, [])
        if len(history) < 3:
            continue
        values = np.array([item[1] for item in history[-5:]])
        baseline = float(np.median(values))
        x = np.array([1.0, (values[-1] - baseline) / 10,
                      (values[-2] - baseline) / 10, (values[-1] - values[0]) / 10,
                      float(row["lap_number"]) / 60])
        if model.latest_label is not None and model.latest_label > timestamp:
            raise ValueError("Training label crosses forecast cutoff")
        pending[index] = (x, baseline)
        predictions.append({"session_key": int(row["session_key"]), "driver_number": driver,
                            "lap_number": int(row["lap_number"]), "forecast_at": timestamp.isoformat(),
                            "feature_available_at": history[-1][0].isoformat(),
                            "max_training_label_at": model.latest_label.isoformat() if model.latest_label else None,
                            "training_labels": model.n, "ridge": baseline + model.predict(x),
                            "last_lap": float(values[-1]), "recent_median": baseline,
                            "actual": float(row["duration"]) if row["valid"] else None,
                            "target_available_at": row["available"].isoformat() if row["valid"] else None,
                            "features": x.tolist()})
    return pd.DataFrame(predictions)


def metrics(predictions):
    if predictions.empty:
        raise ValueError("No scored forecasts")
    scored = predictions.dropna(subset=["actual"])
    if scored.empty:
        raise ValueError("No scored forecasts")
    result = []
    for model in ("ridge", "last_lap", "recent_median"):
        if not np.isfinite(scored[model]).all():
            raise ValueError(f"Nonfinite forecasts: {model}")
        error = scored[model] - scored["actual"]
        result.append({"model": model, "n": len(scored), "mae_s": float(error.abs().mean()),
                       "rmse_s": float(np.sqrt(np.mean(error ** 2))),
                       "median_ae_s": float(error.abs().median()),
                       "p90_ae_s": float(error.abs().quantile(.9))})
    return result


def select_sessions(session_rows, race_count, latest_completed=False, now=None):
    """Reserve the final selected race without consulting any lap outcomes."""
    now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("Selection cutoff must be timezone aware")
    eligible = []
    for row in session_rows:
        if row.get("is_cancelled", False):
            continue
        if latest_completed:
            end = pd.to_datetime(row.get("date_end"), utc=True, errors="coerce")
            if pd.isna(end) or end >= now:
                continue
        eligible.append(row)
    eligible.sort(key=lambda row: pd.Timestamp(row["date_start"]))
    sessions = eligible[-race_count:] if latest_completed else eligible[:race_count]
    if len(sessions) != race_count:
        raise ValueError("Insufficient source sessions")
    return sessions


def run(output, year=2024, race_count=6, latency_s=1.0, alphas=(1, 10, 100, 1000),
        latest_completed=False):
    if race_count < 3:
        raise ValueError("At least one training, validation and test race required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session_rows, session_source = fetch("sessions", {"year": year, "session_name": "Race"},
                                         output / "cache", refresh=latest_completed)
    selection_cutoff = datetime.now(timezone.utc)
    sessions = select_sessions(session_rows, race_count, latest_completed, selection_cutoff)
    frames, sources = [], [session_source]
    for session in sessions:
        rows, source = fetch("laps", {"session_key": session["session_key"]}, output / "cache")
        frame = normalize_laps(rows, latency_s)
        if not frame["session_key"].eq(session["session_key"]).all():
            raise ValueError("Source returned laps from another session")
        frames.append(frame)
        sources.append(source)
        print(f"Loaded {session['country_name']}: {len(rows)} laps", flush=True)
    # Entire final two races reserved before tuning. No final-race metrics used for selection.
    trials = []
    for alpha in alphas:
        model = OnlineRidge(alpha)
        for frame in frames[:-2]:
            replay(frame, model)
        validation = replay(frames[-2], model)
        score = metrics(validation)[0]
        trials.append({"alpha": alpha, **score})
    selected = min(trials, key=lambda item: (item["mae_s"], item["alpha"]))["alpha"]
    model = OnlineRidge(selected)
    for frame in frames[:-1]:
        replay(frame, model)
    initial_labels = model.n
    test = replay(frames[-1], model)
    summary = metrics(test)
    test.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "tuning.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    forecast_times = pd.to_datetime(test.forecast_at, format="ISO8601", utc=True)
    feature_times = pd.to_datetime(test.feature_available_at, format="ISO8601", utc=True)
    label_times = pd.to_datetime(test.max_training_label_at, format="ISO8601", utc=True)
    target_times = pd.to_datetime(test.target_available_at, format="ISO8601", utc=True)
    checks = {
        "features_after_cutoff": int((feature_times > forecast_times).sum()),
        "training_labels_after_cutoff": int((label_times > forecast_times).sum()),
        "targets_available_before_forecast": int((target_times <= forecast_times).sum()),
    }
    if any(checks.values()):
        raise AssertionError(checks)
    report = {
        "schema_version": 1, "series": "f1", "task": "next_lap_duration_event_time_replay",
        "data_kind": "historical", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_documentation": "https://openf1.org/docs/", "sources": sources,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": {"latest_completed": latest_completed, "year": year,
                      "selection_cutoff_utc": selection_cutoff.isoformat(),
                      "rule": "last N noncancelled sessions with date_end < cutoff" if latest_completed
                      else "first N noncancelled sessions by date_start"},
        "train_sessions": sessions[:-2], "validation_session": sessions[-2], "test_session": sessions[-1],
        "selected_alpha": selected, "tuning": trials, "summary": summary,
        "initial_training_labels": initial_labels, "final_training_labels": model.n,
        "test_coverage": {"source_rows": len(frames[-1]), "forecast_rows": len(test),
                          "scored_rows": int(test.actual.notna().sum()),
                          "forecast_missing_target": int(test.actual.isna().sum()),
                          "no_forecast_warmup_or_missing_start": len(frames[-1]) - len(test)},
        "simulated_latency_s": latency_s, "cutoff_checks": checks,
        "rows": [{"session_key": s["session_key"], "raw": len(f), "valid_duration": int(f.valid.sum())}
                 for s, f in zip(sessions, frames)],
        "limitations": [
            "Historical event-time simulation, not verified live data ingestion or service latency.",
            "OpenF1 is an unofficial F1 source; provider documentation and API verified directly.",
            "date_start is approximate; available = start + duration + simulated latency, not publication timestamp.",
            "Free historical API; live access requires paid subscription per provider documentation.",
            "Next-lap duration only; no winner, pit strategy, DNF or full-race outcome prediction.",
            "Pit and safety-car laps retained in scoring; absent/invalid durations unscored, three observed laps required.",
            "Ridge updates after each observed training label, including earlier test-race labels; hyperparameters remain fixed.",
            "One reserved race is a development evaluation, not a sealed external benchmark; circuits and conditions shift.",
            "Start events replay historical lap identities; live start detection and corrections still require an ingestion adapter.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Historical next-lap replay", "", f"Test: {sessions[-1]['country_name']} {year}",
             f"Validation: {sessions[-2]['country_name']}; selected alpha: {selected}", "",
             "| Model | MAE seconds | RMSE seconds | Scored laps |", "|---|---:|---:|---:|"]
    lines += [f"| {m['model']} | {m['mae_s']:.3f} | {m['rmse_s']:.3f} | {m['n']} |" for m in summary]
    lines += ["", "## Limits", ""] + ["- " + item for item in report["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_alpha": selected, "summary": summary, "cutoff_checks": checks}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/local/replay"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--race-count", type=int, default=6)
    parser.add_argument("--latency-s", type=float, default=1.0)
    parser.add_argument("--latest-completed", action="store_true",
                        help="Select the most recent completed races within --year, excluding future/cancelled sessions")
    args = parser.parse_args()
    run(args.output, args.year, args.race_count, args.latency_s, latest_completed=args.latest_completed)


if __name__ == "__main__":
    main()
