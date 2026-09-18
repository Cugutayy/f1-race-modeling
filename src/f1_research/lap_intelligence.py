"""Event-time next-lap dataset, modern challengers and live inference.

Historical OpenF1 joins are designed to avoid target leakage from completed lap
labels. Stint metadata has no publication timestamp in the historical REST schema,
so compound/tyre-age features are explicitly tagged retrospective unless they came
from a captured live stream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .value_parsing import strict_optional_bool

NUMERIC_FEATURES = [
    "lap_number",
    "last_lap_s",
    "recent_median_3_s",
    "recent_median_5_s",
    "recent_trend_s_per_lap",
    "recent_variability_s",
    "tyre_age",
    "stint_number",
    "pit_stops_before",
    "air_temperature_c",
    "track_temperature_c",
    "humidity_pct",
    "rainfall",
    "safety_car_active",
    "yellow_recent",
]
CATEGORICAL_FEATURES = ["compound", "driver_number"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class LapModelSpec:
    name: str
    params: dict[str, Any]


@dataclass
class LapBenchmarkResult:
    model: str
    session_key: int
    n: int
    mae_s: float
    rmse_s: float
    median_ae_s: float
    p90_ae_s: float


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _times(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], format="ISO8601", utc=True, errors="coerce")


def _stint_features(laps: pd.DataFrame, stint_rows: list[dict[str, Any]]) -> pd.DataFrame:
    stints = pd.DataFrame(stint_rows)
    laps = laps.copy()
    laps["compound"] = "UNKNOWN"
    laps["stint_number"] = np.nan
    laps["tyre_age"] = np.nan
    if stints.empty:
        return laps
    for column in ("driver_number", "lap_start", "lap_end", "stint_number", "tyre_age_at_start"):
        stints[column] = _numeric(stints, column)
    for _, stint in stints.dropna(subset=["driver_number", "lap_start"]).iterrows():
        driver = int(stint.driver_number)
        start = int(stint.lap_start)
        end = int(stint.lap_end) if np.isfinite(stint.lap_end) else int(laps.lap_number.max())
        mask = laps.driver_number.eq(driver) & laps.lap_number.between(start, end)
        laps.loc[mask, "compound"] = str(stint.get("compound") or "UNKNOWN").upper()
        laps.loc[mask, "stint_number"] = stint.stint_number
        age = stint.tyre_age_at_start
        if np.isfinite(age):
            laps.loc[mask, "tyre_age"] = age + laps.loc[mask, "lap_number"] - start
    return laps


def _asof_weather(laps: pd.DataFrame, weather_rows: list[dict[str, Any]]) -> pd.DataFrame:
    weather = pd.DataFrame(weather_rows)
    output = laps.sort_values("start").copy()
    if weather.empty:
        for column in ("air_temperature_c", "track_temperature_c", "humidity_pct", "rainfall"):
            output[column] = np.nan
        output["weather_available_at"] = pd.NaT
        return output
    weather["weather_time"] = _times(weather, "date")
    weather = weather.dropna(subset=["weather_time"]).sort_values("weather_time")
    weather["air_temperature_c"] = _numeric(weather, "air_temperature")
    weather["track_temperature_c"] = _numeric(weather, "track_temperature")
    weather["humidity_pct"] = _numeric(weather, "humidity")
    weather["rainfall"] = _numeric(weather, "rainfall")
    keep = ["weather_time", "air_temperature_c", "track_temperature_c", "humidity_pct", "rainfall"]
    merged = pd.merge_asof(output, weather[keep], left_on="start", right_on="weather_time", direction="backward")
    return merged.rename(columns={"weather_time": "weather_available_at"})


def _race_control_features(laps: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    control = pd.DataFrame(rows)
    output = laps.copy()
    output["safety_car_active"] = 0.0
    output["yellow_recent"] = 0.0
    output["race_control_available_at"] = pd.NaT
    if control.empty:
        return output
    control["control_time"] = _times(control, "date")
    control = control.dropna(subset=["control_time"]).sort_values("control_time")
    categories = control.get("category", pd.Series("", index=control.index)).astype(str)
    messages = control.get("message", pd.Series("", index=control.index)).astype(str).str.upper()
    flags = control.get("flag", pd.Series("", index=control.index)).astype(str).str.upper()
    sc_events = control[categories.eq("SafetyCar") | messages.str.contains("SAFETY CAR|VSC", regex=True)]
    yellow = control[flags.str.contains("YELLOW", regex=False)]
    for idx, start in output["start"].items():
        if pd.isna(start):
            continue
        past = control[control.control_time <= start]
        if not past.empty:
            output.at[idx, "race_control_available_at"] = past.control_time.iloc[-1]
        recent_window = start - pd.Timedelta(minutes=8)
        if not yellow[(yellow.control_time <= start) & (yellow.control_time >= recent_window)].empty:
            output.at[idx, "yellow_recent"] = 1.0
        recent_sc = sc_events[(sc_events.control_time <= start) & (sc_events.control_time >= recent_window)]
        if not recent_sc.empty:
            last_message = str(recent_sc.iloc[-1].get("message") or "").upper()
            ending = any(token in last_message for token in ("ENDING", "IN THIS LAP", "WITHDRAWN"))
            output.at[idx, "safety_car_active"] = 0.0 if ending else 1.0
    return output


def build_lap_dataset(lap_rows: list[dict[str, Any]], *,
                      stint_rows: list[dict[str, Any]] | None = None,
                      weather_rows: list[dict[str, Any]] | None = None,
                      pit_rows: list[dict[str, Any]] | None = None,
                      race_control_rows: list[dict[str, Any]] | None = None,
                      latency_s: float = 1.0,
                      minimum_history: int = 3) -> pd.DataFrame:
    """Build one row per forecast-at-lap-start observation.

    The target lap duration is never part of its own features. Previous lap labels
    become usable only after start + duration + simulated latency. Historical stint
    joins are retrospective because OpenF1 REST does not expose their publication time.
    """
    if not np.isfinite(latency_s) or latency_s < 0:
        raise ValueError("latency_s must be finite and nonnegative")
    laps = pd.DataFrame(lap_rows)
    required = {
        "session_key", "driver_number", "lap_number", "date_start",
        "lap_duration", "is_pit_out_lap",
    }
    if required - set(laps):
        raise ValueError(f"Missing lap fields: {sorted(required - set(laps))}")
    for column in ("session_key", "driver_number", "lap_number"):
        laps[column] = _numeric(laps, column)
        if not (laps[column].notna() & laps[column].gt(0) & laps[column].mod(1).eq(0)).all():
            raise ValueError(f"Invalid identity column: {column}")
        laps[column] = laps[column].astype(int)
    if laps.duplicated(["session_key", "driver_number", "lap_number"]).any():
        raise ValueError("Duplicate lap identity")
    laps["start"] = _times(laps, "date_start")
    laps["target_s"] = _numeric(laps, "lap_duration")
    laps["target_valid"] = laps.start.notna() & np.isfinite(laps.target_s) & laps.target_s.gt(0)
    laps["target_available_at"] = laps.start + pd.to_timedelta(laps.target_s.where(laps.target_valid), unit="s")
    laps["target_available_at"] += pd.to_timedelta(latency_s, unit="s")
    laps["is_pit_out_lap"] = pd.array(
        [
            strict_optional_bool(value, field="openf1.laps.is_pit_out_lap")
            for value in laps["is_pit_out_lap"]
        ],
        dtype="boolean",
    )
    # Unknown pit-out state is not a trustworthy target regime.
    laps["target_valid"] &= laps["is_pit_out_lap"].notna()
    laps = _stint_features(laps, stint_rows or [])
    laps = _asof_weather(laps, weather_rows or [])
    laps = _race_control_features(laps, race_control_rows or [])

    pits = pd.DataFrame(pit_rows or [])
    if not pits.empty:
        pits["pit_time"] = _times(pits, "date")
        pits["driver_number"] = _numeric(pits, "driver_number")
    rows: list[dict[str, Any]] = []
    for driver, group in laps.sort_values(["start", "driver_number", "lap_number"]).groupby("driver_number"):
        completed: list[tuple[pd.Timestamp, float]] = []
        for _, lap in group.sort_values("start").iterrows():
            start = lap.start
            if pd.isna(start):
                continue
            usable = [(available, duration) for available, duration in completed if available <= start]
            values = np.asarray([duration for _, duration in usable[-5:]], dtype=float)
            if len(values) >= minimum_history:
                last = float(values[-1])
                recent3 = values[-3:]
                recent5 = values[-5:]
                trend = float((recent3[-1] - recent3[0]) / max(1, len(recent3) - 1))
                variability = float(np.std(recent5, ddof=1)) if len(recent5) > 1 else 0.0
                pit_count = 0
                if not pits.empty:
                    pit_count = int(((pits.driver_number == driver) & (pits.pit_time <= start)).sum())
                source_times = [available for available, _ in usable[-5:]]
                for field in ("weather_available_at", "race_control_available_at"):
                    value = lap.get(field)
                    if pd.notna(value):
                        source_times.append(value)
                rows.append({
                    "session_key": int(lap.session_key), "driver_number": str(int(driver)),
                    "lap_number": int(lap.lap_number), "forecast_at": start,
                    "feature_available_at": max(source_times) if source_times else start,
                    "target_available_at": lap.target_available_at,
                    "target_s": float(lap.target_s) if lap.target_valid else np.nan,
                    "target_valid": bool(lap.target_valid),
                    "is_pit_out_lap": (
                        None if pd.isna(lap.is_pit_out_lap) else bool(lap.is_pit_out_lap)
                    ),
                    "last_lap_s": last, "recent_median_3_s": float(np.median(recent3)),
                    "recent_median_5_s": float(np.median(recent5)),
                    "recent_trend_s_per_lap": trend, "recent_variability_s": variability,
                    "tyre_age": lap.tyre_age, "stint_number": lap.stint_number,
                    "pit_stops_before": pit_count, "compound": str(lap.compound or "UNKNOWN"),
                    "air_temperature_c": lap.air_temperature_c,
                    "track_temperature_c": lap.track_temperature_c,
                    "humidity_pct": lap.humidity_pct, "rainfall": lap.rainfall,
                    "safety_car_active": lap.safety_car_active, "yellow_recent": lap.yellow_recent,
                    "stint_feature_provenance": "historical_rest_no_publication_timestamp",
                })
            if lap.target_valid:
                completed.append((lap.target_available_at, float(lap.target_s)))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if (pd.to_datetime(result.feature_available_at, utc=True) > pd.to_datetime(result.forecast_at, utc=True)).any():
        raise AssertionError("A next-lap feature crosses its forecast cutoff")
    return result.sort_values(["forecast_at", "driver_number", "lap_number"]).reset_index(drop=True)


def _preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_lap_estimator(spec: LapModelSpec) -> Pipeline:
    params = dict(spec.params)
    if spec.name == "hist_gradient_boosting":
        defaults = {"loss": "absolute_error", "max_iter": 180, "learning_rate": 0.04,
                    "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 4.0,
                    "random_state": 42}
        defaults.update(params)
        model = HistGradientBoostingRegressor(**defaults)
    elif spec.name == "extra_trees":
        defaults = {"n_estimators": 500, "min_samples_leaf": 4, "max_features": 0.8,
                    "n_jobs": -1, "random_state": 42}
        defaults.update(params)
        model = ExtraTreesRegressor(**defaults)
    elif spec.name == "tabicl_v2":
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise RuntimeError("Install the foundation extra to evaluate TabICLv2") from exc
        model = TabICLRegressor(**params)
    else:
        raise ValueError(f"Unknown lap model: {spec.name}")
    return Pipeline([("preprocess", _preprocessor()), ("model", model)])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        raise ValueError("No finite next-lap predictions")
    error = predicted[mask] - actual[mask]
    absolute = np.abs(error)
    return {"n": int(mask.sum()), "mae_s": float(absolute.mean()),
            "rmse_s": float(np.sqrt(np.mean(error ** 2))),
            "median_ae_s": float(np.median(absolute)), "p90_ae_s": float(np.quantile(absolute, 0.9))}


def benchmark_lap_models(datasets: list[pd.DataFrame], *,
                         specs: tuple[LapModelSpec, ...] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune on the penultimate whole session and score a sealed final session."""
    if len(datasets) < 3:
        raise ValueError("Need at least train, validation and test sessions")
    specs = specs or (
        LapModelSpec("hist_gradient_boosting", {}),
        LapModelSpec("extra_trees", {}),
    )
    train = pd.concat(datasets[:-2], ignore_index=True)
    validation = datasets[-2].copy()
    test = datasets[-1].copy()
    train = train[train.target_valid]
    validation = validation[validation.target_valid]
    test = test[test.target_valid]
    if train.empty or validation.empty or test.empty:
        raise ValueError("One benchmark block has no valid lap targets")

    trials = []
    fitted = {}
    for spec in specs:
        try:
            model = build_lap_estimator(spec).fit(train[FEATURES], train.target_s)
            pred = model.predict(validation[FEATURES])
            score = _metrics(validation.target_s.to_numpy(), pred)
            trials.append({"model": spec.name, "params": json.dumps(spec.params, sort_keys=True), **score})
            fitted[spec.name] = spec
        except (ValueError, RuntimeError, MemoryError) as exc:
            trials.append({"model": spec.name, "params": json.dumps(spec.params, sort_keys=True),
                           "n": 0, "mae_s": np.inf, "rmse_s": np.inf,
                           "median_ae_s": np.inf, "p90_ae_s": np.inf, "error": str(exc)})
    valid = [row for row in trials if np.isfinite(row["mae_s"])]
    if not valid:
        raise ValueError("No next-lap challenger completed validation")
    selected_name = min(valid, key=lambda row: (row["mae_s"], row["model"]))["model"]
    selected_spec = fitted[selected_name]
    final_train = pd.concat([train, validation], ignore_index=True)
    final_model = build_lap_estimator(selected_spec).fit(final_train[FEATURES], final_train.target_s)
    model_prediction = final_model.predict(test[FEATURES])
    baselines = {
        selected_name: model_prediction,
        "recent_median_5": test.recent_median_5_s.to_numpy(dtype=float),
        "last_lap": test.last_lap_s.to_numpy(dtype=float),
    }
    results = []
    session_key = int(test.session_key.iloc[0])
    for name, prediction in baselines.items():
        score = _metrics(test.target_s.to_numpy(), prediction)
        results.append(asdict(LapBenchmarkResult(name, session_key, **score)))
    audit = {
        "protocol": "whole-session train -> validation model selection -> sealed test",
        "selected_model": selected_name,
        "validation_trials": [{key: (None if value == np.inf else value) for key, value in row.items()}
                              for row in trials],
        "features": FEATURES,
        "stint_provenance": "historical REST lacks publication timestamp; not strict as-published proof",
        "test_updates_model": False,
    }
    return pd.DataFrame(results), audit


def _safety_car_active(value: Any) -> float:
    if not value:
        return 0.0
    message = str(value).upper()
    if any(token in message for token in ("ENDING", "IN THIS LAP", "WITHDRAWN", "ENDED")):
        return 0.0
    return 1.0


def live_feature_rows(snapshot: dict[str, Any]) -> pd.DataFrame:
    """Build inference rows from the canonical live state using the training schema."""
    weather = snapshot.get("weather") or {}
    rows = []
    for driver in snapshot.get("drivers", []):
        history = np.asarray(driver.get("recent_laps_s") or [], dtype=float)
        history = history[np.isfinite(history)]
        if len(history) < 3:
            continue
        recent3, recent5 = history[-3:], history[-5:]
        rows.append({
            "driver_number": str(int(driver["driver_number"])),
            "lap_number": int((driver.get("lap_number") or snapshot.get("current_lap") or 0) + 1),
            "last_lap_s": float(history[-1]), "recent_median_3_s": float(np.median(recent3)),
            "recent_median_5_s": float(np.median(recent5)),
            "recent_trend_s_per_lap": float((recent3[-1] - recent3[0]) / max(1, len(recent3) - 1)),
            "recent_variability_s": float(np.std(recent5, ddof=1)) if len(recent5) > 1 else 0.0,
            "tyre_age": driver.get("tyre_age"), "stint_number": driver.get("stint_number"),
            "pit_stops_before": driver.get("pit_stops", 0),
            "compound": str(driver.get("compound") or "UNKNOWN").upper(),
            "air_temperature_c": weather.get("air_temperature_c"),
            "track_temperature_c": weather.get("track_temperature_c"),
            "humidity_pct": weather.get("humidity_pct"),
            "rainfall": float(bool(weather.get("rainfall"))) if weather.get("rainfall") is not None else np.nan,
            "safety_car_active": _safety_car_active(snapshot.get("safety_car")),
            "yellow_recent": float("YELLOW" in str(snapshot.get("flag") or "").upper()),
        })
    return pd.DataFrame(rows)


def save_benchmark(metrics: pd.DataFrame, audit: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "summary.csv", index=False)
    (output / "report.json").write_text(json.dumps({"summary": metrics.to_dict("records"), "audit": audit},
                                                   indent=2, allow_nan=False), encoding="utf-8")
