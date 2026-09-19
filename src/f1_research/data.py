"""Cached, paginated Jolpica ingestion with source and content provenance."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class JsonCache:
    def __init__(self, directory, offline=False, delay=0.65):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.delay = delay
        self.provenance = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "CugutayyRaceResearch/0.3.0"
        retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        file = self.directory / f"{key}.json"
        if file.exists():
            envelope = json.loads(file.read_text(encoding="utf-8"))
            body = json.dumps(envelope["data"], sort_keys=True).encode()
            if hashlib.sha256(body).hexdigest() != envelope["sha256"]:
                raise ValueError(f"Cache checksum mismatch: {file}")
            self.provenance.append({k: envelope[k] for k in ("url", "retrieved_at", "sha256")})
            return envelope["data"]
        if self.offline:
            raise FileNotFoundError(f"Not cached: {url}")
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        payload = response.json()
        envelope = {
            "url": url, "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "data": payload,
        }
        temp = file.with_suffix(".tmp")
        temp.write_text(json.dumps(envelope), encoding="utf-8")
        temp.replace(file)
        self.provenance.append({k: envelope[k] for k in ("url", "retrieved_at", "sha256")})
        time.sleep(self.delay)
        return payload


def races(client, year, endpoint):
    """Pagination counts driver results, so a page may split a race."""
    combined = {}
    offset = 0
    field = "Results" if endpoint == "results" else "QualifyingResults"
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/{endpoint}/?limit=100&offset={offset}"
        root = client.get(url)["MRData"]
        page = root["RaceTable"]["Races"]
        count = 0
        for race in page:
            key = int(race["round"])
            count += len(race[field])
            if key not in combined:
                combined[key] = {**race, field: []}
            combined[key][field].extend(race[field])
        offset += count
        if offset >= int(root["total"]):
            break
        if not count:
            raise ValueError("Pagination stopped before advertised total")
    return combined


def seconds(value):
    if not value:
        return np.nan
    parts = str(value).replace("'", ":").split(":")
    try:
        result = 0.0
        for part in parts:
            result = result * 60 + float(part)
        return result if result > 0 else np.nan
    except ValueError:
        return np.nan


def collect(years, cache, offline=False):
    client = JsonCache(cache, offline=offline)
    rows = []
    for year in sorted(set(years)):
        result_map = races(client, year, "results")
        quali_map = races(client, year, "qualifying")
        for rnd, race in sorted(result_map.items()):
            qualifying = {
                r["Driver"]["driverId"]: r
                for r in quali_map.get(rnd, {}).get("QualifyingResults", [])
            }
            for result in race["Results"]:
                driver = result["Driver"]["driverId"]
                q = qualifying.get(driver, {})
                rows.append({
                    "event_id": f"{year}-{rnd:02d}", "year": year, "round": rnd,
                    "date": race["date"], "circuit": race["Circuit"]["circuitId"],
                    "driver": driver, "team": result["Constructor"]["constructorId"],
                    "quali_position": float(q.get("position", np.nan)),
                    # Compare only Q1, never mix evolving Q1/Q2/Q3 conditions.
                    "quali_seconds": seconds(q.get("Q1")),
                    "quali_time_basis": "Q1",
                    "grid_position": float(result["grid"]),
                    "finish_position": float(result["position"]),
                    "points": float(result["points"]),
                    "status": str(result["status"]),
                    "starter_count": len(race["Results"]),
                    "dnf": int(not (result["status"] == "Finished"
                                    or result["status"].startswith("+"))),
                })
    if not rows:
        raise ValueError("No completed races returned")
    result = validate(pd.DataFrame(rows))
    result.attrs["provenance"] = {"provider": "Jolpica", "requests": client.provenance,
                                  "years": sorted(set(years)), "qualifying_time_basis": "Q1",
                                  "publication_timestamps_available": False}
    return result


def validate(frame):
    required = {"event_id", "date", "year", "round", "driver", "team", "circuit",
                "quali_position", "quali_seconds", "finish_position", "points", "dnf"}
    if required - set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    df = frame.copy()
    for col in ["event_id", "driver", "team", "circuit"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifiers: {col}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    if df["date"].isna().any() or df.duplicated(["event_id", "driver"]).any():
        raise ValueError("Null dates or duplicate driver/event rows")
    for col in ["finish_position", "points", "dnf"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Nonfinite target: {col}")
    if not df["dnf"].isin([0, 1]).all() or (df["finish_position"] < 1).any():
        raise ValueError("Invalid outcome range")
    for _, group in df.groupby("event_id"):
        if group["date"].nunique() != 1 or group["finish_position"].eq(1).sum() != 1:
            raise ValueError("Each event needs one date and one winner")
        if group["finish_position"].duplicated().any():
            raise ValueError("Duplicate classified positions")
        if len(group) < 2 or sorted(group["finish_position"]) != list(range(1, len(group) + 1)):
            raise ValueError("Each event must contain a complete consecutive classification")
        if any(group[c].nunique() != 1 for c in ("year", "round", "circuit")):
            raise ValueError("Inconsistent event metadata")
    for col in ["quali_position", "quali_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col]).any() or (df[col].dropna() <= 0).any():
            raise ValueError(f"Invalid qualifying feature: {col}")
    if (df["points"] < 0).any():
        raise ValueError("Negative points")
    if "starter_count" in df:
        df["starter_count"] = pd.to_numeric(df["starter_count"], errors="raise")
        if (
            ~np.isfinite(df["starter_count"])
            | (df["starter_count"] < 2)
            | ~df["starter_count"].mod(1).eq(0)
        ).any():
            raise ValueError("Invalid starter_count")
        for event_id, group in df.groupby("event_id"):
            expected = group["starter_count"].unique()
            if len(expected) != 1 or len(group) != int(expected[0]):
                raise ValueError(
                    f"Survivorship guard failed for {event_id}: "
                    f"retained {len(group)} of {expected.tolist()} source starters"
                )
    if "status" in df and (
        df["status"].isna().any() | df["status"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Missing result status")
    return df.sort_values(["date", "event_id", "driver"]).reset_index(drop=True)


def survivorship_audit(frame: pd.DataFrame) -> dict:
    """Describe whether all source result entrants survived downstream filtering."""
    df = validate(frame)
    verified = "starter_count" in df
    event_rows = []
    for event_id, group in df.groupby("event_id", sort=True):
        expected = int(group["starter_count"].iloc[0]) if verified else None
        event_rows.append({
            "event_id": str(event_id),
            "retained_entrants": int(len(group)),
            "source_starters": expected,
            "dnf_rows": int(group["dnf"].sum()),
            "complete": bool(expected == len(group)) if verified else None,
        })
    return {
        "starter_count_verified": verified,
        "events": len(event_rows),
        "entrants": int(len(df)),
        "dnf_rows": int(df["dnf"].sum()),
        "all_source_starters_retained": (
            all(row["complete"] for row in event_rows) if verified else None
        ),
        "event_counts": event_rows,
    }
