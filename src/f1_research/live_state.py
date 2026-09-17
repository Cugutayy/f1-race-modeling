"""Canonical event-time state for live and historical F1 streams.

The rest of the project never needs to know whether observations arrived through
OpenF1 REST, MQTT or a historical replay. Every observation is merged with an
explicit timestamp and stale/out-of-order messages cannot overwrite newer state.
Live MQTT/WebSocket revisions additionally respect OpenF1's monotonically increasing
``_id`` for each ``_key`` document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .data_truth import parse_provider_timestamp


def _utc(value: Any, fallback: datetime | None = None) -> datetime:
    parsed = parse_provider_timestamp(value)
    return parsed if parsed is not None else fallback or datetime.now(UTC)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None or number % 1:
        return None
    return int(number)


@dataclass
class DriverState:
    driver_number: int
    acronym: str | None = None
    full_name: str | None = None
    team_name: str | None = None
    team_colour: str | None = None
    position: int | None = None
    gap_to_leader_s: float | None = None
    interval_s: float | None = None
    lap_number: int | None = None
    last_lap_s: float | None = None
    sector_1_s: float | None = None
    sector_2_s: float | None = None
    sector_3_s: float | None = None
    compound: str | None = None
    stint_number: int | None = None
    tyre_age: int | None = None
    pit_stops: int = 0
    last_pit_lap: int | None = None
    speed_kmh: float | None = None
    throttle_pct: float | None = None
    brake: bool | None = None
    rpm: int | None = None
    gear: int | None = None
    drs: int | None = None
    x: float | None = None
    y: float | None = None
    last_seen_at: str | None = None
    topic_times: dict[str, str] = field(default_factory=dict)
    recent_laps_s: list[float] = field(default_factory=list)
    recent_lap_numbers: list[int] = field(default_factory=list)

    def remember_lap(self, lap: int | None, duration: float, keep: int = 8) -> None:
        if not np.isfinite(duration) or duration <= 0:
            return
        if lap is not None and lap in self.recent_lap_numbers:
            index = self.recent_lap_numbers.index(lap)
            self.recent_laps_s[index] = float(duration)
            return
        self.recent_laps_s.append(float(duration))
        self.recent_lap_numbers.append(lap if lap is not None else -1)
        self.recent_laps_s = self.recent_laps_s[-keep:]
        self.recent_lap_numbers = self.recent_lap_numbers[-keep:]


@dataclass
class WeatherState:
    air_temperature_c: float | None = None
    track_temperature_c: float | None = None
    humidity_pct: float | None = None
    pressure_mbar: float | None = None
    rainfall: bool | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    observed_at: str | None = None


@dataclass
class RaceState:
    session_key: int | None = None
    meeting_key: int | None = None
    session_name: str | None = None
    meeting_name: str | None = None
    status: str | None = None
    flag: str | None = None
    safety_car: str | None = None
    current_lap: int | None = None
    drivers: dict[int, DriverState] = field(default_factory=dict)
    weather: WeatherState = field(default_factory=WeatherState)
    updated_at: str | None = None
    latest_provider_event_at: str | None = None
    source: str = "OpenF1"
    received_messages: int = 0
    rejected_stale_messages: int = 0
    rejected_provider_order_messages: int = 0
    rejected_invalid_timestamp_messages: int = 0

    def driver(self, number: int) -> DriverState:
        if number not in self.drivers:
            self.drivers[number] = DriverState(number)
        return self.drivers[number]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["drivers"] = [asdict(self.drivers[key]) for key in sorted(self.drivers)]
        return payload


class RaceStateStore:
    """Merge OpenF1 topic observations into one monotonic state snapshot."""

    DRIVER_TOPICS = {
        "drivers", "position", "intervals", "laps", "stints", "pit", "car_data", "location"
    }

    def __init__(self, session_key: int | None = None):
        self.state = RaceState(session_key=session_key)
        self._global_topic_time: dict[str, datetime] = {}
        self._driver_topic_time: dict[tuple[int, str], datetime] = {}
        self._provider_versions: dict[tuple[str, str], int] = {}

    def _provider_version(self, topic: str, payload: dict[str, Any]) -> tuple[tuple[str, str], int] | None:
        provider_id = _integer(payload.get("_id"))
        provider_key = payload.get("_key")
        if provider_id is None or provider_key is None or str(provider_key) == "":
            return None
        return (topic, str(provider_key)), provider_id

    def _provider_is_fresh(self, version: tuple[tuple[str, str], int] | None) -> bool:
        if version is None:
            return True
        key, provider_id = version
        previous = self._provider_versions.get(key)
        if previous is not None and provider_id <= previous:
            self.state.rejected_provider_order_messages += 1
            return False
        return True

    def _record_provider_version(self, version: tuple[tuple[str, str], int] | None) -> None:
        if version is not None:
            key, provider_id = version
            self._provider_versions[key] = provider_id

    def _accept(self, topic: str, at: datetime, driver: int | None = None) -> bool:
        key = (driver, topic) if driver is not None else None
        previous = self._driver_topic_time.get(key) if key else self._global_topic_time.get(topic)
        if previous is not None and at < previous:
            self.state.rejected_stale_messages += 1
            return False
        if key:
            self._driver_topic_time[key] = at
        else:
            self._global_topic_time[topic] = at
        return True

    def _record_provider_event_time(self, at: datetime, explicit: bool) -> None:
        if not explicit:
            return
        previous = parse_provider_timestamp(self.state.latest_provider_event_at)
        if previous is None or at > previous:
            self.state.latest_provider_event_at = at.astimezone(UTC).isoformat()

    def ingest(self, topic: str, payload: dict[str, Any], received_at: datetime | None = None) -> bool:
        """Ingest one provider row; malformed/stale event time or revision returns False."""
        topic = topic.rsplit("/", 1)[-1]
        received_at = (received_at or datetime.now(UTC)).astimezone(UTC)
        raw_time = payload.get("date") or payload.get("date_start")
        explicit_time = raw_time is not None and raw_time != ""
        if explicit_time:
            at = parse_provider_timestamp(raw_time)
            if at is None:
                self.state.rejected_invalid_timestamp_messages += 1
                return False
        else:
            at = received_at

        driver_number = _integer(payload.get("driver_number"))
        provider_version = self._provider_version(topic, payload)
        if not self._provider_is_fresh(provider_version):
            return False
        if topic in self.DRIVER_TOPICS and driver_number is not None:
            if not self._accept(topic, at, driver_number):
                return False
        elif not self._accept(topic, at):
            return False
        self._record_provider_version(provider_version)
        self._record_provider_event_time(at, explicit_time)

        self.state.received_messages += 1
        self.state.updated_at = received_at.isoformat()
        self.state.session_key = _integer(payload.get("session_key")) or self.state.session_key
        self.state.meeting_key = _integer(payload.get("meeting_key")) or self.state.meeting_key

        if topic == "sessions":
            self.state.session_name = payload.get("session_name") or self.state.session_name
            self.state.meeting_name = payload.get("location") or self.state.meeting_name
        elif topic == "drivers" and driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.acronym = payload.get("name_acronym") or driver.acronym
            driver.full_name = payload.get("full_name") or driver.full_name
            driver.team_name = payload.get("team_name") or driver.team_name
            driver.team_colour = payload.get("team_colour") or driver.team_colour
        elif topic == "position" and driver_number is not None:
            self.state.driver(driver_number).position = _integer(payload.get("position"))
        elif topic == "intervals" and driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.gap_to_leader_s = _finite(payload.get("gap_to_leader"))
            driver.interval_s = _finite(payload.get("interval"))
        elif topic == "laps" and driver_number is not None:
            driver = self.state.driver(driver_number)
            lap = _integer(payload.get("lap_number"))
            duration = _finite(payload.get("lap_duration"))
            driver.lap_number = lap or driver.lap_number
            self.state.current_lap = max(self.state.current_lap or 0, lap or 0) or self.state.current_lap
            driver.last_lap_s = duration or driver.last_lap_s
            driver.sector_1_s = _finite(payload.get("duration_sector_1"))
            driver.sector_2_s = _finite(payload.get("duration_sector_2"))
            driver.sector_3_s = _finite(payload.get("duration_sector_3"))
            if duration is not None and not payload.get("is_pit_out_lap", False):
                driver.remember_lap(lap, duration)
        elif topic == "stints" and driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.compound = payload.get("compound") or driver.compound
            driver.stint_number = _integer(payload.get("stint_number")) or driver.stint_number
            start_age = _integer(payload.get("tyre_age_at_start"))
            lap_start = _integer(payload.get("lap_start"))
            lap_end = _integer(payload.get("lap_end")) or self.state.current_lap
            if start_age is not None and lap_start is not None and lap_end is not None:
                driver.tyre_age = max(0, start_age + lap_end - lap_start)
        elif topic == "pit" and driver_number is not None:
            driver = self.state.driver(driver_number)
            pit_lap = _integer(payload.get("lap_number"))
            if pit_lap is not None and pit_lap != driver.last_pit_lap:
                driver.pit_stops += 1
                driver.last_pit_lap = pit_lap
        elif topic == "car_data" and driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.speed_kmh = _finite(payload.get("speed"))
            driver.throttle_pct = _finite(payload.get("throttle"))
            brake = _integer(payload.get("brake"))
            driver.brake = None if brake is None else bool(brake)
            driver.rpm = _integer(payload.get("rpm"))
            driver.gear = _integer(payload.get("n_gear"))
            driver.drs = _integer(payload.get("drs"))
        elif topic == "location" and driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.x, driver.y = _finite(payload.get("x")), _finite(payload.get("y"))
        elif topic == "weather":
            self.state.weather = WeatherState(
                air_temperature_c=_finite(payload.get("air_temperature")),
                track_temperature_c=_finite(payload.get("track_temperature")),
                humidity_pct=_finite(payload.get("humidity")),
                pressure_mbar=_finite(payload.get("pressure")),
                rainfall=None if payload.get("rainfall") is None else bool(payload.get("rainfall")),
                wind_speed_ms=_finite(payload.get("wind_speed")),
                wind_direction_deg=_finite(payload.get("wind_direction")),
                observed_at=at.isoformat(),
            )
        elif topic == "race_control":
            category = str(payload.get("category") or "")
            if category == "SessionStatus":
                self.state.status = payload.get("message") or self.state.status
            if category == "Flag":
                self.state.flag = payload.get("flag") or self.state.flag
            if category == "SafetyCar":
                self.state.safety_car = payload.get("message") or self.state.safety_car

        if driver_number is not None:
            driver = self.state.driver(driver_number)
            driver.last_seen_at = at.isoformat()
            driver.topic_times[topic] = at.isoformat()
        return True

    def ingest_many(self, topic: str, rows: list[dict[str, Any]]) -> int:
        def order(item: dict[str, Any]) -> tuple[int, datetime]:
            raw = item.get("date") or item.get("date_start")
            parsed = parse_provider_timestamp(raw)
            if parsed is None:
                return 1, datetime.max.replace(tzinfo=UTC)
            return 0, parsed

        accepted = 0
        for row in sorted(rows, key=order):
            accepted += int(self.ingest(topic, row))
        return accepted

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        now = (now or datetime.now(UTC)).astimezone(UTC)
        payload = self.state.to_dict()
        updated = parse_provider_timestamp(self.state.updated_at) or now
        payload["snapshot_at"] = now.isoformat()
        payload["data_age_s"] = max(0.0, (now - updated).total_seconds())
        return payload
