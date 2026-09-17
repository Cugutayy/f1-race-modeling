"""Fail-closed event identity checks for cross-provider race reconciliation.

This module answers only one question: do the provider snapshots belong to the same
completed race? It intentionally uses deterministic metadata checks rather than fuzzy
matching or majority voting. Provider-specific location labels are retained as audit
evidence but are not a hard identity key because public sources use different circuit
versus city naming conventions.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

_COUNTRY_ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "uae": "united arab emirates",
}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_label(value: Any) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", ascii_text.casefold())
    return " ".join(tokens) or None


def _normalized_country(value: Any) -> str | None:
    text = _normalized_label(value)
    return _COUNTRY_ALIASES.get(text, text) if text is not None else None


def _int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _date(value: Any) -> date | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _jolpica_event(raw: dict[str, Any] | None) -> dict[str, Any]:
    try:
        races = raw["results"]["MRData"]["RaceTable"]["Races"]  # type: ignore[index]
    except (KeyError, TypeError):
        return {}
    if not isinstance(races, list) or len(races) != 1 or not isinstance(races[0], dict):
        return {}
    race = races[0]
    circuit = race.get("Circuit") if isinstance(race.get("Circuit"), dict) else {}
    location = circuit.get("Location") if isinstance(circuit.get("Location"), dict) else {}
    return {
        "season": _int(race.get("season")),
        "round": _int(race.get("round")),
        "event_name": _clean(race.get("raceName")),
        "date": _clean(race.get("date")),
        "time": _clean(race.get("time")),
        "country": _clean(location.get("country")),
        "location": _clean(location.get("locality")),
        "circuit": _clean(circuit.get("circuitName")) or _clean(circuit.get("circuitId")),
    }


def _openf1_event(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    session = raw.get("session") if isinstance(raw.get("session"), dict) else {}
    meeting = raw.get("meeting") if isinstance(raw.get("meeting"), dict) else {}
    return {
        "session_key": _int(session.get("session_key")),
        "session_name": _clean(session.get("session_name")),
        "session_type": _clean(session.get("session_type")),
        "session_meeting_key": _int(session.get("meeting_key")),
        "meeting_key": _int(meeting.get("meeting_key")),
        "event_name": _clean(meeting.get("meeting_name")),
        "official_name": _clean(meeting.get("meeting_official_name")),
        "year": _int(session.get("year")),
        "meeting_year": _int(meeting.get("year")),
        "date_start": _clean(session.get("date_start")),
        "country": _clean(meeting.get("country_name")) or _clean(session.get("country_name")),
        "location": _clean(meeting.get("location")) or _clean(session.get("location")),
        "session_cancelled": session.get("is_cancelled"),
        "meeting_cancelled": meeting.get("is_cancelled"),
    }


def _fastf1_event(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    event = raw.get("event") if isinstance(raw.get("event"), dict) else {}
    return {
        "event_name": _clean(event.get("EventName")),
        "round": _int(event.get("RoundNumber")),
        "date": _clean(event.get("EventDate")),
        "country": _clean(event.get("Country")),
        "location": _clean(event.get("Location")),
    }


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    *,
    values: dict[str, Any],
    passed: bool,
    required: bool,
    reason: str,
) -> None:
    checks[name] = {
        "required": required,
        "passed": bool(passed),
        "values": values,
        "reason": reason,
    }


def build_event_identity(
    *,
    year: int,
    round_number: int,
    openf1_session_key: int,
    jolpica_raw: dict[str, Any] | None,
    openf1_raw: dict[str, Any] | None,
    fastf1_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build auditable same-event evidence without electing provider truth."""
    jolpica = _jolpica_event(jolpica_raw)
    openf1 = _openf1_event(openf1_raw)
    fastf1 = _fastf1_event(fastf1_raw)
    checks: dict[str, dict[str, Any]] = {}

    _check(
        checks,
        "requested_year",
        values={
            "requested": int(year),
            "jolpica": jolpica.get("season"),
            "openf1_session": openf1.get("year"),
            "openf1_meeting": openf1.get("meeting_year"),
            "fastf1_event_date_year": _date(fastf1.get("date")).year if _date(fastf1.get("date")) else None,
        },
        passed=(
            jolpica.get("season") == int(year)
            and openf1.get("year") == int(year)
            and openf1.get("meeting_year") == int(year)
            and _date(fastf1.get("date")) is not None
            and _date(fastf1.get("date")).year == int(year)
        ),
        required=True,
        reason="All provider event metadata must identify the requested season.",
    )
    _check(
        checks,
        "requested_round",
        values={
            "requested": int(round_number),
            "jolpica": jolpica.get("round"),
            "fastf1": fastf1.get("round"),
        },
        passed=(jolpica.get("round") == int(round_number) and fastf1.get("round") == int(round_number)),
        required=True,
        reason="Jolpica and FastF1 must identify the requested championship round.",
    )
    _check(
        checks,
        "openf1_race_session",
        values={
            "requested_session_key": int(openf1_session_key),
            "session_key": openf1.get("session_key"),
            "session_name": openf1.get("session_name"),
            "session_type": openf1.get("session_type"),
        },
        passed=(
            openf1.get("session_key") == int(openf1_session_key)
            and _normalized_label(openf1.get("session_name")) == "race"
            and (
                openf1.get("session_type") is None
                or _normalized_label(openf1.get("session_type")) == "race"
            )
        ),
        required=True,
        reason="The explicit OpenF1 key must resolve to the requested Race session.",
    )
    _check(
        checks,
        "openf1_meeting_link",
        values={
            "session_meeting_key": openf1.get("session_meeting_key"),
            "meeting_key": openf1.get("meeting_key"),
        },
        passed=(
            openf1.get("session_meeting_key") is not None
            and openf1.get("session_meeting_key") == openf1.get("meeting_key")
        ),
        required=True,
        reason="OpenF1 Race session and meeting payload must reference the same meeting_key.",
    )

    normalized_names = {
        "jolpica": _normalized_label(jolpica.get("event_name")),
        "openf1": _normalized_label(openf1.get("event_name")),
        "fastf1": _normalized_label(fastf1.get("event_name")),
    }
    _check(
        checks,
        "event_name",
        values={
            "raw": {
                "jolpica": jolpica.get("event_name"),
                "openf1": openf1.get("event_name"),
                "fastf1": fastf1.get("event_name"),
            },
            "normalized": normalized_names,
        },
        passed=(None not in normalized_names.values() and len(set(normalized_names.values())) == 1),
        required=True,
        reason="Event names must match after deterministic Unicode/case/punctuation normalization; fuzzy matching is forbidden.",
    )

    normalized_countries = {
        "jolpica": _normalized_country(jolpica.get("country")),
        "openf1": _normalized_country(openf1.get("country")),
        "fastf1": _normalized_country(fastf1.get("country")),
    }
    _check(
        checks,
        "country",
        values={
            "raw": {
                "jolpica": jolpica.get("country"),
                "openf1": openf1.get("country"),
                "fastf1": fastf1.get("country"),
            },
            "normalized": normalized_countries,
        },
        passed=(None not in normalized_countries.values() and len(set(normalized_countries.values())) == 1),
        required=True,
        reason="Countries must match after an explicit alias table; no fuzzy country matching is used.",
    )

    dates = {
        "jolpica": _date(jolpica.get("date")),
        "openf1": _date(openf1.get("date_start")),
        "fastf1": _date(fastf1.get("date")),
    }
    observed_dates = [value for value in dates.values() if value is not None]
    date_span_days = (
        (max(observed_dates) - min(observed_dates)).days if len(observed_dates) == 3 else None
    )
    _check(
        checks,
        "race_date",
        values={
            "jolpica": str(dates["jolpica"]) if dates["jolpica"] else None,
            "openf1": str(dates["openf1"]) if dates["openf1"] else None,
            "fastf1": str(dates["fastf1"]) if dates["fastf1"] else None,
            "max_calendar_day_span": date_span_days,
        },
        passed=(len(observed_dates) == 3 and date_span_days is not None and date_span_days <= 1),
        required=True,
        reason="Race dates must all be present and within one calendar day to tolerate provider timezone/date semantics.",
    )

    cancelled = {
        "openf1_session": openf1.get("session_cancelled"),
        "openf1_meeting": openf1.get("meeting_cancelled"),
    }
    _check(
        checks,
        "not_cancelled",
        values=cancelled,
        passed=(
            openf1.get("session_cancelled") is False
            and openf1.get("meeting_cancelled") is False
        ),
        required=True,
        reason=(
            "OpenF1 meeting/session cancellation flags must both be explicitly observed False; "
            "missing evidence is not treated as not-cancelled."
        ),
    )

    normalized_locations = {
        "jolpica": _normalized_label(jolpica.get("location")),
        "openf1": _normalized_label(openf1.get("location")),
        "fastf1": _normalized_label(fastf1.get("location")),
    }
    _check(
        checks,
        "location_label",
        values={
            "raw": {
                "jolpica": jolpica.get("location"),
                "openf1": openf1.get("location"),
                "fastf1": fastf1.get("location"),
            },
            "normalized": normalized_locations,
        },
        passed=(None not in normalized_locations.values() and len(set(normalized_locations.values())) == 1),
        required=False,
        reason="Location labels are audit evidence only because providers may use city versus circuit locality names.",
    )

    failures = [name for name, item in checks.items() if item["required"] and not item["passed"]]
    warnings = [name for name, item in checks.items() if not item["required"] and not item["passed"]]
    return {
        "schema_version": 1,
        "verified": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "providers": {
            "Jolpica": jolpica,
            "OpenF1": openf1,
            "FastF1": fastf1,
        },
        "policy": {
            "fuzzy_matching": False,
            "majority_vote": False,
            "date_tolerance_calendar_days": 1,
            "location_is_hard_identity": False,
        },
    }


def require_event_identity(identity: dict[str, Any]) -> None:
    """Fail closed before result reconciliation when same-event evidence is incomplete."""
    if identity.get("verified") is True:
        return
    failures = identity.get("failures") or ["unknown_event_identity_failure"]
    raise ValueError("Provider event identity verification failed: " + ", ".join(map(str, failures)))
