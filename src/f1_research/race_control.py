"""Race-control state machine with explicit fail-closed transitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATES = {"UNKNOWN", "GREEN", "YELLOW", "VSC", "SC", "RED", "RESTART", "FINISHED"}


@dataclass(frozen=True)
class TrackState:
    state: str = "UNKNOWN"
    changed_at: str | None = None
    message: str | None = None

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unsupported track state: {self.state}")


def reduce_race_control(current: TrackState, event: dict[str, Any]) -> TrackState:
    message = str(event.get("message") or "").upper()
    flag = str(event.get("flag") or "").upper()
    category = str(event.get("category") or "").upper()
    date = event.get("date")

    if flag == "CHEQUERED" or "CHEQUERED FLAG" in message or "SESSION ENDED" in message:
        state = "FINISHED"
    elif flag == "RED" or "RED FLAG" in message:
        state = "RED"
    elif "VIRTUAL SAFETY CAR" in message or "VSC" in message:
        state = "GREEN" if any(x in message for x in ("ENDING", "ENDED")) else "VSC"
    elif "SAFETY CAR" in message:
        state = "GREEN" if any(x in message for x in ("IN THIS LAP", "ENDING", "ENDED")) else "SC"
    elif flag in {"YELLOW", "DOUBLE YELLOW"}:
        state = "YELLOW"
    elif flag == "GREEN":
        state = "RESTART" if current.state in {"RED", "SC", "VSC"} else "GREEN"
    elif category == "SESSIONSTATUS" and "STARTED" in message:
        state = "GREEN"
    else:
        return current
    return TrackState(state=state, changed_at=str(date) if date is not None else current.changed_at,
                      message=str(event.get("message") or "") or None)
