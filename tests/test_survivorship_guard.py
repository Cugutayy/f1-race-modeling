import pandas as pd
import pytest

from f1_research.data import survivorship_audit, validate


def _event():
    rows = []
    for position, driver, dnf in [(1, "a", 0), (2, "b", 0), (3, "c", 1)]:
        rows.append(
            {
                "event_id": "2026-01",
                "date": "2026-03-01",
                "year": 2026,
                "round": 1,
                "driver": driver,
                "team": f"team-{driver}",
                "circuit": "test",
                "quali_position": float(position),
                "quali_seconds": 80.0 + position,
                "finish_position": float(position),
                "points": 25.0 if position == 1 else 0.0,
                "dnf": dnf,
                "status": "Finished" if not dnf else "Accident",
                "starter_count": 3,
            }
        )
    return pd.DataFrame(rows)


def test_source_starter_count_prevents_survivorship_filtered_dataset():
    frame = _event()
    audit = survivorship_audit(frame)
    assert audit["starter_count_verified"] is True
    assert audit["all_source_starters_retained"] is True
    assert audit["dnf_rows"] == 1

    filtered = frame[frame.dnf.eq(0)].copy()
    with pytest.raises(ValueError, match="Survivorship guard failed"):
        validate(filtered)
