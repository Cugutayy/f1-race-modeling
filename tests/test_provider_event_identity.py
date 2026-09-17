import copy

import pytest

from f1_research.provider_event_identity import build_event_identity, require_event_identity


def _raw_triplet():
    jolpica = {
        "results": {"MRData": {"RaceTable": {"Races": [{
            "season": "2025",
            "round": "1",
            "raceName": "Australian Grand Prix",
            "date": "2025-03-16",
            "time": "04:00:00Z",
            "Circuit": {
                "circuitName": "Albert Park Grand Prix Circuit",
                "Location": {"locality": "Melbourne", "country": "Australia"},
            },
        }]}}},
    }
    openf1 = {
        "session": {
            "session_key": 9693,
            "meeting_key": 1254,
            "session_name": "Race",
            "session_type": "Race",
            "year": 2025,
            "date_start": "2025-03-16T04:00:00+00:00",
            "country_name": "Australia",
            "location": "Melbourne",
            "is_cancelled": False,
        },
        "meeting": {
            "meeting_key": 1254,
            "meeting_name": "Australian Grand Prix",
            "meeting_official_name": "FORMULA 1 AUSTRALIAN GRAND PRIX 2025",
            "year": 2025,
            "country_name": "Australia",
            "location": "Melbourne",
            "is_cancelled": False,
        },
    }
    fastf1 = {
        "event": {
            "EventName": "Australian Grand Prix",
            "RoundNumber": 1,
            "EventDate": "2025-03-16 00:00:00",
            "Country": "Australia",
            "Location": "Melbourne",
        }
    }
    return jolpica, openf1, fastf1


def _identity(jolpica, openf1, fastf1):
    return build_event_identity(
        year=2025,
        round_number=1,
        openf1_session_key=9693,
        jolpica_raw=jolpica,
        openf1_raw=openf1,
        fastf1_raw=fastf1,
    )


def test_matching_three_provider_event_metadata_is_verified():
    identity = _identity(*_raw_triplet())
    assert identity["verified"] is True
    assert identity["failures"] == []
    require_event_identity(identity)


def test_same_year_wrong_openf1_race_is_rejected_before_result_comparison():
    jolpica, openf1, fastf1 = _raw_triplet()
    openf1 = copy.deepcopy(openf1)
    openf1["meeting"]["meeting_name"] = "Chinese Grand Prix"
    openf1["meeting"]["country_name"] = "China"

    identity = _identity(jolpica, openf1, fastf1)
    assert identity["verified"] is False
    assert "event_name" in identity["failures"]
    assert "country" in identity["failures"]
    with pytest.raises(ValueError, match="event identity verification failed"):
        require_event_identity(identity)


def test_openf1_session_must_link_to_the_fetched_meeting():
    jolpica, openf1, fastf1 = _raw_triplet()
    openf1 = copy.deepcopy(openf1)
    openf1["meeting"]["meeting_key"] = 9999

    identity = _identity(jolpica, openf1, fastf1)
    assert identity["verified"] is False
    assert identity["checks"]["openf1_meeting_link"]["passed"] is False


def test_location_label_difference_is_visible_but_not_fuzzy_repaired():
    jolpica, openf1, fastf1 = _raw_triplet()
    openf1 = copy.deepcopy(openf1)
    openf1["meeting"]["location"] = "Albert Park"

    identity = _identity(jolpica, openf1, fastf1)
    assert identity["verified"] is True
    assert identity["warnings"] == ["location_label"]
    assert identity["checks"]["location_label"]["required"] is False
    assert identity["policy"]["fuzzy_matching"] is False
