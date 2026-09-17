import json

import pytest

from f1_research.live_protocol import encode, envelope


def test_live_envelope_is_versioned_and_hashed():
    item = envelope(sequence=1, event_type="state_update", state={"lap": 2},
                    payload={"lap": 2}, provider_time="2026-03-08T05:00:00Z",
                    generated_at="2026-03-08T05:00:01Z")
    raw = json.loads(encode(item))
    assert raw["schema_version"] == 1
    assert raw["sequence"] == 1
    assert len(raw["state_sha256"]) == 64


def test_live_envelope_rejects_nonpositive_sequence():
    with pytest.raises(ValueError, match="sequence"):
        envelope(sequence=0, event_type="state", state={}, payload={})
