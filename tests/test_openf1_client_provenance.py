import hashlib
from datetime import datetime

from f1_research.openf1_live import OpenF1Client


class _FakeResponse:
    def __init__(self):
        self.content = b'[{"driver_number":4}]'
        self.url = "https://api.openf1.org/v1/drivers?session_key=9693"
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return [{"driver_number": 4}]


def test_openf1_get_records_response_byte_provenance(monkeypatch):
    client = OpenF1Client()
    response = _FakeResponse()
    monkeypatch.setattr(client.session, "get", lambda *_args, **_kwargs: response)

    rows = client.get("drivers", session_key=9693)

    assert rows == [{"driver_number": 4}]
    assert len(client.provenance) == 1
    record = client.provenance[0]
    assert record["provider"] == "OpenF1"
    assert record["endpoint"] == "drivers"
    assert record["url"] == response.url
    assert record["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert record["row_count"] == 1
    assert record["http_status"] == 200
    datetime.fromisoformat(record["retrieved_at"])


class _FakeNotFoundResponse:
    def __init__(self):
        self.content = b'{"detail":"No results found."}'
        self.url = "https://api.openf1.org/v1/starting_grid?session_key=9693"
        self.status_code = 404

    def raise_for_status(self):
        import requests
        raise requests.HTTPError("404 Client Error", response=self)

    def json(self):
        return {"detail": "No results found."}


def test_openf1_http_failure_is_hashed_before_raise(monkeypatch):
    import pytest
    import requests

    client = OpenF1Client()
    response = _FakeNotFoundResponse()
    monkeypatch.setattr(client.session, "get", lambda *_args, **_kwargs: response)

    with pytest.raises(requests.HTTPError):
        client.get("starting_grid", session_key=9693)

    assert client.provenance[-1]["endpoint"] == "starting_grid"
    assert client.provenance[-1]["http_status"] == 404
    assert client.provenance[-1]["row_count"] is None
    assert client.provenance[-1]["sha256"] == hashlib.sha256(response.content).hexdigest()
