from f1_research import provider_reconciliation as pr


class _HTTP404(Exception):
    def __init__(self, endpoint):
        super().__init__(f"404 for {endpoint}")
        self.response = type("Response", (), {"status_code": 404})()


class _Client:
    provenance = []

    def get(self, endpoint, **params):
        if endpoint == "sessions":
            return [{"session_key": 7787, "meeting_key": 1143, "session_name": "Race"}]
        if endpoint == "meetings":
            return [{"meeting_key": 1143}]
        if endpoint == "session_result":
            return [{"driver_number": 1, "position": 1, "number_of_laps": 58}]
        if endpoint == "drivers":
            return [{"driver_number": 1, "name_acronym": "AAA"}]
        if endpoint == "laps":
            return [{"driver_number": 1, "lap_number": 1}]
        if endpoint in {"pit", "starting_grid"}:
            raise _HTTP404(endpoint)
        raise AssertionError(endpoint)


def test_openf1_historical_optional_404_preserves_core_provider(monkeypatch):
    monkeypatch.setattr(pr, "OpenF1Client", _Client)
    raw = pr.collect_openf1_raw(7787)

    assert raw["session"]["session_key"] == 7787
    assert raw["session_result"]
    assert raw["drivers"]
    assert raw["laps"]
    assert raw["pit"] == []
    assert raw["starting_grid"] is None
    assert raw["optional_collection_errors"]["pit"]["http_status"] == 404
    assert raw["optional_collection_errors"]["starting_grid"]["http_status"] == 404


def test_openf1_non_404_collection_failure_remains_hard(monkeypatch):
    class Client(_Client):
        def get(self, endpoint, **params):
            if endpoint == "pit":
                raise RuntimeError("upstream corrupt response")
            return super().get(endpoint, **params)

    monkeypatch.setattr(pr, "OpenF1Client", Client)
    try:
        pr.collect_openf1_raw(7787)
    except RuntimeError as exc:
        assert "corrupt response" in str(exc)
    else:
        raise AssertionError("non-404 provider failures must not be downgraded")
