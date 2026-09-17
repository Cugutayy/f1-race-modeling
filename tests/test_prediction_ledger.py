import json

import pytest

from f1_research.prediction_ledger import append_jsonl, make_record


def _record(**overrides):
    args = dict(
        event_id="2026-01", forecast_origin="post_qualifying",
        model_id="rank-ensemble-v1", model_sha256="m" * 64,
        features={"driver": "VER", "quali_position": 1},
        evidence_sha256="e" * 64, cutoff_at="2026-03-07T08:00:00+00:00",
        payload={"win_probability": 0.4}, created_at="2026-03-07T08:01:00+00:00",
    )
    args.update(overrides)
    return make_record(**args)


def test_prediction_id_is_content_addressed_not_clock_addressed():
    first = _record(created_at="2026-03-07T08:01:00+00:00")
    second = _record(created_at="2026-03-07T08:02:00+00:00")
    assert first.prediction_id == second.prediction_id
    assert first.feature_sha256 == second.feature_sha256


def test_append_is_idempotent_for_same_prediction(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_jsonl(path, _record())
    append_jsonl(path, _record(created_at="2026-03-07T09:00:00+00:00"))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 1


def test_missing_provenance_fails_closed():
    with pytest.raises(ValueError, match="provenance"):
        _record(evidence_sha256="")
