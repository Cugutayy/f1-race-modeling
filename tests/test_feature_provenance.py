import hashlib

import pytest

from f1_research.feature_provenance import FeatureEvidence, evidence_sha256


def _item(**overrides):
    values = dict(
        feature="grid_position", value=1, observed_at="2026-03-07T07:50:00Z",
        available_at="2026-03-07T07:51:00Z", source="OpenF1",
        source_sha256=hashlib.sha256(b"source").hexdigest(), quality="OBSERVED",
    )
    values.update(overrides)
    return FeatureEvidence(**values)


def test_feature_evidence_hash_is_deterministic():
    cutoff = "2026-03-07T08:00:00Z"
    assert evidence_sha256([_item()], cutoff_at=cutoff) == evidence_sha256([_item()], cutoff_at=cutoff)


def test_feature_available_after_cutoff_is_rejected():
    with pytest.raises(ValueError, match="unavailable"):
        evidence_sha256([_item(available_at="2026-03-07T08:00:01Z")],
                        cutoff_at="2026-03-07T08:00:00Z")


def test_unknown_quality_label_is_rejected():
    with pytest.raises(ValueError, match="quality"):
        evidence_sha256([_item(quality="GUESSED")], cutoff_at="2026-03-07T08:00:00Z")


def test_feature_cannot_be_available_before_observation():
    with pytest.raises(ValueError, match="before it was observed"):
        evidence_sha256(
            [_item(observed_at="2026-03-07T07:55:00Z", available_at="2026-03-07T07:54:59Z")],
            cutoff_at="2026-03-07T08:00:00Z",
        )


def test_feature_source_digest_must_be_hex():
    with pytest.raises(ValueError, match="SHA-256"):
        evidence_sha256([_item(source_sha256="z" * 64)], cutoff_at="2026-03-07T08:00:00Z")
