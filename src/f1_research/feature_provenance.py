"""Feature-level provenance contract for event-time model inputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

QUALITY = {"OBSERVED", "DERIVED", "CALIBRATED", "MODELLED", "IMPUTED", "DEFAULT", "UNKNOWN"}


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    value: Any
    observed_at: str
    available_at: str
    source: str
    source_sha256: str
    quality: str

    def validate(self, *, cutoff_at: str) -> None:
        if self.quality not in QUALITY:
            raise ValueError(f"invalid feature quality: {self.quality}")
        if not self.feature or not self.source:
            raise ValueError("feature/source must be explicit")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        observed = pd.Timestamp(self.observed_at)
        available = pd.Timestamp(self.available_at)
        cutoff = pd.Timestamp(cutoff_at)
        if observed.tzinfo is None or available.tzinfo is None or cutoff.tzinfo is None:
            raise ValueError("feature timestamps and cutoff must be timezone-aware")
        if available.tz_convert("UTC") > cutoff.tz_convert("UTC"):
            raise ValueError(f"feature {self.feature} was unavailable at forecast cutoff")


def evidence_sha256(items: list[FeatureEvidence], *, cutoff_at: str) -> str:
    if not items:
        raise ValueError("feature evidence cannot be empty")
    for item in items:
        item.validate(cutoff_at=cutoff_at)
    payload = [asdict(item) for item in sorted(items, key=lambda x: (x.feature, x.source))]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()
