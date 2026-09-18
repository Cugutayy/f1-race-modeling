"""Content-addressed model manifests and fail-closed artifact verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .revision import validate_git_sha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    model_sha256: str
    feature_schema_sha256: str
    training_data_sha256: str
    calibration_sha256: str
    trained_until: str
    benchmark_run_id: str
    git_sha: str

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported model manifest schema")
        for name in ("model_id", "trained_until", "benchmark_run_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"missing model manifest field: {name}")
        validate_git_sha(self.git_sha, field="git_sha")
        for name in ("model_sha256", "feature_schema_sha256", "training_data_sha256", "calibration_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError(f"{name} must be a SHA-256 hex digest")


def write_manifest(path: Path, manifest: ModelManifest) -> None:
    manifest.validate()
    Path(path).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def load_manifest(path: Path, *, model_path: Path | None = None) -> ModelManifest:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = ModelManifest(**raw)
    manifest.validate()
    if model_path is not None and sha256_file(model_path) != manifest.model_sha256:
        raise ValueError("model artifact SHA-256 does not match manifest")
    return manifest
