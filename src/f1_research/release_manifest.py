"""Deployment/release manifest binding code, model, data truth and benchmark evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReleaseManifest:
    schema_version: int
    git_sha: str
    model_manifest_sha256: str
    data_truth_sha256: str
    benchmark_sha256: str
    replay_sha256: str | None


def build_release_manifest(*, git_sha: str, model_manifest: Path, data_truth: Path,
                           benchmark: Path, replay: Path | None = None) -> ReleaseManifest:
    if not git_sha.strip():
        raise ValueError("git_sha is required")
    return ReleaseManifest(
        schema_version=1,
        git_sha=git_sha,
        model_manifest_sha256=_sha(model_manifest),
        data_truth_sha256=_sha(data_truth),
        benchmark_sha256=_sha(benchmark),
        replay_sha256=_sha(replay) if replay is not None else None,
    )


def write_release_manifest(path: Path, manifest: ReleaseManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
