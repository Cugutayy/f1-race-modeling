"""Source-revision helpers for evidence and model provenance."""
from __future__ import annotations

import os
import subprocess


def validate_git_sha(value: str, *, field: str = "git_sha") -> str:
    text = str(value or "").strip().lower()
    if len(text) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be a 40- or 64-character hexadecimal Git object id")
    return text


def current_git_sha(*, required: bool = True) -> str | None:
    """Return the exact checked-out Git revision; never invent a provenance id."""
    candidate = os.environ.get("GITHUB_SHA")
    if not candidate:
        try:
            candidate = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            candidate = None
    if candidate:
        return validate_git_sha(candidate)
    if required:
        raise RuntimeError("A verifiable Git revision is required for evidence production")
    return None
