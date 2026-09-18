"""Strict parsing helpers for provider-supplied scalar values.

Provider payloads must not be interpreted through Python truthiness. In particular,
non-empty strings such as "false" are truthy in Python and would silently corrupt
boolean race facts if passed through bool(value).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def strict_optional_bool(value: Any, *, field: str) -> bool | None:
    """Parse an explicitly encoded boolean or preserve missing evidence as None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
    raise ValueError(f"Unsupported explicit boolean for {field}: {value!r}")
