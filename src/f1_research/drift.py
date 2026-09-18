"""Detect material feature/prediction drift without pretending statistical certainty."""
from __future__ import annotations

import numpy as np


def population_stability_index(reference, current, *, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        raise ValueError("PSI requires at least 20 finite observations per sample")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(ref, cur.mean()) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    rp = np.histogram(ref, bins=edges)[0] / len(ref)
    cp = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    rp, cp = np.clip(rp, eps, None), np.clip(cp, eps, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def drift_report(reference: dict[str, list[float]], current: dict[str, list[float]],
                 *, warning_psi: float = 0.20) -> dict:
    shared = sorted(set(reference) & set(current))
    if not shared:
        raise ValueError("no shared drift features")
    rows = []
    for name in shared:
        psi = population_stability_index(reference[name], current[name])
        rows.append({"feature": name, "psi": psi, "warning": bool(psi >= warning_psi)})
    return {
        "schema_version": 1,
        "warning_threshold_psi": warning_psi,
        "warning": any(row["warning"] for row in rows),
        "features": rows,
        "note": "PSI is an operational drift signal, not proof of causal model degradation.",
    }
