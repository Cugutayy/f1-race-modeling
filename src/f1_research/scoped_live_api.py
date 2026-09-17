"""Production API overlay for unit-safe lap-deficit handling.

The legacy API remains the source of shared I/O/auth/health helpers. This module
replaces only the probability and strategy routes so OpenF1 ``+N LAP(S)`` rows stay
visible in classification without being coerced into fabricated seconds.
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from . import live_api as base
from .live_intelligence import combined_live_report, combined_pit_windows
from .reliability import reliability_overrides_from_state
from .simulation_scope import (
    annotate_simulation_report,
    ensure_strategy_eligible,
    split_simulation_scope,
)
from .strategy import compare_pit_windows, predict_from_state

app = base.app

# Remove only the two routes that need a unit-safe simulation scope. All health,
# evidence and telemetry routes keep their existing implementation and auth contract.
app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "path", None) not in {"/v1/live", "/v1/strategy"}
]


def _scoped_live_report(total_laps: int, samples: int) -> dict:
    state = base._read_state()
    truth_audit = base._trusted_live_audit(state)
    try:
        simulation_state, classification_only = split_simulation_scope(state)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    config, prior_audit = base._simulation_config(samples)
    reliability_model = prior_audit.get("reliability") if isinstance(prior_audit, dict) else None
    reliability_overrides = reliability_overrides_from_state(simulation_state, reliability_model)
    artifact = base._load_artifact()

    try:
        if artifact is not None:
            report = combined_live_report(
                simulation_state,
                total_laps,
                artifact,
                config=config,
                reliability_model=reliability_model,
            )
            pace_status = "strict_model"
        else:
            report = predict_from_state(
                simulation_state,
                total_laps,
                config=config,
                dnf_hazard_overrides=reliability_overrides,
            )
            report["pace_predictions"] = []
            report["pace_model"] = {
                "task": None,
                "status": "fallback_recent_laps",
                "warning": "Strict model artifact is unavailable",
            }
            report["reliability_model"] = {
                "enabled": bool(reliability_model and reliability_model.get("enabled")),
                "override_drivers": sorted(reliability_overrides),
                "source": (
                    "hierarchical_public_results_survival"
                    if reliability_overrides
                    else "pooled_config_fallback"
                ),
            }
            pace_status = "fallback_recent_laps"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    annotate_simulation_report(report, classification_only)
    # Always return the full observed state. The scoped copy is an internal model input,
    # not a replacement for provider truth.
    report["state"] = state
    report["state_age_s"] = base._state_age_s(state)
    report["data_truth"] = truth_audit
    report["strategy_prior_source"] = prior_audit
    report["pace_status"] = pace_status
    report["simulation_scope"] = simulation_state.get("simulation_scope")
    return base._safe(report)


@app.get("/v1/live")
def live(
    total_laps: int = Query(ge=2, le=100),
    samples: int = Query(default=4000, ge=1000, le=50000),
    _: None = Depends(base._authorize),
) -> JSONResponse:
    return JSONResponse(_scoped_live_report(total_laps, samples))


@app.get("/v1/strategy")
def strategy(
    driver_number: int = Query(ge=1, le=999),
    total_laps: int = Query(ge=2, le=100),
    samples: int = Query(default=4000, ge=1000, le=50000),
    _: None = Depends(base._authorize),
) -> JSONResponse:
    state = base._read_state()
    truth_audit = base._trusted_live_audit(state)
    try:
        simulation_state, classification_only = split_simulation_scope(state)
        ensure_strategy_eligible(driver_number, classification_only)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    config, prior_audit = base._simulation_config(samples)
    reliability_model = prior_audit.get("reliability") if isinstance(prior_audit, dict) else None
    reliability_overrides = reliability_overrides_from_state(simulation_state, reliability_model)
    artifact = base._load_artifact()
    try:
        if artifact is not None:
            scenarios = combined_pit_windows(
                simulation_state,
                total_laps,
                driver_number,
                artifact,
                config=config,
                reliability_model=reliability_model,
            )
            pace_status = "strict_model"
        else:
            scenarios = compare_pit_windows(
                simulation_state,
                total_laps,
                driver_number,
                config=config,
                dnf_hazard_overrides=reliability_overrides,
            )
            pace_status = "fallback_recent_laps"
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JSONResponse(base._safe({
        "driver_number": driver_number,
        "session_key": state.get("session_key"),
        "state_updated_at": state.get("updated_at"),
        "state_age_s": base._state_age_s(state),
        "data_truth": truth_audit,
        "pace_status": pace_status,
        "reliability_status": (
            "hierarchical_public_results_survival"
            if reliability_overrides
            else "pooled_config_fallback"
        ),
        "strategy_prior_source": prior_audit,
        "simulation_scope": simulation_state.get("simulation_scope"),
        "classification_only": classification_only,
        "scenarios": scenarios,
    }))


def main() -> None:
    """Run the unit-safe persistent-worker API locally or on an always-on host."""
    import uvicorn

    host = os.environ.get("F1_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("F1_API_PORT", "8000")))
    uvicorn.run("f1_research.scoped_live_api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
