"""Railway-facing ASGI wrapper around the protected live API.

Railway's platform health checker cannot attach our bearer token. Keep the
application's operational `/healthz` and every data endpoint protected, while
exposing only a minimal process-liveness probe for infrastructure.
"""

from fastapi import FastAPI

from .live_api import app as live_app

app: FastAPI = live_app


@app.get("/railway_healthz", include_in_schema=False)
def railway_healthz() -> dict[str, bool]:
    """Return process liveness only; do not expose race state or credentials."""
    return {"ok": True}
