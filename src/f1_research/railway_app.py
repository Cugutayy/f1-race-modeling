"""Railway-facing ASGI wrapper around the protected live API.

Railway's platform health checker cannot attach our bearer token. Keep the
application's operational `/healthz` and every data endpoint protected, while
exposing only a minimal process-liveness probe for infrastructure.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from .live_api import app as live_app

app: FastAPI = live_app


@app.get("/railway_healthz", include_in_schema=False)
def railway_healthz() -> dict[str, bool]:
    """Return process liveness only; do not expose race state or credentials."""
    return {"ok": True}


def main() -> None:
    """Run the Railway wrapper without relying on shell expansion of ``$PORT``."""
    import uvicorn

    host = os.environ.get("F1_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("F1_API_PORT", "8000")))
    uvicorn.run("f1_research.railway_app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
