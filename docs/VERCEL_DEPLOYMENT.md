# Vercel deployment

The live platform is intentionally split into two processes.

## 1. Persistent Python worker

Run this on an always-on host or a machine that can keep the OpenF1 stream and local artifacts alive. Do **not** use Vercel's ephemeral filesystem as the source of truth for `state.json`, model artifacts, or raw capture files.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[live,api]'

# terminal/process 1
export OPENF1_USERNAME='...'
export OPENF1_PASSWORD='...'
f1-live capture --output reports/local/live --session-key latest

# terminal/process 2
export F1_API_TOKEN='replace-with-a-long-random-secret'
export F1_LIVE_STATE_PATH="$PWD/reports/local/live/state.json"
export F1_LIVE_EVENTS_PATH="$PWD/reports/local/live/events.jsonl"
export F1_STRICT_MODEL_PATH="$PWD/reports/local/lap-strict/next_lap_strict.joblib"
export F1_STRATEGY_PRIORS_PATH="$PWD/reports/local/lap-strict/strategy_priors.json"
f1-api
```

The API exposes read-only endpoints:

- `GET /healthz`
- `GET /v1/live?total_laps=57&samples=4000`
- `GET /v1/telemetry?driver_number=1&limit=700`
- `GET /v1/strategy?driver_number=1&total_laps=57&samples=4000`

If `F1_API_TOKEN` is set, clients must send `Authorization: Bearer <token>`.

## 2. Vercel frontend

Create/import the GitHub repository in Vercel and configure:

- **Root Directory:** `web`
- **Framework Preset:** Next.js
- **Node.js:** 22.x
- **Build Command:** `npm run build` (automatic is fine)
- **Install Command:** `npm install` (automatic is fine)

Set server-side Vercel environment variables:

```text
F1_BACKEND_URL=https://your-persistent-python-api.example.com
F1_BACKEND_TOKEN=<same secret as F1_API_TOKEN>
```

Do **not** prefix the token with `NEXT_PUBLIC_`; it must stay server-side. Browser requests only hit the Vercel route handlers under `/api/*`. Those route handlers attach the token and proxy to the Python worker.

## Why this split exists

The capture service is stateful and continuously consumes OpenF1 updates. It writes append-only raw events and atomically updates the canonical race state. The trained model and strategy-prior files are also local worker artifacts. The Vercel UI, by contrast, is stateless and can be rebuilt/restarted without losing race state.

The separation also prevents OpenF1 credentials, model artifacts and the backend bearer token from reaching the browser.

## Pre-deploy checks

Python/research checks:

```bash
python -m ruff check .
python -m pytest -q
```

Frontend checks:

```bash
cd web
npm install
npm run typecheck
npm run build
```

GitHub Actions runs both sets independently (`Research checks` and `Web checks`). Do not deploy a red commit.

## Common Vercel failures

### `F1_BACKEND_URL is not configured`

Add `F1_BACKEND_URL` to the Vercel project's Preview and/or Production environment and redeploy.

### `401 Unauthorized`

`F1_BACKEND_TOKEN` does not match the worker's `F1_API_TOKEN`.

### `503` from `/api/live`

The Vercel frontend is healthy but the persistent worker is unreachable, timed out, or has no readable live state yet. Check the worker `/healthz` endpoint and the capture process.

### UI shows `STALE`

The site is reachable but `state.updated_at` is old. The UI deliberately freezes the displayed forecast rather than pretending stale data is live.

### No strict AI pace

Train and mount the strict artifact:

```bash
f1-laps-strict --year 2026 --race-count 8 --output reports/local/lap-strict
```

If the artifact is absent, the API clearly reports `fallback_recent_laps`; it does not invent a model prediction.
