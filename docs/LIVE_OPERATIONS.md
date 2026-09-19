# Live Race Intelligence — Operations

This document describes the production-shaped path. Streamlit apps remain useful for
research/debugging, but the live deployment is intentionally split:

```text
OpenF1 REST bootstrap + MQTT stream
        ↓
persistent Python capture worker
        ↓
events.jsonl + state.json + manifest.json
        ↓
strict pace model + calibrated strategy priors
        ↓
FastAPI read-only gateway
        ↓
Vercel / Next.js web UI
```

The Vercel process does **not** own the MQTT connection, training jobs, raw provider
files or model credentials.

## 1. Install

Core research environment:

```bash
python -m pip install -e ".[dev,live,api]"
```

Optional modern tabular challengers:

```bash
python -m pip install -e ".[modern]"
```

Optional local tabular foundation challenger:

```bash
python -m pip install -e ".[foundation]"
```

`TabICLv2` is a challenger only. It must win the same chronological benchmark as every
other model before its result is treated as useful for this project.

## 2. Verify the provider contract first

Before training or a live session, verify that the provider still exposes the fields
our adapters expect:

```bash
f1-openf1-doctor --session-key latest --output reports/local/openf1-doctor.json
```

To inspect returned car-data timestamp spacing for one driver:

```bash
f1-openf1-doctor \
  --session-key latest \
  --include-telemetry \
  --telemetry-driver 1 \
  --output reports/local/openf1-doctor.json
```

The telemetry cadence in this report is an observation from returned samples, not a
provider SLA or a promise about end-to-end live latency.

## 3. Train strict next-lap intelligence and strategy priors

Historical collection stores immutable raw endpoint snapshots, including `intervals`
for retrospective traffic calibration. Compound/stint fields remain excluded from the
strict evidence-bearing next-lap model because historical stint publication time is
not available.

Raw completed laps and model pace history are deliberately different objects. Raw laps
remain unchanged in source snapshots and capture/replay evidence. The strict pace
buffer excludes known pit/pit-out and active neutralization laps, resets across a
known rainfall-state transition, and rejects only extreme *slow* restart/neutralization
outliers relative to already-observed clean pace. The same policy is used by historical
feature construction and the live `RaceStateStore`; new live snapshots expose the
filtered values separately as `pace_laps_s`. If a pit event arrives after its lap row,
that lap is removed from the model pace buffer without deleting provider truth.

```bash
f1-laps-strict \
  --year 2026 \
  --race-count 8 \
  --output reports/local/lap-strict
```

Artifacts used by live inference:

```text
reports/local/lap-strict/next_lap_strict.joblib
reports/local/lap-strict/strategy_priors.json
reports/local/lap-strict/strict_release_manifest.json
```

`strategy_priors.json` contains separately audited public-data priors for pit loss,
Safety Car frequency, reliability, tyres and close-following traffic. These are not
team-private engineering signals.

`strict_release_manifest.json` binds the exact strict model bytes, strategy-prior bytes,
feature schema, Git revision, calibration/test session identities and per-session
OpenF1 source-content hashes. Cache paths and cache-hit status are recorded for
operations but are not allowed to change the content identity when the raw bytes are
identical.

For an immutable CI evidence bundle, run the **Strict live pace evidence** workflow.
It rebuilds the latest completed-race release, verifies the manifest/hash/conformal
contract and uploads the model, priors, report, manifests and raw source snapshots as
one GitHub Actions artifact. The same workflow also runs on the explicit
`.release-trigger` release marker.

## 4. Compare modern and local foundation models correctly

The evidence-bearing pre-race benchmark is chronological and disjoint:

```text
fit → tuning → probability calibration → sealed test
```

Run modern tree challengers:

```bash
f1-research benchmark-v2 \
  --input data/processed/history.csv \
  --output reports/local/benchmark-v2 \
  --models hist_gradient_boosting extra_trees xgboost lightgbm catboost
```

Include the optional local foundation challenger when installed:

```bash
f1-research benchmark-v2 \
  --input data/processed/history.csv \
  --output reports/local/benchmark-foundation \
  --models hist_gradient_boosting extra_trees tabicl_v2
```

The benchmark also reports a rank ensemble challenger. Ensemble weights are chosen
only on the tuning block; its final probability temperature is re-estimated on the
later calibration block. Sealed test results never tune weights, hyperparameters or
temperature.

Add whole-race bootstrap uncertainty to the sealed result:

```bash
python -m f1_research.benchmark_evidence \
  --metrics reports/local/benchmark-v2/fold_metrics.csv \
  --output reports/local/benchmark-v2/evidence \
  --samples 10000
```

The standalone `benchmark-v2` output is analysis evidence, not a deployable model
release. A compact live evidence file is produced only from a sealed release directory
that contains the matching `model.joblib` and `model_manifest.json`:

```bash
python -m f1_research.model_evidence \
  --benchmark-dir reports/model-release/release/benchmark \
  --output reports/local/model_evidence.json
```

`model_evidence.json` binds the displayed sealed metrics to the exact release Git SHA,
model byte SHA, model-manifest SHA, feature schema, training snapshot and calibration
hashes. If the manifest does not match the benchmark run or the model bytes do not match
the manifest, evidence generation fails closed. Missing uncertainty remains unavailable
rather than being inferred.

## 5. Live capture

Configure OpenF1 live credentials using environment variables. Do not commit them.
Depending on the account/provider setup, use a token or username/password:

```bash
export OPENF1_TOKEN="..."
# or
export OPENF1_USERNAME="..."
export OPENF1_PASSWORD="..."
```

Start capture:

```bash
f1-live capture --session-key latest --output reports/local/live
```

Files:

```text
reports/local/live/events.jsonl   immutable replay log
reports/local/live/state.json     atomic canonical state snapshot
reports/local/live/manifest.json  hashes + stream/reconnect/freshness health
```

The capture writer hashes JSONL incrementally, batches multi-row MQTT payloads and
records connection/reconnect state. REST bootstrap rows are canonicalized before both
capture and state mutation, so the JSONL replay path can reproduce the same semantic
state.

Historical/recovery-only bootstrap without MQTT:

```bash
f1-live capture --session-key latest --output reports/local/live --no-stream
```

Replay a saved capture without network access:

```bash
f1-live replay \
  --input reports/local/live/events.jsonl \
  --output reports/local/replay
```

## 6. Start the read-only API

```bash
export F1_API_TOKEN="a-long-random-server-token"
export F1_LIVE_STATE_PATH="$PWD/reports/local/live/state.json"
export F1_LIVE_EVENTS_PATH="$PWD/reports/local/live/events.jsonl"
export F1_LIVE_MANIFEST_PATH="$PWD/reports/local/live/manifest.json"
export F1_STRICT_MODEL_PATH="$PWD/reports/local/lap-strict/next_lap_strict.joblib"
export F1_STRATEGY_PRIORS_PATH="$PWD/reports/local/lap-strict/strategy_priors.json"
export F1_STRICT_RELEASE_MANIFEST_PATH="$PWD/reports/local/lap-strict/strict_release_manifest.json"
export F1_MODEL_EVIDENCE_PATH="$PWD/reports/local/model_evidence.json"

f1-api
```

Useful endpoints:

```text
GET /healthz
GET /v1/evidence
GET /v1/live?total_laps=57&samples=4000
GET /v1/telemetry?driver_number=1&limit=500
GET /v1/strategy?driver_number=1&total_laps=57&samples=4000
```

`/v1/evidence` serves only a validated schema-v1 compact sealed-benchmark artifact. A
missing artifact returns 404; malformed or unsupported evidence returns 503 rather than
being shown as valid model proof.

The API also verifies the strict live release contract before the model is used. By default,
`next_lap_strict.joblib` and `strategy_priors.json` must match the SHA-256 values in
`strict_release_manifest.json`; the feature schema, calibration/test split and
50/80/90/95 conformal radii must match the loaded model. The only bypass is the explicit
research override `F1_ALLOW_UNVERIFIED_STRICT_MODEL=1`, which is reported as
`production_eligible=false`.

`/healthz` separates provider transport health from canonical-state freshness. Watch:

- `connection_state`
- `live_stream_healthy`
- `last_message_age_s`
- `state_age_s`
- `connect_count` / `disconnect_count`
- `last_stream_error`
- `model_evidence`
- rejected stale/provider-order message counters

A connected MQTT socket with stale messages is not reported as a healthy live stream.

## 7. Run capture + API in Docker

Create a local `.env` that is **not committed**:

```text
OPENF1_TOKEN=...
F1_API_TOKEN=...
F1_API_PORT=8000
```

Then:

```bash
docker compose -f compose.live.yaml up --build -d
```

The two services share a persistent `live-data` volume. The API mounts only the
strict live artifacts from `reports/local/lap-strict` at `/models/lap-strict`, so the
mount cannot shadow the verified race-outcome evidence bundled into the worker image
at `/models/model_evidence.json`. Keep `next_lap_strict.joblib`,
`strategy_priors.json` and `strict_release_manifest.json` in the local strict-artifact
directory. Rebuild the worker image when the tracked
`reports/local/model_evidence.json` changes.

## 8. Vercel / Next.js

Deploy the `web/` directory as the Vercel project root. Configure server-side env vars:

```text
F1_BACKEND_URL=https://<persistent-python-worker>
F1_BACKEND_TOKEN=<same value as F1_API_TOKEN>
```

The token is used only by Vercel route handlers and must never be exposed as a public
`NEXT_PUBLIC_*` variable. The browser requests `/api/evidence`; the Vercel route handler
proxies it server-side to `/v1/evidence` without exposing the backend bearer token.
Historical model evidence is fetched once on page load rather than joined to the
2-second live polling loop.

## 9. Live evidence rules

The UI should never silently promote a fallback into “AI live prediction”:

- if the strict artifact is missing, pace uses the recent-lap fallback and says so;
- if strategy priors are missing, simulation uses explicit built-in defaults and says so;
- if the capture is stale or MQTT is reconnecting, the UI should show stale/delayed;
- if the sealed evidence artifact is missing, the Evidence drawer says it is not installed;
- retrospective tyre/traffic calibration is not presented as team telemetry or causal tyre/aero physics;
- sealed benchmark metrics remain separate from tuning and calibration metrics;
- a bootstrap interval crossing zero is shown as uncertainty, not hidden behind the point estimate.
