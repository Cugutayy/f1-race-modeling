# Data contract

One historical row represents a driver in a completed event. Required identifiers: `event_id`, `date`, `year`, `round`, `driver`, `team`, `circuit`. Qualifying: `quali_position`, `quali_seconds` (Q1 for collected data). Targets: `finish_position`, `points`, `dnf`. Collector adds `quali_time_basis=Q1` and `grid_position`; final grid is not a feature.

An event has unique driver IDs, contiguous classified positions 1..N and one winner. Invalid numerics, duplicates and incomplete classification sequences fail validation. Missing qualifying values remain missing until train-fitted preprocessing.

Inference entries use the same identifiers/date and qualifying fields. Targets are discarded. Results at or after the target event are excluded. Unknown drivers/teams use explicit historical priors and unknown-safe encoding.

Origin is retrospective after qualifying. Current APIs can include revised classifications and a post-event entrant set. Retrieval time does not prove historical publication time. A strict as-published system needs archived entry/qualifying snapshots.

`report.json` schema 1: `series`, `data_kind`, `run_id`, `forecast_origin`, `summary`, `metrics`, `predictions`, `provenance`, `runtime`, `limitations`, paired baseline differences and reliability bins. Inference JSON is not a measured backtest report.

Cache envelope: URL, retrieval UTC, canonical-payload SHA-256 and payload. Collection sidecars list source requests. Checksum mismatch fails; `--offline` requires all requested URLs already cached. Raw cache is excluded from Git.

Requirements-lock pins the exercised Python 3.12 environment; reports fingerprint source modules and runtime versions. A seed alone is not complete reproducibility.
