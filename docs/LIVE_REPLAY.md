# Lap-by-lap forecasting: verified historical replay

The implemented task predicts a driver's **lap duration at the start of that lap**.
It then learns from the completed lap after an explicit simulated availability
delay. This is a functioning online-learning replay, not a verified live timing
service or a race-winner predictor.

## Reproduce the experiment

```bash
python -m f1_research.live_replay --output reports/local/replay --year 2024 --race-count 6 --latency-s 1
python -m pytest tests/test_live_replay.py -q
```

The command downloads the first six non-cancelled Race sessions in chronological
order. Responses are cached as exact bytes with source URL, retrieval time and
SHA-256. Repeated execution verifies each cached response hash. Predictions and
raw responses remain under the ignored local report directory; the repository
contains [aggregate evidence](../reports/replay-benchmark/REPORT.md).

## What was actually run

The experiment trains on Bahrain, Saudi Arabia, Australia and Japan 2024. China
2024 selects ridge regularization from `1, 10, 100, 1000` by validation MAE. The
chosen value is **1**. That value is frozen before replaying **Miami, 5 May 2024**.
This historical example is not the most recently completed Formula 1 race.

| Test model | MAE (seconds) | RMSE (seconds) | Median absolute error (seconds) |
|---|---:|---:|---:|
| Online ridge | 4.670 | 10.397 | 0.723 |
| Last observed lap | 4.693 | 11.309 | 0.372 |
| Recent five-lap median | 4.795 | 11.053 | 0.444 |

All models score the same **1,030 driver-lap forecasts**. Ridge reduces mean error
by only about **0.5%** relative to the last-lap baseline, while its median error
is worse. This single-race result does not establish a general improvement.
Pit and safety-car laps stay in the evaluation; we do not remove difficult
targets using their eventual durations. Errors therefore include abrupt pace
changes which this small model cannot anticipate.

The test session contains 1,112 source rows. Exact scored, unscored and warm-up
counts are retained in `report.json` under `test_coverage`. The model enters the
test with 4,572 learned labels and leaves with 5,602: every scored test label
updates the model **after** its forecast. This is prequential evaluation
(predict, observe the result, then learn), not a frozen-model test.

## Forecast and training cutoffs

1. A completed lap becomes observable at `date_start + lap_duration + latency`.
2. At least three previously observed laps for that driver are required.
3. Features use at most five observed laps: median-relative last/previous lap,
   recent trend and current lap number. Feature scaling is fixed in advance.
4. Ridge learns the residual from the recent median using cumulative sufficient
   statistics. It carries learning across sessions; driver lap histories reset.
5. At identical timestamps, completed-label events precede forecast events.
6. Features and training labels must be available no later than forecast time.
   Each forecast logs both cutoffs. The Miami run has **zero violations**.

Validation itself follows this same online-update policy. No Miami metric is
used to select alpha. Missing/nonfinite labels are never learned or scored.
No winner, standings, final race position, future stint or future weather
appears among these features.

Tests cover future-target mutation, delayed observations, exact online/batch
ridge equivalence, missing targets, invalid identities, nonfinite inputs,
future-trained-model rejection, input-order invariance and frozen-model mode.

## Provider facts checked directly

On 17 September 2026, the [OpenF1 provider documentation](https://openf1.org/docs/)
states that historical data from 2023 onward is free without authentication,
while real-time access requires a paid subscription. It identifies OpenF1 as an
unofficial project. Its lap schema defines `date_start` as approximate and
`lap_duration` in seconds. These are first-party statements from the **data
provider**, not confirmation or certification by Formula 1/FIA.

The downloaded historical endpoint responses verify that the data can be
processed, fitted and used for forecasts. They do **not** verify live
publication delays, delivery continuity or correction behavior.

## Most recently completed race check (17 September 2026)

```bash
python -m f1_research.live_replay --year 2026 --latest-completed --race-count 6 --output reports/local/replay-latest
```

This mode refreshes the provider's session list and selects the last six
non-cancelled races whose `date_end` is strictly before the recorded UTC cutoff.
The source returned **Madrid, Spain, 13 September 2026**, session **11369**, as
the last eligible race. This is a source-based scheduled-end check, not a
guarantee of an official final classification. Its downloaded lap records were
nonempty and successfully processed.

Training: United Kingdom, Belgium, Hungary and Netherlands 2026. Validation:
Italy, 6 September. The four-value alpha search selected **1,000**, with Italy
MAE **9.700 seconds**. The reserved Madrid race is not used in that selection.
The fresh run and source hashes are recorded in the separate
[latest-race evidence](../reports/replay-latest-benchmark/REPORT.md).

| Madrid test model | MAE (seconds) | RMSE (seconds) | Median absolute error (seconds) |
|---|---:|---:|---:|
| Online ridge | 2.398 | 6.362 | 0.907 |
| Last observed lap | 3.382 | 8.889 | 0.427 |
| Recent five-lap median | 2.216 | 6.310 | 0.602 |

**The recent median beats the tuned ridge on test MAE.** Tuning is real, but does
not guarantee improvement. The held-out result is retained without choosing a
new alpha using Madrid outcomes. Last-lap persistence still has the best median
error; these models respond differently to occasional large pace changes.

The run processed 1,109 source rows, emitted 1,021 forecasts, scored 1,017 and
left four targets unscored because their durations were unavailable/invalid.
Eighty-eight rows had no forecast because of warm-up or missing start time.
Training label count grew **5,399 to 6,416**, and all three cutoff-violation
counts were zero. This validates download → normalization → tuning → online
learning → forecasting → scoring on that historical source.

The new run exposed mixed whole-second/fractional-second ISO timestamps in the
provider data. Parsing now explicitly accepts ISO8601 instead of silently
dropping valid rows based on the first timestamp's precision. A regression
test covers this case, and the 2024 study was also rerun with the correction.

## What remains before a real live deployment

- A permitted live source and ingestion adapter with actual receive timestamps.
- An append-only event log handling duplicates, out-of-order delivery, updates
  and reconnects without learning a revised label twice.
- Start-of-lap detection without looking ahead at historical lap identities.
- Saved model state and reproducible checkpoint recovery across restarts.
- A live shadow run measuring availability lag, missing forecasts and latency.
- More chronologically reserved races, circuit/condition breakdowns and
  uncertainty estimates that account for correlated laps and shared incidents.

The assumed one-second delay is not a measured service guarantee. Historical
records may have been corrected after the race; this replay cannot reconstruct
what the API actually published at each original instant. Until a live event
log passes those checks, claims of proven live performance are unsupported.
