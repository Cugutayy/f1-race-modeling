# Historical next-lap replay

Test: United States 2024
Validation: China; selected alpha: 1

| Model | MAE seconds | RMSE seconds | Scored laps |
|---|---:|---:|---:|
| ridge | 4.670 | 10.397 | 1030 |
| last_lap | 4.693 | 11.309 | 1030 |
| recent_median | 4.795 | 11.053 | 1030 |

## Limits

- Historical event-time simulation, not verified live data ingestion or service latency.
- OpenF1 is an unofficial F1 source; provider documentation and API verified directly.
- date_start is approximate; available = start + duration + simulated latency, not publication timestamp.
- Free historical API; live access requires paid subscription per provider documentation.
- Next-lap duration only; no winner, pit strategy, DNF or full-race outcome prediction.
- Pit and safety-car laps retained in scoring; absent/invalid durations unscored, three observed laps required.
- Ridge updates after each observed training label, including earlier test-race labels; hyperparameters remain fixed.
- One reserved race is a development evaluation, not a sealed external benchmark; circuits and conditions shift.
- Start events replay historical lap identities; live start detection and corrections still require an ingestion adapter.
