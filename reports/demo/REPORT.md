# F1 chronological research report

Run `2db016ad01f38185`; data: **synthetic**; 6 held-out events.

Forecast origin: post-qualifying retrospective event-cutoff; qualifying order, not final race grid.

Each event is held out as a whole. Training and temperature selection use strictly earlier events. The last calibration block is not used to fit the score learner. Baselines use that same calibration block.

```text
            model  position_mae  winner_log_loss  winner_brier  winner_accuracy  podium_recall
gradient_boosting        1.7000           1.7394        0.7949           0.5000         0.7222
 qualifying_order        1.4667           1.3622        0.6864           0.5000         0.7222
      recent_form        1.5333           1.4107        0.7565           0.3333         0.6667
       ridge_rank        1.6000           1.4927        0.7115           0.5000         0.7222
          uniform        3.7000           2.3026        0.9000           0.0000         0.2778
```

MAE, winner log loss and multiclass Brier sum: lower is better. Accuracy and podium recall: higher. Race bootstrap and paired log-loss differences are in report.json. Negative paired differences favor the named model over qualifying order. These are descriptive uncertainty estimates.

## Limitations

- Retrospective source snapshots lack historical publication/revision timestamps; not a strict as-published backtest.
- Result entrants define historical fields; withdrawals before race and later disqualifications may change membership.
- No target-race weather, race telemetry, final grid penalties or sprint points are predictive inputs.
- Qualifying gap uses Q1 where supplied by collector; wet/evolving Q1 conditions remain a confounder.
- Rank regression parameterizes a Plackett-Luce distribution; it is not a maximum-likelihood PL estimator.
- Temperature uses a small earlier holdout; calibrated is a procedure, not a claim of demonstrated reliability.
- Podium/top10/intervals use 4096 fixed-seed PL simulations; expected rank and winner probabilities are analytic.
- Uniform rank metrics use alphabetical tie order and are not meaningful as a ranking benchmark.
- Race bootstrap is descriptive and ignores serial dependence; no significance or prospective guarantee is claimed.
- Hyperparameters are fixed; tuning after seeing this report requires a new untouched holdout.
