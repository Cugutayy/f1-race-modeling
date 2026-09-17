# Model card — research v1

Use: retrospective post-qualifying F1 outcome distributions. Not established: prospective 2026 skill, profitability, live strategy optimization or strict historical publication-time replay.

Ridge(alpha=10) and 80-estimator depth-2 Huber gradient boosting predict normalized finishing rank. Team/circuit encoding and numeric transformations fit only on earlier fitting data. Form and points update after whole events. Four preceding races select temperature on winner log loss; no score-model refit follows calibration. PL sampling produces coherent outcome marginals; this is rank-regression parameterization, not PL-likelihood training.

Real benchmark: current Jolpica snapshots, 2022–2025, 92 events/1,838 rows; 70 chronological test events after 22 warm-up races. These folds are development evidence, not a sealed final-season model-selection test. Scores macro-average by event. Paired bootstrap resamples races and ignores serial dependence.

Ridge improves mean probability scores in this study. Qualifying order more often places the winner first and has slightly smaller rank MAE. Do not replace this tradeoff with a universal accuracy claim.

Limits: revised classifications, post-event entrant membership, Q1 weather evolution, no sprint points or grid penalties, small calibration blocks, regulation/team drift and confounded driver/team effects. DNF is a historical binary summary, not a hazard model. Telemetry is a separate retrospective product; its missing fields and rejected traces remain visible.

Before stronger claims: archive pre-race snapshots, pre-register the next evaluation period, add practice features through ablation and save forecasts before races occur. Strategy, physical tyre degradation and fuel modeling are not implemented.
