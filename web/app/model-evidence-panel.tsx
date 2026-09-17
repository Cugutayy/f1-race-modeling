"use client";

import { useEffect, useMemo, useState } from "react";

import styles from "./model-evidence.module.css";

type ModelMetrics = {
  model: string;
  position_mae?: number | null;
  winner_log_loss?: number | null;
  winner_brier?: number | null;
  winner_accuracy?: number | null;
  podium_recall?: number | null;
};

type PairedMetric = {
  mean_difference_model_minus_baseline?: number | null;
  interval_95?: [number | null, number | null] | null;
  events?: number | null;
  model_better_events?: number | null;
  baseline_better_events?: number | null;
  ties?: number | null;
  bootstrap_fraction_favorable?: number | null;
  direction?: string | null;
};

type Evidence = {
  schema_version: number;
  evidence_kind: string;
  benchmark_run_id?: string | null;
  provider?: string | null;
  years?: number[] | null;
  protocol?: string | null;
  sealed_test_events: number;
  selected_modern?: { name?: string; params?: Record<string, unknown> } | null;
  ensemble_weights?: Record<string, number | null> | null;
  models: ModelMetrics[];
  uncertainty?: {
    baseline?: string;
    bootstrap_samples?: number | null;
    paired_vs_baseline?: Record<string, Record<string, PairedMetric>>;
  } | null;
  limitations?: string[];
};

function number(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function modelLabel(name: string): string {
  if (name === "rank_ensemble") return "Rank ensemble";
  if (name === "qualifying_order") return "Qualifying only";
  if (name === "ridge_rank") return "Ridge rank";
  if (name === "plackett_luce_mle") return "Direct PL";
  if (name.startsWith("modern::")) return name.replace("modern::", "");
  return name;
}

function intervalText(metric: PairedMetric | undefined): string {
  const interval = metric?.interval_95;
  if (!interval || interval[0] == null || interval[1] == null) return "No bootstrap interval";
  return `[${Number(interval[0]).toFixed(3)}, ${Number(interval[1]).toFixed(3)}]`;
}

function crossesZero(metric: PairedMetric | undefined): boolean | null {
  const interval = metric?.interval_95;
  if (!interval || interval[0] == null || interval[1] == null) return null;
  return Number(interval[0]) <= 0 && Number(interval[1]) >= 0;
}

export default function ModelEvidencePanel() {
  const [open, setOpen] = useState(false);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const response = await fetch("/api/evidence", { cache: "no-store" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.detail || `Evidence backend returned HTTP ${response.status}`);
      }
      const payload = (await response.json()) as Evidence;
      setEvidence(payload);
      setError(null);
    } catch (reason) {
      setEvidence(null);
      setError(reason instanceof Error ? reason.message : "Model evidence unavailable");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const modelByName = useMemo(
    () => new Map((evidence?.models || []).map((row) => [row.model, row])),
    [evidence],
  );
  const ensemble = modelByName.get("rank_ensemble");
  const qualifying = modelByName.get("qualifying_order");
  const modernName = evidence?.selected_modern?.name;
  const modern = modernName ? modelByName.get(`modern::${modernName}`) : undefined;
  const paired = evidence?.uncertainty?.paired_vs_baseline?.rank_ensemble;
  const logLossDelta = paired?.winner_log_loss;
  const maeDelta = paired?.position_mae;
  const logLossCrossesZero = crossesZero(logLossDelta);

  return (
    <>
      <button
        type="button"
        className={styles.trigger}
        onClick={() => setOpen(true)}
        aria-label="Open sealed model evidence"
      >
        <span className={styles.triggerDot} />
        Evidence
        {evidence && <strong>{evidence.sealed_test_events} races</strong>}
      </button>

      {open && (
        <div className={styles.backdrop} role="presentation" onMouseDown={() => setOpen(false)}>
          <aside
            className={styles.drawer}
            role="dialog"
            aria-modal="true"
            aria-label="Model evidence"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className={styles.header}>
              <div>
                <span className={styles.kicker}>SEALED HISTORICAL EVIDENCE</span>
                <h2>Model Evidence</h2>
              </div>
              <button className={styles.close} type="button" onClick={() => setOpen(false)} aria-label="Close">×</button>
            </div>

            {loading && <div className={styles.empty}>Loading benchmark evidence…</div>}
            {!loading && error && (
              <div className={styles.warning}>
                <strong>Evidence artifact not installed.</strong>
                <span>{error}</span>
                <button type="button" onClick={() => void load()}>Retry</button>
              </div>
            )}

            {!loading && evidence && (
              <div className={styles.content}>
                <div className={styles.metaStrip}>
                  <span>Provider <strong>{evidence.provider || "—"}</strong></span>
                  <span>Years <strong>{evidence.years?.join("–") || "—"}</strong></span>
                  <span>Sealed test <strong>{evidence.sealed_test_events} races</strong></span>
                </div>

                <div className={styles.heroGrid}>
                  <div className={styles.heroCard}>
                    <span>Winner log-loss</span>
                    <strong>{number(ensemble?.winner_log_loss)}</strong>
                    <small>Ensemble · lower is better</small>
                  </div>
                  <div className={styles.heroCard}>
                    <span>Qualifying baseline</span>
                    <strong>{number(qualifying?.winner_log_loss)}</strong>
                    <small>Same sealed races</small>
                  </div>
                  <div className={styles.heroCard}>
                    <span>Finish MAE</span>
                    <strong>{number(ensemble?.position_mae)}</strong>
                    <small>Ensemble positions</small>
                  </div>
                  <div className={styles.heroCard}>
                    <span>Qualifying MAE</span>
                    <strong>{number(qualifying?.position_mae)}</strong>
                    <small>Do not hide the stronger baseline</small>
                  </div>
                </div>

                <section className={styles.section}>
                  <div className={styles.sectionHead}>
                    <div><span>SELECTION</span><h3>What the tuning block chose</h3></div>
                    <small>{evidence.selected_modern?.name || "No modern selection"}</small>
                  </div>
                  <div className={styles.weights}>
                    {Object.entries(evidence.ensemble_weights || {}).map(([name, weight]) => (
                      <div key={name}>
                        <span>{modelLabel(name)}</span>
                        <div className={styles.weightTrack}><i style={{ width: `${Math.max(0, Number(weight || 0) * 100)}%` }} /></div>
                        <strong>{percent(weight)}</strong>
                      </div>
                    ))}
                  </div>
                  {modern && (
                    <p className={styles.note}>
                      Selected modern challenger: <strong>{modelLabel(modern.model)}</strong> · winner log-loss {number(modern.winner_log_loss)} · finish MAE {number(modern.position_mae)}.
                    </p>
                  )}
                </section>

                <section className={styles.section}>
                  <div className={styles.sectionHead}>
                    <div><span>WHOLE-RACE BOOTSTRAP</span><h3>How stable is the advantage?</h3></div>
                    <small>{evidence.uncertainty?.bootstrap_samples?.toLocaleString() || "—"} resamples</small>
                  </div>
                  {logLossDelta ? (
                    <div className={styles.comparisonGrid}>
                      <div>
                        <span>Log-loss Δ vs qualifying</span>
                        <strong>{number(logLossDelta.mean_difference_model_minus_baseline)}</strong>
                        <small>95% {intervalText(logLossDelta)}</small>
                      </div>
                      <div>
                        <span>Finish-MAE Δ</span>
                        <strong>{number(maeDelta?.mean_difference_model_minus_baseline)}</strong>
                        <small>95% {intervalText(maeDelta)}</small>
                      </div>
                      <div>
                        <span>Race wins</span>
                        <strong>{logLossDelta.model_better_events ?? "—"}/{logLossDelta.events ?? "—"}</strong>
                        <small>lower log-loss than baseline</small>
                      </div>
                      <div>
                        <span>Bootstrap favorable</span>
                        <strong>{percent(logLossDelta.bootstrap_fraction_favorable)}</strong>
                        <small>descriptive, not a p-value</small>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.empty}>Bootstrap uncertainty was not packaged with this benchmark.</div>
                  )}
                  {logLossCrossesZero === true && (
                    <div className={styles.caution}>The 95% event-bootstrap interval crosses zero. The sealed improvement is promising, but this test block alone does not establish a stable universal advantage.</div>
                  )}
                  {logLossCrossesZero === false && (
                    <div className={styles.good}>The reported 95% event-bootstrap interval does not cross zero on this held-out block. It is still retrospective evidence, not a prospective guarantee.</div>
                  )}
                </section>

                <section className={styles.section}>
                  <div className={styles.sectionHead}><div><span>ALL CHALLENGERS</span><h3>Same sealed races</h3></div></div>
                  <div className={styles.tableWrap}>
                    <table>
                      <thead><tr><th>Model</th><th>Log-loss</th><th>MAE</th><th>Win acc.</th><th>Podium recall</th></tr></thead>
                      <tbody>
                        {evidence.models.map((row) => (
                          <tr key={row.model}>
                            <td><strong>{modelLabel(row.model)}</strong></td>
                            <td>{number(row.winner_log_loss)}</td>
                            <td>{number(row.position_mae)}</td>
                            <td>{percent(row.winner_accuracy)}</td>
                            <td>{percent(row.podium_recall)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                <div className={styles.limitations}>
                  {(evidence.limitations || []).map((item) => <p key={item}>{item}</p>)}
                  {evidence.protocol && <p><strong>Protocol:</strong> {evidence.protocol}</p>}
                  {evidence.benchmark_run_id && <p><strong>Run:</strong> {evidence.benchmark_run_id}</p>}
                </div>
              </div>
            )}
          </aside>
        </div>
      )}
    </>
  );
}
