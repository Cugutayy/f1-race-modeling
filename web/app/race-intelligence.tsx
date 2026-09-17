"use client";

import { useEffect, useMemo, useState } from "react";

type Driver = {
  driver_number: number;
  acronym?: string | null;
  full_name?: string | null;
  team_name?: string | null;
  position?: number | null;
  gap_to_leader_s?: number | null;
  interval_s?: number | null;
  compound?: string | null;
  tyre_age?: number | null;
  last_lap_s?: number | null;
  recent_laps_s?: number[];
  pit_stops?: number | null;
  speed_kmh?: number | null;
  x?: number | null;
  y?: number | null;
};

type RacePrediction = {
  driver_number: number;
  label: string;
  expected_position: number;
  win_probability: number;
  podium_probability: number;
  top10_probability: number;
  position_p10: number;
  position_p90: number;
  dnf_probability: number;
};

type PacePrediction = {
  driver_number: number | string;
  predicted_green_lap_s?: number | null;
  green_lap_lower_s?: number | null;
  green_lap_upper_s?: number | null;
  recent_median_5_s?: number | null;
  last_lap_s?: number | null;
  p_green?: number | null;
  p_pit?: number | null;
  p_neutralized?: number | null;
};

type LiveState = {
  session_key?: number | null;
  session_name?: string | null;
  current_lap?: number | null;
  updated_at?: string | null;
  received_messages?: number;
  flag?: string | null;
  safety_car?: string | null;
  weather?: {
    track_temperature_c?: number | null;
    air_temperature_c?: number | null;
    humidity_pct?: number | null;
    rainfall?: boolean | number | null;
  };
  drivers?: Driver[];
};

type LiveReport = {
  session_key?: number | null;
  current_lap?: number | null;
  total_laps: number;
  state_age_s?: number | null;
  pace_status?: string;
  state: LiveState;
  predictions?: RacePrediction[];
  pace_predictions?: PacePrediction[];
  pace_model?: Record<string, unknown>;
  strategy_prior_source?: Record<string, unknown>;
  audit?: Record<string, unknown>;
};

type TelemetrySample = {
  date?: string | null;
  speed_kmh?: number | null;
  throttle_pct?: number | null;
  brake?: number | null;
  rpm?: number | null;
  gear?: number | null;
  drs?: number | null;
};

type StrategyScenario = RacePrediction & {
  pit_in_laps: number;
  compound: string;
};

type StrategyReport = {
  scenarios: StrategyScenario[];
  pace_status?: string;
  state_age_s?: number | null;
};

type Tab = "race" | "pace" | "track" | "telemetry" | "strategy" | "audit";

const TABS: Array<{ key: Tab; label: string }> = [
  { key: "race", label: "Race" },
  { key: "pace", label: "AI Pace" },
  { key: "track", label: "Track" },
  { key: "telemetry", label: "Telemetry" },
  { key: "strategy", label: "Strategy Lab" },
  { key: "audit", label: "Model Audit" },
];

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function percent(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function seconds(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}s` : "—";
}

function driverLabel(driver: Driver): string {
  return driver.acronym || driver.full_name || `#${driver.driver_number}`;
}

function liveTone(age: number | null | undefined): "live" | "warn" | "down" {
  if (typeof age !== "number" || !Number.isFinite(age)) return "down";
  if (age <= 8) return "live";
  if (age <= 20) return "warn";
  return "down";
}

function Sparkline({ values }: { values: Array<number | null | undefined> }) {
  const clean = values.map((value, index) => ({ value: asNumber(value), index })).filter((row) => row.value !== null) as Array<{ value: number; index: number }>;
  if (clean.length < 2) return <div className="empty-mini">No samples</div>;
  const min = Math.min(...clean.map((row) => row.value));
  const max = Math.max(...clean.map((row) => row.value));
  const span = Math.max(max - min, 1e-6);
  const lastIndex = Math.max(values.length - 1, 1);
  const points = clean
    .map((row) => `${(row.index / lastIndex) * 100},${34 - ((row.value - min) / span) * 30}`)
    .join(" ");
  return (
    <svg className="sparkline" viewBox="0 0 100 38" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function TrackMap({ drivers }: { drivers: Driver[] }) {
  const points = drivers.filter((driver) => asNumber(driver.x) !== null && asNumber(driver.y) !== null);
  if (!points.length) return <div className="empty-state">Waiting for OpenF1 location samples.</div>;
  const xs = points.map((driver) => driver.x as number);
  const ys = points.map((driver) => driver.y as number);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const project = (driver: Driver) => ({
    x: 7 + (((driver.x as number) - minX) / spanX) * 86,
    y: 93 - (((driver.y as number) - minY) / spanY) * 86,
  });
  const sorted = [...points].sort((a, b) => (a.position ?? 999) - (b.position ?? 999));
  const path = sorted.map((driver, index) => {
    const p = project(driver);
    return `${index === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
  }).join(" ");
  return (
    <div className="track-shell">
      <svg viewBox="0 0 100 100" className="track-map" role="img" aria-label="Approximate live track positions">
        <path d={path} className="track-ghost" />
        {points.map((driver) => {
          const p = project(driver);
          return (
            <g key={driver.driver_number} transform={`translate(${p.x} ${p.y})`}>
              <circle r="2.2" className="car-dot" />
              <text x="3.2" y="1.25" className="car-label">{driverLabel(driver)}</text>
            </g>
          );
        })}
      </svg>
      <p className="footnote">Approximate public x/y position; not precision GPS or racing-line reconstruction.</p>
    </div>
  );
}

export default function RaceIntelligence() {
  const [tab, setTab] = useState<Tab>("race");
  const [totalLaps, setTotalLaps] = useState(57);
  const [samples, setSamples] = useState(4000);
  const [report, setReport] = useState<LiveReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedDriver, setSelectedDriver] = useState<number | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetrySample[]>([]);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<StrategyReport | null>(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategyError, setStrategyError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;

    const poll = async () => {
      controller = new AbortController();
      try {
        const response = await fetch(`/api/live?total_laps=${totalLaps}&samples=${samples}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.detail || `Live API returned ${response.status}`);
        if (active) {
          setReport(payload as LiveReport);
          setError(null);
        }
      } catch (reason) {
        if (active && !(reason instanceof DOMException && reason.name === "AbortError")) {
          setError(reason instanceof Error ? reason.message : "Live API request failed");
        }
      } finally {
        if (active) timer = setTimeout(poll, 2000);
      }
    };

    void poll();
    return () => {
      active = false;
      controller?.abort();
      if (timer) clearTimeout(timer);
    };
  }, [totalLaps, samples]);

  const drivers = useMemo(
    () => [...(report?.state?.drivers || [])].sort((a, b) => (a.position ?? 999) - (b.position ?? 999)),
    [report],
  );

  useEffect(() => {
    if (!drivers.length) return;
    if (selectedDriver === null || !drivers.some((driver) => driver.driver_number === selectedDriver)) {
      setSelectedDriver(drivers[0].driver_number);
    }
  }, [drivers, selectedDriver]);

  useEffect(() => {
    if (selectedDriver === null) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;

    const poll = async () => {
      controller = new AbortController();
      try {
        const response = await fetch(`/api/telemetry?driver_number=${selectedDriver}&limit=700`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.detail || `Telemetry API returned ${response.status}`);
        if (active) {
          setTelemetry(Array.isArray(payload?.samples) ? payload.samples : []);
          setTelemetryError(null);
        }
      } catch (reason) {
        if (active && !(reason instanceof DOMException && reason.name === "AbortError")) {
          setTelemetryError(reason instanceof Error ? reason.message : "Telemetry request failed");
        }
      } finally {
        if (active) timer = setTimeout(poll, 2500);
      }
    };

    void poll();
    return () => {
      active = false;
      controller?.abort();
      if (timer) clearTimeout(timer);
    };
  }, [selectedDriver]);

  const predictionByDriver = useMemo(
    () => new Map((report?.predictions || []).map((row) => [row.driver_number, row])),
    [report],
  );
  const paceByDriver = useMemo(
    () => new Map((report?.pace_predictions || []).map((row) => [Number(row.driver_number), row])),
    [report],
  );

  const selected = drivers.find((driver) => driver.driver_number === selectedDriver) || null;
  const age = report?.state_age_s;
  const tone = liveTone(age);
  const predictions = [...(report?.predictions || [])].sort((a, b) => b.win_probability - a.win_probability);
  const weather = report?.state?.weather || {};

  const runStrategy = async () => {
    if (selectedDriver === null) return;
    setStrategyLoading(true);
    setStrategyError(null);
    try {
      const response = await fetch(
        `/api/strategy?driver_number=${selectedDriver}&total_laps=${totalLaps}&samples=${samples}`,
        { cache: "no-store" },
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail || `Strategy API returned ${response.status}`);
      setStrategy(payload as StrategyReport);
    } catch (reason) {
      setStrategyError(reason instanceof Error ? reason.message : "Strategy request failed");
      setStrategy(null);
    } finally {
      setStrategyLoading(false);
    }
  };

  const bestScenario = useMemo(() => {
    const rows = strategy?.scenarios || [];
    return rows.length ? [...rows].sort((a, b) => a.expected_position - b.expected_position)[0] : null;
  }, [strategy]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">F1 / LIVE RACE INTELLIGENCE</div>
          <h1>Virtual Pit Wall</h1>
          <p className="subtitle">Public timing → strict pace model → coherent race simulation → strategy sensitivity.</p>
        </div>
        <div className="top-controls">
          <label>
            <span>Total laps</span>
            <input type="number" min={2} max={100} value={totalLaps} onChange={(event) => setTotalLaps(Math.min(100, Math.max(2, Number(event.target.value) || 2)))} />
          </label>
          <label>
            <span>Monte Carlo</span>
            <select value={samples} onChange={(event) => setSamples(Number(event.target.value))}>
              <option value={2000}>2k</option>
              <option value={4000}>4k</option>
              <option value={8000}>8k</option>
              <option value={12000}>12k</option>
              <option value={20000}>20k</option>
            </select>
          </label>
        </div>
      </header>

      <section className="status-strip">
        <span className={`status-pill ${tone}`}><i />{tone === "live" ? "LIVE" : tone === "warn" ? "DELAYED" : "STALE"}</span>
        <span>Session <strong>{report?.state?.session_name || report?.session_key || "—"}</strong></span>
        <span>Lap <strong>{report?.current_lap ?? "—"} / {totalLaps}</strong></span>
        <span>State age <strong>{typeof age === "number" ? `${age.toFixed(1)}s` : "—"}</strong></span>
        <span>Pace <strong>{report?.pace_status === "strict_model" ? "Strict AI" : "Recent-lap fallback"}</strong></span>
        <span>Messages <strong>{report?.state?.received_messages ?? "—"}</strong></span>
      </section>

      {error && <div className="banner error-banner"><strong>Live backend unavailable.</strong> {error}</div>}
      {!error && tone !== "live" && report && <div className="banner warn-banner">Predictions are frozen or delayed because the captured state is not fresh.</div>}

      <nav className="tabs" aria-label="Race intelligence sections">
        {TABS.map((item) => (
          <button key={item.key} className={tab === item.key ? "active" : ""} onClick={() => setTab(item.key)}>
            {item.label}
          </button>
        ))}
      </nav>

      {tab === "race" && (
        <section className="grid two-col">
          <div className="panel leaderboard-panel">
            <div className="panel-head"><div><span className="kicker">CURRENT ORDER</span><h2>Race board</h2></div><span className="panel-note">win / podium are Monte Carlo marginals</span></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>P</th><th>Driver</th><th>Tyre</th><th>Gap</th><th>Last</th><th>Win</th><th>Podium</th><th>Exp.</th></tr></thead>
                <tbody>
                  {drivers.map((driver) => {
                    const prediction = predictionByDriver.get(driver.driver_number);
                    return (
                      <tr key={driver.driver_number} className={driver.driver_number === selectedDriver ? "selected-row" : ""} onClick={() => setSelectedDriver(driver.driver_number)}>
                        <td className="position">{driver.position ?? "—"}</td>
                        <td><strong>{driverLabel(driver)}</strong><small>{driver.team_name || `#${driver.driver_number}`}</small></td>
                        <td><span className={`compound ${String(driver.compound || "UNK").toLowerCase()}`}>{driver.compound || "UNK"}</span><small>{driver.tyre_age != null ? `${driver.tyre_age}L` : ""}</small></td>
                        <td>{driver.position === 1 ? "LEADER" : seconds(driver.gap_to_leader_s, 1)}</td>
                        <td>{seconds(driver.last_lap_s)}</td>
                        <td>{percent(prediction?.win_probability)}</td>
                        <td>{percent(prediction?.podium_probability)}</td>
                        <td>{prediction ? prediction.expected_position.toFixed(2) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="stack">
            <div className="panel probability-panel">
              <div className="panel-head"><div><span className="kicker">OUTCOME DISTRIBUTION</span><h2>Win probability</h2></div></div>
              <div className="bars">
                {predictions.slice(0, 10).map((row) => (
                  <div className="bar-row" key={row.driver_number}>
                    <span>{row.label}</span>
                    <div className="bar-track"><i style={{ width: `${Math.max(0.8, row.win_probability * 100)}%` }} /></div>
                    <strong>{percent(row.win_probability)}</strong>
                  </div>
                ))}
                {!predictions.length && <div className="empty-state compact">Waiting for enough live pace + position observations.</div>}
              </div>
            </div>
            <div className="metric-grid">
              <div className="metric-card"><span>Track</span><strong>{weather.track_temperature_c != null ? `${weather.track_temperature_c}°C` : "—"}</strong></div>
              <div className="metric-card"><span>Air</span><strong>{weather.air_temperature_c != null ? `${weather.air_temperature_c}°C` : "—"}</strong></div>
              <div className="metric-card"><span>Rain</span><strong>{weather.rainfall ? "YES" : "NO"}</strong></div>
              <div className="metric-card"><span>Track state</span><strong>{report?.state?.safety_car || report?.state?.flag || "GREEN / ?"}</strong></div>
            </div>
          </div>
        </section>
      )}

      {tab === "pace" && (
        <section className="panel">
          <div className="panel-head"><div><span className="kicker">STRICT AS-OF MODEL</span><h2>Next green-lap pace</h2></div><span className="panel-note">historical stint fields excluded from strict model</span></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Driver</th><th>Model</th><th>Interval</th><th>5-lap median</th><th>Last</th><th>P(green)</th><th>P(pit)</th><th>P(neutralized)</th></tr></thead>
              <tbody>
                {drivers.map((driver) => {
                  const pace = paceByDriver.get(driver.driver_number);
                  return (
                    <tr key={driver.driver_number} onClick={() => setSelectedDriver(driver.driver_number)}>
                      <td><strong>{driverLabel(driver)}</strong><small>{driver.team_name || ""}</small></td>
                      <td className="accent-value">{seconds(pace?.predicted_green_lap_s)}</td>
                      <td>{pace?.green_lap_lower_s != null && pace?.green_lap_upper_s != null ? `${pace.green_lap_lower_s.toFixed(3)}–${pace.green_lap_upper_s.toFixed(3)}` : "—"}</td>
                      <td>{seconds(pace?.recent_median_5_s)}</td>
                      <td>{seconds(pace?.last_lap_s ?? driver.last_lap_s)}</td>
                      <td>{percent(pace?.p_green)}</td>
                      <td>{percent(pace?.p_pit)}</td>
                      <td>{percent(pace?.p_neutralized)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="footnote">Conformal interval is empirical coverage evidence, not a Gaussian confidence interval. If the strict artifact is missing, race simulation visibly falls back to robust recent-lap pace.</p>
        </section>
      )}

      {tab === "track" && (
        <section className="panel">
          <div className="panel-head"><div><span className="kicker">OPENF1 LOCATION</span><h2>Live field map</h2></div><span className="panel-note">relative public x/y samples</span></div>
          <TrackMap drivers={drivers} />
        </section>
      )}

      {tab === "telemetry" && (
        <section className="grid telemetry-grid">
          <div className="panel telemetry-driver">
            <div className="panel-head"><div><span className="kicker">CAR DATA</span><h2>{selected ? driverLabel(selected) : "Select driver"}</h2></div><span className="panel-note">public broadcast telemetry</span></div>
            <select className="driver-select" value={selectedDriver ?? ""} onChange={(event) => setSelectedDriver(Number(event.target.value))}>
              {drivers.map((driver) => <option key={driver.driver_number} value={driver.driver_number}>{driver.position ?? "—"}. {driverLabel(driver)} · {driver.team_name || ""}</option>)}
            </select>
            <div className="telemetry-now">
              <div><span>Speed</span><strong>{selected?.speed_kmh != null ? `${selected.speed_kmh.toFixed(0)} km/h` : "—"}</strong></div>
              <div><span>Compound</span><strong>{selected?.compound || "—"}</strong></div>
              <div><span>Tyre age</span><strong>{selected?.tyre_age != null ? `${selected.tyre_age} laps` : "—"}</strong></div>
              <div><span>Last lap</span><strong>{seconds(selected?.last_lap_s)}</strong></div>
            </div>
            {telemetryError && <div className="inline-error">{telemetryError}</div>}
          </div>
          <div className="telemetry-panels">
            {([
              ["Speed", "km/h", telemetry.map((row) => row.speed_kmh)],
              ["Throttle", "%", telemetry.map((row) => row.throttle_pct)],
              ["RPM", "rpm", telemetry.map((row) => row.rpm)],
              ["Gear", "", telemetry.map((row) => row.gear)],
            ] as Array<[string, string, Array<number | null | undefined>]>).map(([label, unit, values]) => {
              const latest = [...values].reverse().find((value) => asNumber(value) !== null);
              return (
                <div className="panel telemetry-card" key={label}>
                  <div className="telemetry-card-head"><span>{label}</span><strong>{latest != null ? `${Number(latest).toFixed(label === "Speed" || label === "RPM" ? 0 : 1)}${unit ? ` ${unit}` : ""}` : "—"}</strong></div>
                  <Sparkline values={values} />
                </div>
              );
            })}
          </div>
        </section>
      )}

      {tab === "strategy" && (
        <section className="grid two-col strategy-layout">
          <div className="panel">
            <div className="panel-head"><div><span className="kicker">COUNTERFACTUALS</span><h2>Pit-window lab</h2></div><span className="panel-note">common-seed scenario comparison</span></div>
            <label className="field-label">Driver</label>
            <select className="driver-select" value={selectedDriver ?? ""} onChange={(event) => { setSelectedDriver(Number(event.target.value)); setStrategy(null); }}>
              {drivers.map((driver) => <option key={driver.driver_number} value={driver.driver_number}>{driver.position ?? "—"}. {driverLabel(driver)} · {driver.compound || "?"}</option>)}
            </select>
            <button className="primary-button" onClick={() => void runStrategy()} disabled={strategyLoading || selectedDriver === null}>
              {strategyLoading ? "Simulating…" : "Run pit scenarios"}
            </button>
            {strategyError && <div className="inline-error">{strategyError}</div>}
            <p className="footnote">This is sensitivity analysis, not a claim that an unobserved alternate pit stop would causally produce the simulated result.</p>
          </div>
          <div className="panel">
            <div className="panel-head"><div><span className="kicker">BEST EXPECTED FINISH</span><h2>{bestScenario ? `Pit +${bestScenario.pit_in_laps} · ${bestScenario.compound}` : "No scenario yet"}</h2></div>{bestScenario && <strong className="hero-number">P{bestScenario.expected_position.toFixed(2)}</strong>}</div>
            {strategy?.scenarios?.length ? (
              <div className="table-wrap strategy-table"><table><thead><tr><th>Stop</th><th>Tyre</th><th>Exp.</th><th>Win</th><th>Podium</th><th>DNF</th></tr></thead><tbody>
                {[...strategy.scenarios].sort((a, b) => a.expected_position - b.expected_position).map((row) => (
                  <tr key={`${row.pit_in_laps}-${row.compound}`} className={row === bestScenario ? "selected-row" : ""}>
                    <td>+{row.pit_in_laps}</td><td><span className={`compound ${row.compound.toLowerCase()}`}>{row.compound}</span></td><td>{row.expected_position.toFixed(2)}</td><td>{percent(row.win_probability)}</td><td>{percent(row.podium_probability)}</td><td>{percent(row.dnf_probability)}</td>
                  </tr>
                ))}
              </tbody></table></div>
            ) : <div className="empty-state">Select a driver and run the common-seed Monte Carlo comparison.</div>}
          </div>
        </section>
      )}

      {tab === "audit" && (
        <section className="grid two-col">
          <div className="panel audit-panel"><span className="kicker">MODEL</span><h2>Strict pace provenance</h2><pre>{JSON.stringify(report?.pace_model || { status: report?.pace_status || "unknown" }, null, 2)}</pre></div>
          <div className="panel audit-panel"><span className="kicker">SIMULATION</span><h2>Public-data assumptions</h2><pre>{JSON.stringify({ state_age_s: report?.state_age_s, strategy_prior_source: report?.strategy_prior_source, audit: report?.audit }, null, 2)}</pre></div>
        </section>
      )}

      <footer>
        <span>Research interface · not betting odds</span>
        <span>OpenF1 public timing is not team telemetry</span>
        <span>{report?.state?.updated_at ? `State ${new Date(report.state.updated_at).toLocaleTimeString()}` : "No state timestamp"}</span>
      </footer>
    </main>
  );
}
