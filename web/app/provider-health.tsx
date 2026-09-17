"use client";

import { useEffect, useState } from "react";

type HealthPayload = {
  ok?: boolean;
  connection_state?: string | null;
  live_stream_healthy?: boolean;
  last_message_age_s?: number | null;
  state_age_s?: number | null;
  connect_count?: number | null;
  disconnect_count?: number | null;
  last_stream_error?: string | null;
};

function tone(payload: HealthPayload | null, error: string | null) {
  if (error || !payload) return { label: "PROVIDER UNKNOWN", border: "rgba(255,81,104,.45)", bg: "rgba(255,81,104,.12)", dot: "#ff5168" };
  if (payload.live_stream_healthy) return { label: "PROVIDER LIVE", border: "rgba(56,224,143,.42)", bg: "rgba(56,224,143,.10)", dot: "#38e08f" };
  if (payload.connection_state === "connecting" || payload.connection_state === "reconnecting") {
    return { label: "PROVIDER RECONNECTING", border: "rgba(255,189,61,.42)", bg: "rgba(255,189,61,.10)", dot: "#ffbd3d" };
  }
  return { label: "PROVIDER STALE", border: "rgba(255,81,104,.45)", bg: "rgba(255,81,104,.12)", dot: "#ff5168" };
}

export default function ProviderHealth() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;

    const poll = async () => {
      controller = new AbortController();
      try {
        const response = await fetch("/api/health", { cache: "no-store", signal: controller.signal });
        const payload = (await response.json()) as HealthPayload & { detail?: string };
        if (!response.ok) throw new Error(payload.detail || `Health API returned ${response.status}`);
        if (active) {
          setHealth(payload);
          setError(null);
        }
      } catch (reason) {
        if (active && !(reason instanceof DOMException && reason.name === "AbortError")) {
          setError(reason instanceof Error ? reason.message : "Provider health request failed");
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
  }, []);

  const current = tone(health, error);
  const details = error
    ? error
    : `${health?.connection_state || "unknown"} · provider ${typeof health?.last_message_age_s === "number" ? `${health.last_message_age_s.toFixed(1)}s` : "—"} · state ${typeof health?.state_age_s === "number" ? `${health.state_age_s.toFixed(1)}s` : "—"}`;

  return (
    <aside
      role="status"
      aria-live="polite"
      style={{
        position: "fixed",
        right: 14,
        bottom: 14,
        zIndex: 100,
        maxWidth: "min(430px, calc(100vw - 28px))",
        border: `1px solid ${current.border}`,
        background: current.bg,
        backdropFilter: "blur(16px)",
        borderRadius: 13,
        padding: "10px 12px",
        boxShadow: "0 18px 50px rgba(0,0,0,.35)",
        color: "#f5f5f5",
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, fontWeight: 850, letterSpacing: ".08em" }}>
        <span style={{ width: 8, height: 8, borderRadius: 999, background: current.dot, boxShadow: `0 0 12px ${current.dot}` }} />
        {current.label}
      </div>
      <div style={{ marginTop: 5, color: "#a5aab4", fontSize: 10, lineHeight: 1.45 }}>
        {details}
        {health?.last_stream_error ? ` · ${health.last_stream_error}` : ""}
      </div>
    </aside>
  );
}
