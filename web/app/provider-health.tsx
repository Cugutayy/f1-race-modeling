"use client";

import { useEffect, useMemo, useState } from "react";

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

type ProviderTone = "live" | "reconnecting" | "stale" | "unknown";

function providerTone(payload: HealthPayload | null, error: string | null): ProviderTone {
  if (error || !payload) return "unknown";
  if (payload.live_stream_healthy) return "live";
  if (payload.connection_state === "connecting" || payload.connection_state === "reconnecting") {
    return "reconnecting";
  }
  return "stale";
}

function formatAge(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}s` : "—";
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
        const response = await fetch("/api/health", {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload = (await response.json()) as HealthPayload & { detail?: string };
        if (!response.ok) {
          throw new Error(payload.detail || `Health API returned ${response.status}`);
        }
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

  const tone = providerTone(health, error);

  useEffect(() => {
    document.documentElement.dataset.providerHealth = tone;
    return () => {
      delete document.documentElement.dataset.providerHealth;
    };
  }, [tone]);

  const detail = useMemo(() => {
    if (error) return error;
    const connection = health?.connection_state || "unknown";
    return `${connection} · provider ${formatAge(health?.last_message_age_s)} · state ${formatAge(health?.state_age_s)}`;
  }, [error, health]);

  const presentation = {
    live: { label: "PROVIDER LIVE", border: "rgba(56,224,143,.42)", bg: "rgba(56,224,143,.10)", dot: "#38e08f" },
    reconnecting: { label: "PROVIDER RECONNECTING", border: "rgba(255,189,61,.42)", bg: "rgba(255,189,61,.10)", dot: "#ffbd3d" },
    stale: { label: "PROVIDER STALE", border: "rgba(255,81,104,.45)", bg: "rgba(255,81,104,.12)", dot: "#ff5168" },
    unknown: { label: "PROVIDER UNKNOWN", border: "rgba(255,81,104,.45)", bg: "rgba(255,81,104,.12)", dot: "#ff5168" },
  }[tone];

  return (
    <>
      <style>{`
        html[data-provider-health="reconnecting"] .status-pill,
        html[data-provider-health="stale"] .status-pill,
        html[data-provider-health="unknown"] .status-pill {
          font-size: 0 !important;
        }
        html[data-provider-health="reconnecting"] .status-pill::after,
        html[data-provider-health="stale"] .status-pill::after,
        html[data-provider-health="unknown"] .status-pill::after {
          font-size: 12px;
          font-weight: 800;
          letter-spacing: .08em;
        }
        html[data-provider-health="reconnecting"] .status-pill::after { content: "DELAYED"; }
        html[data-provider-health="stale"] .status-pill::after,
        html[data-provider-health="unknown"] .status-pill::after { content: "STALE"; }
        html[data-provider-health="reconnecting"] .status-pill {
          color: var(--amber) !important;
          border-color: rgba(255,189,61,.26) !important;
          background: rgba(255,189,61,.07) !important;
        }
        html[data-provider-health="stale"] .status-pill,
        html[data-provider-health="unknown"] .status-pill {
          color: var(--red) !important;
          border-color: rgba(255,81,104,.26) !important;
          background: rgba(255,81,104,.07) !important;
        }
      `}</style>
      {tone !== "live" && (
        <aside
          role="status"
          aria-live="polite"
          style={{
            position: "fixed",
            right: 14,
            bottom: 14,
            zIndex: 100,
            maxWidth: "min(430px, calc(100vw - 28px))",
            border: `1px solid ${presentation.border}`,
            background: presentation.bg,
            backdropFilter: "blur(16px)",
            borderRadius: 13,
            padding: "10px 12px",
            boxShadow: "0 18px 50px rgba(0,0,0,.35)",
            color: "#f5f5f5",
            fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, fontWeight: 850, letterSpacing: ".08em" }}>
            <span style={{ width: 8, height: 8, borderRadius: 999, background: presentation.dot, boxShadow: `0 0 12px ${presentation.dot}` }} />
            {presentation.label}
          </div>
          <div style={{ marginTop: 5, color: "#a5aab4", fontSize: 10, lineHeight: 1.45 }}>
            {detail}
            {health?.last_stream_error ? ` · ${health.last_stream_error}` : ""}
          </div>
        </aside>
      )}
    </>
  );
}
