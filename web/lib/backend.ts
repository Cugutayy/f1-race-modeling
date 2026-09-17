import { NextResponse } from "next/server";

const DEFAULT_TIMEOUT_MS = 6000;

function backendBaseUrl(): string {
  const value = process.env.F1_BACKEND_URL?.trim();
  if (!value) {
    throw new Error("F1_BACKEND_URL is not configured");
  }
  return value.replace(/\/$/, "");
}

function backendHeaders(): HeadersInit {
  const token = process.env.F1_BACKEND_TOKEN?.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function proxyJson(path: string, search: URLSearchParams): Promise<NextResponse> {
  try {
    const url = new URL(`${backendBaseUrl()}${path}`);
    search.forEach((value, key) => url.searchParams.set(key, value));
    const response = await fetch(url, {
      headers: backendHeaders(),
      cache: "no-store",
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    });
    const text = await response.text();
    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch {
      payload = { detail: text || `Backend returned HTTP ${response.status}` };
    }
    return NextResponse.json(payload, {
      status: response.status,
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend request failed";
    return NextResponse.json(
      { detail: message, source: "vercel_proxy" },
      { status: 503, headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  }
}

export function pickSearchParams(request: Request, allowed: readonly string[]): URLSearchParams {
  const incoming = new URL(request.url).searchParams;
  const output = new URLSearchParams();
  for (const key of allowed) {
    const value = incoming.get(key);
    if (value !== null) output.set(key, value);
  }
  return output;
}
