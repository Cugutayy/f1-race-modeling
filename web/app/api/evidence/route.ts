import { proxyJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  return proxyJson("/v1/evidence", new URLSearchParams());
}
