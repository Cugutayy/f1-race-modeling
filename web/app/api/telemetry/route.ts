import { pickSearchParams, proxyJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return proxyJson("/v1/telemetry", pickSearchParams(request, ["driver_number", "limit"]));
}
