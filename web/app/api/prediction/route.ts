import { pickSearchParams, proxyJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return proxyJson("/v1/prediction", pickSearchParams(request, ["total_laps", "samples"]));
}
