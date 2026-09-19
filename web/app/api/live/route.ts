import { pickSearchParams, proxyJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const search = pickSearchParams(request, ["total_laps", "samples"]);
  if (!search.has("total_laps")) search.set("total_laps", "60");
  if (!search.has("samples")) search.set("samples", "4000");
  return proxyJson("/v1/live", search);
}
