import ModelEvidencePanel from "./model-evidence-panel";
import ProviderHealth from "./provider-health";
import RaceIntelligence from "./race-intelligence";
import HistoricalTelemetry from "./historical-telemetry";

export default function Page() {
  return (
    <>
      <RaceIntelligence />
      <HistoricalTelemetry />
      <ModelEvidencePanel />
      <ProviderHealth />
    </>
  );
}
