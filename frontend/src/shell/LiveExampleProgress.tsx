import type { LiveExampleController } from '../liveExample/useLiveExampleFlow';

/** submission.txt SUB-12.1 P1 #4: "LIVE COMPUTATION · REFERENCE INPUTS" --
 * shown while the real production pipeline is actually computing this
 * incident (real network import, real analysis, real WNTR verification).
 * Distinct from both the REFERENCE INCIDENT replay (a checksummed replay
 * of a *pre-computed* result) and DEMO_FALLBACK (no computation at all):
 * this is real, current computation, just against known reference inputs
 * instead of field telemetry -- the label makes both halves of that
 * honest. Once complete, App.tsx switches into the ordinary mission-
 * control shell in true LIVE mode showing the incident this flow created.
 */
const STAGE_LABELS: Record<string, string> = {
  idle: 'Starting…',
  importing_network: 'Importing the reference EPANET network',
  creating_incident: 'Creating a real incident against the imported network',
  analyzing_initial: 'Running real analysis on the initial observation',
  awaiting_sample_collection: 'Real sampling recommendation ready',
  submitting_sample: 'Submitting the real reference sample',
  reanalyzing: 'Re-running real analysis on the updated evidence',
  generating_plans: 'Generating real bounded response plan candidates',
  verifying_plans: 'Running real exact WNTR/EPANET verification',
  awaiting_approval: 'Verified plan ready for approval',
  approving: 'Recording approval',
  complete: 'Complete',
  error: 'Something went wrong',
};

export function LiveExampleProgress({
  controller,
  onExploreFallback,
}: {
  controller: LiveExampleController;
  onExploreFallback: () => void;
}) {
  const { stage } = controller;

  return (
    <main className="first-launch-gateway" aria-live="polite">
      <div className="first-launch-panel">
        <p className="eyebrow">LIVE COMPUTATION · REFERENCE INPUTS</p>
        <h1>{STAGE_LABELS[stage] ?? stage}</h1>
        <p className="supporting">
          HydroSwarm is computing this incident now using the frozen runtime. Input observations are
          from the included reference scenario, not live utility telemetry.
        </p>

        {stage === 'awaiting_sample_collection' && controller.recommendedNode && (
          <div className="live-example-pause">
            <p>
              HydroSwarm recommends <strong>{controller.recommendedNode}</strong>
              {typeof controller.expectedInformationGainBits === 'number' && (
                <>
                  {' '}
                  · expected information gain{' '}
                  <strong>{controller.expectedInformationGainBits.toFixed(2)} bits</strong>
                </>
              )}
              .
            </p>
            <button type="button" onClick={controller.collectSample}>
              Collect reference sample
            </button>
          </div>
        )}

        {stage === 'awaiting_approval' && controller.verifiedPlan && (
          <div className="live-example-pause">
            <p>
              Verified plan ready: <strong>{controller.verifiedPlan.name}</strong>. HydroSwarm never
              executes a response autonomously.
            </p>
            <button type="button" onClick={controller.approve}>
              Approve plan
            </button>
          </div>
        )}

        {stage === 'error' && (
          <div className="live-example-pause">
            <p role="alert" className="supporting">
              {controller.errorMessage ?? 'The live example could not complete.'}
            </p>
            <button type="button" onClick={controller.restart}>
              Try again
            </button>
          </div>
        )}

        <button type="button" className="first-launch-secondary" onClick={onExploreFallback}>
          Explore illustrative fallback instead
        </button>
      </div>
    </main>
  );
}
