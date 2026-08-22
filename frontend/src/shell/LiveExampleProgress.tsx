import type { LiveExampleController } from '../liveExample/useLiveExampleFlow';
import { formatDisplayId } from '../displayId';
import { StatusBadge } from '../components/StatusBadge';
import type { EvidenceCertificate } from '../types';

/** Mirrors SamplingWorkspace.tsx's STATUS_LABEL -- same real
 * EvidenceCertificate contract, same wording, just reused here for the
 * LIVE judge path's own governed-stop pause. */
const CERTIFICATE_STATUS_LABEL: Record<EvidenceCertificate['status'], string> = {
  EVIDENCE_SUFFICIENT: 'EVIDENCE SUFFICIENT',
  CONTINUE_SAMPLING: 'CONTINUE SAMPLING',
  STOP_BUDGET_EXHAUSTED: 'STOP: SAMPLE BUDGET EXHAUSTED',
  STOP_NO_USEFUL_CANDIDATE: 'STOP: NO USEFUL CANDIDATE',
  STOP_ABSTAIN: 'STOP: ABSTAIN',
};

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
  governed_stop: 'Planning is not currently permitted for this incident',
  error: 'Something went wrong',
};

const PIPELINE_STAGES = [
  ['importing_network', 'Network import'],
  ['creating_incident', 'Incident creation'],
  ['analyzing_initial', 'Initial analysis'],
  ['awaiting_sample_collection', 'Active sample selection'],
  ['submitting_sample', 'Sample submission'],
  ['reanalyzing', 'Reanalysis'],
  ['generating_plans', 'Plan generation'],
  ['verifying_plans', 'Exact WNTR verification'],
  ['awaiting_approval', 'Human approval'],
] as const;

function stageIndex(stage: string): number {
  const explicit = PIPELINE_STAGES.findIndex(([id]) => id === stage);
  if (explicit >= 0) return explicit;
  if (stage === 'approving' || stage === 'complete') return PIPELINE_STAGES.length;
  return -1;
}

export function LiveExampleProgress({
  controller,
  onExploreFallback,
}: {
  controller: LiveExampleController;
  onExploreFallback: () => void;
}) {
  const { stage } = controller;
  const currentStage = stageIndex(stage);

  return (
    <main className="first-launch-gateway" aria-live="polite">
      <div className="first-launch-panel">
        <p className="eyebrow">LIVE COMPUTATION · REFERENCE INPUTS</p>
        <h1>{STAGE_LABELS[stage] ?? stage}</h1>
        <p className="supporting">
          HydroSwarm is computing this incident now using the frozen runtime. Input observations are
          from the included reference scenario, not live utility telemetry.
        </p>
        {controller.incidentId && (
          <p className="live-incident-id">
            Incident{' '}
            <span title={controller.incidentId}>{formatDisplayId(controller.incidentId)}</span> ·{' '}
            created locally
          </p>
        )}
        <ol className="live-pipeline" aria-label="Live computation stages">
          {PIPELINE_STAGES.map(([id, label], index) => (
            <li
              key={id}
              className={
                index < currentStage ? 'complete' : index === currentStage ? 'current' : ''
              }
            >
              <span aria-hidden="true">
                {index < currentStage ? '✓' : index === currentStage ? '●' : '○'}
              </span>
              {label}
            </li>
          ))}
        </ol>
        <div className="live-trust">
          <span>Local / offline</span>
          <span>Exact verification before approval</span>
        </div>

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

        {stage === 'governed_stop' && controller.evidenceCertificate && (
          <div className="live-example-pause">
            <StatusBadge tone="warn">
              {CERTIFICATE_STATUS_LABEL[controller.evidenceCertificate.status]}
            </StatusBadge>
            <p className="supporting">{controller.evidenceCertificate.message}</p>
            <p className="supporting">
              This is the real deterministic evidence/sampling policy correctly declining to
              recommend a further sample or permit planning for this incident's current evidence --
              not an application error. No sample is fabricated and no plan is generated to force a
              result.
            </p>
            <button type="button" onClick={controller.restart}>
              Run again
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
