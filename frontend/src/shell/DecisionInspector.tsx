import type { ReactNode } from 'react';
import type { IncidentView } from '../types';
import { useConsoleStore } from '../store';
import { EmptyState } from '../components/common/EmptyState';
import { StatusBadge } from '../components/StatusBadge';

const NOT_YET_IMPLEMENTED: Partial<Record<string, string>> = {
  source: 'UI-3 (governed source-localization workspace) adds the ranked candidate inspector here.',
  sampling: 'UI-4 (evidence-value sampling workspace) adds the evidence/stop certificate here.',
  response: 'UI-5 (response-plan decision workspace) adds full plan/verification detail here.',
  approval: 'UI-6 (guarded approval workflow) adds the approval flow here.',
  replay: 'UI-8 (replay/failure/demo) adds the selected replay-event detail here.',
  network: 'UI-9 (utilities) adds network import/validation detail here.',
  authority: 'UI-9 (utilities) adds the Decision Certificate authority table here.',
};

/**
 * ui-work.txt 13: "This pane must always answer: What does HydroSwarm
 * currently believe, recommend, or require the operator to decide?" For
 * the Incident workspace this renders real, already-available fields
 * (13.1). Every other workspace is honestly `EmptyState`d until its own
 * phase lands -- never a fabricated preview of unbuilt content.
 */
function IncidentSummary({ incident }: { incident: IncidentView }) {
  const leading = incident.candidates[0];
  const nextStep = incident.recommendedSample
    ? 'Collect recommended sample'
    : incident.plans.length > 0 && !incident.selectedPlanId
      ? 'Review response plan'
      : incident.approvalPending
        ? 'Approve verified plan'
        : 'Monitor';
  return (
    <div className="inspector-stack">
      <dl className="key-value-grid">
        <div>
          <dt>Controller state</dt>
          <dd>{incident.status}</dd>
        </div>
        <div>
          <dt>Runtime mode</dt>
          <dd>{incident.runtimeAnalysisMode ?? '—'}</dd>
        </div>
        <div>
          <dt>OOD</dt>
          <dd>{incident.ood}</dd>
        </div>
        <div>
          <dt>Calibration</dt>
          <dd>{incident.calibrationValid ? 'valid' : 'invalid'}</dd>
        </div>
        <div>
          <dt>Approval</dt>
          <dd>
            {incident.selectedPlanId
              ? `approved (${incident.selectedPlanId})`
              : incident.approvalPending
                ? 'pending'
                : 'none pending'}
          </dd>
        </div>
      </dl>
      {leading ? (
        <div className="candidate-hero">
          <strong>{leading.nodeId}</strong>
          <span>{Math.round(leading.probability * 100)}%</span>
        </div>
      ) : (
        <EmptyState title="No source candidates for this incident." />
      )}
      <p className="supporting">
        Suggested next step (derived from current state, not an authoritative controller output):{' '}
        <strong>{nextStep}</strong>.
      </p>
      <p className="supporting">
        Simulator budget: not yet exposed to the console (a known gap -- IncidentState carries
        remaining_epanet_budget, but the /view response does not yet include it).
      </p>
    </div>
  );
}

export function DecisionInspector({ incident }: { incident: IncidentView }) {
  const { workspace, inspectorCollapsed, toggleInspector } = useConsoleStore();

  if (inspectorCollapsed) {
    return (
      <button
        type="button"
        className="inspector-expand-toggle"
        onClick={toggleInspector}
        aria-label="Expand decision inspector"
      >
        «
      </button>
    );
  }

  let body: ReactNode;
  if (workspace === 'incident') {
    body = <IncidentSummary incident={incident} />;
  } else if (workspace === 'validation') {
    body = (
      <p className="supporting">
        Governed evaluation and promotion evidence loaded from committed artifacts. See the
        Validation workspace for the full table.
      </p>
    );
  } else if (workspace === 'benchmarks') {
    body = (
      <p className="supporting">
        {incident.benchmarks.length} operational benchmark row
        {incident.benchmarks.length === 1 ? '' : 's'} loaded for this build. See the Benchmarks
        workspace for detail.
      </p>
    );
  } else {
    body = (
      <EmptyState
        title="Not yet implemented in the mission-control shell."
        detail={NOT_YET_IMPLEMENTED[workspace]}
      />
    );
  }

  return (
    <aside className="decision-inspector" aria-label="Decision inspector">
      <div className="inspector-header">
        <h2>Decision inspector</h2>
        <StatusBadge tone="info">{incident.mode}</StatusBadge>
        <button
          type="button"
          className="inspector-collapse-toggle"
          onClick={toggleInspector}
          aria-label="Collapse decision inspector"
        >
          »
        </button>
      </div>
      {body}
    </aside>
  );
}
