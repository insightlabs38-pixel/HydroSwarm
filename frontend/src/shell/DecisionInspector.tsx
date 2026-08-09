import type { ReactNode } from 'react';
import type { IncidentView } from '../types';
import { useConsoleStore } from '../store';
import { EmptyState } from '../components/common/EmptyState';
import { StatusBadge } from '../components/StatusBadge';

// Defensive fallback only -- every Workspace value is handled explicitly
// below; this stays empty and exists so an unhandled future workspace
// value fails honestly (an EmptyState with no detail) instead of
// crashing on an undefined lookup.
const NOT_YET_IMPLEMENTED: Partial<Record<string, string>> = {};

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

/** ui-work.txt 13.2: real-time summary only -- the full authority
 * certificate and grounded explanation live in the Source workspace
 * body (a separate query), not duplicated here. */
function SourceSummary({ incident }: { incident: IncidentView }) {
  if (incident.candidates.length === 0) {
    return <EmptyState title="No source candidates for this incident." />;
  }
  return (
    <div className="inspector-stack">
      {incident.candidates.slice(0, 3).map((candidate, index) => (
        <div className="compact-row" key={candidate.nodeId}>
          <span>
            {index + 1}. {candidate.nodeId}
          </span>
          <strong>{Math.round(candidate.probability * 100)}%</strong>
        </div>
      ))}
      <dl className="key-value-grid">
        <div>
          <dt>Candidate-set size</dt>
          <dd>{incident.candidates.length}</dd>
        </div>
        <div>
          <dt>Conformal target</dt>
          <dd>{Math.round(incident.candidateCoverage * 100)}%</dd>
        </div>
        <div>
          <dt>Calibration</dt>
          <dd>{incident.calibrationValid ? 'valid' : 'invalid'}</dd>
        </div>
        <div>
          <dt>Disagreement</dt>
          <dd>{(incident.disagreement * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt>OOD</dt>
          <dd>{incident.ood}</dd>
        </div>
      </dl>
    </div>
  );
}

/** ui-work.txt 13.3: full evidence certificate (a separate query) lives
 * in the Sampling workspace body -- this is the always-available
 * real-time summary. */
function SamplingSummary({ incident }: { incident: IncidentView }) {
  if (!incident.recommendedSample) {
    return (
      <EmptyState
        title="No further sampling recommended."
        detail="Sampling budget exhausted, or active sampling found no further useful measurement."
      />
    );
  }
  return (
    <div className="inspector-stack">
      <div className="candidate-hero">
        <strong>{incident.recommendedSample.nodeId}</strong>
        <span>{incident.recommendedSample.informationGain.toFixed(2)} bits</span>
      </div>
      <p className="supporting">{incident.recommendedSample.rationale}</p>
    </div>
  );
}

/** ui-work.txt 13.4: full plan/verification detail (and the Pareto
 * frontier, a separate query) lives in the Response workspace body --
 * this is the always-available real-time summary for whichever plan is
 * selected (falling back to the recommended plan). */
function ResponseSummary({ incident }: { incident: IncidentView }) {
  const { selectedPlanId } = useConsoleStore();
  const plan =
    incident.plans.find((item) => item.id === selectedPlanId) ??
    incident.plans.find((item) => item.id === incident.recommendedPlanId) ??
    incident.plans[0];
  if (!plan) {
    return <EmptyState title="No response plans available for this incident." />;
  }
  return (
    <div className="inspector-stack">
      <p className="supporting">
        {plan.id} · {plan.name}
      </p>
      <StatusBadge
        tone={
          plan.status === 'RECOMMENDED' || plan.status === 'VALID'
            ? 'good'
            : plan.status === 'REJECTED'
              ? 'danger'
              : 'warn'
        }
      >
        {plan.status}
      </StatusBadge>
      {plan.verification && (
        <p className="supporting">Verification: {plan.verification.verificationStatus}</p>
      )}
      <p className="supporting">{plan.actions.length} action(s) in this plan.</p>
    </div>
  );
}

/** ui-work.txt 13.5: the guarded approval form itself lives in the
 * Approval workspace body -- this is the always-available real-time
 * summary of where the incident stands. */
function ApprovalSummary({ incident }: { incident: IncidentView }) {
  if (incident.selectedPlanId) {
    return (
      <div className="inspector-stack">
        <StatusBadge tone="good">APPROVED</StatusBadge>
        <p className="supporting">Plan {incident.selectedPlanId} was approved by an operator.</p>
      </div>
    );
  }
  if (!incident.approvalPending) {
    return (
      <EmptyState
        title="No plan is awaiting approval."
        detail="A plan must be verified and current before it can be reviewed for approval."
      />
    );
  }
  return (
    <div className="inspector-stack">
      <StatusBadge tone="warn">PENDING</StatusBadge>
      <p className="supporting">
        A verified plan is awaiting operator review. Approval records an operator decision only --
        HydroSwarm does not actuate infrastructure.
      </p>
    </div>
  );
}

/** ui-work.txt 13.6: the selected replay event's detail. */
function ReplaySummary({ incident }: { incident: IncidentView }) {
  const { selectedAuditSequence } = useConsoleStore();
  const event = incident.audit.find((item) => item.sequence === selectedAuditSequence);
  if (!event) {
    return (
      <EmptyState title="No event selected." detail="Select an event from the event ledger." />
    );
  }
  return (
    <div className="inspector-stack">
      <dl className="key-value-grid">
        <div>
          <dt>Sequence</dt>
          <dd>{event.sequence}</dd>
        </div>
        <div>
          <dt>Timestamp</dt>
          <dd className="mono">{event.timestamp}</dd>
        </div>
        <div>
          <dt>Actor</dt>
          <dd>{event.actor}</dd>
        </div>
        <div>
          <dt>Type</dt>
          <dd>{event.type.replaceAll('_', ' ')}</dd>
        </div>
      </dl>
      <p className="supporting">{event.detail}</p>
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
  } else if (workspace === 'source') {
    body = <SourceSummary incident={incident} />;
  } else if (workspace === 'sampling') {
    body = <SamplingSummary incident={incident} />;
  } else if (workspace === 'response') {
    body = <ResponseSummary incident={incident} />;
  } else if (workspace === 'approval') {
    body = <ApprovalSummary incident={incident} />;
  } else if (workspace === 'replay') {
    body = <ReplaySummary incident={incident} />;
  } else if (workspace === 'network') {
    body = (
      <p className="supporting">
        Local .inp import and validation only -- no cloud upload. See the Network workspace to
        import or inspect a network.
      </p>
    );
  } else if (workspace === 'authority') {
    body = (
      <p className="supporting">
        Governance table of which subsystem is authoritative for source localization, sample
        recommendation, OOD, and plan verification. See the Model &amp; Authority workspace for the
        full table.
      </p>
    );
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
