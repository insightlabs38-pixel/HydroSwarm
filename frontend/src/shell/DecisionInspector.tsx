import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { IncidentView } from '../types';
import {
  calibrationStatusText,
  candidateCoverageLabel,
  candidateCoverageValueText,
  measuredCoverageValueText,
} from '../calibrationDisplay';
import { useConsoleStore } from '../store';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates } from '../demoFixture';
import { deriveWorkflowProgression } from '../workflow';
import { EmptyState } from '../components/common/EmptyState';
import { StatusBadge } from '../components/StatusBadge';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { ApplicabilityBadge } from '../components/status/ApplicabilityBadge';
import { KeyValueGrid } from '../components/common/KeyValueGrid';
import { formatDisplayId } from '../displayId';

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
  const { nextStep } = deriveWorkflowProgression(incident);
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
          <dd>{calibrationStatusText(incident)}</dd>
        </div>
        <div>
          <dt>Approval</dt>
          <dd>
            {incident.selectedPlanId ? (
              <>
                approved (
                <span title={incident.selectedPlanId}>
                  {formatDisplayId(incident.selectedPlanId)}
                </span>
                )
              </>
            ) : incident.approvalPending ? (
              'pending'
            ) : (
              'none pending'
            )}
          </dd>
        </div>
        <div>
          <dt>Remaining exact simulation budget</dt>
          <dd>
            {incident.simulatorBudget
              ? incident.simulatorBudget.remainingEpanetBudget
              : 'not measured'}
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
        Next step (derived from the controller's real stage, <code>{incident.status}</code>):{' '}
        <strong>{nextStep}</strong>.
      </p>
    </div>
  );
}

/** ui-work.txt 13.2: real-time summary. Full ranked candidates, the full
 * authority certificate, and the grounded explanation stay PRIMARY in the
 * Source workspace body (ui-work.txt "UI-10.5" 8's duplication rule) --
 * this queries the same `['authority', incident.id]` cache the Source
 * workspace populates (TanStack Query dedupes by key, so this never
 * double-fetches) purely to surface a compact current
 * authority/applicability summary here, per "UI-10.5" 2's SOURCE list. */
function SourceSummary({ incident }: { incident: IncidentView }) {
  const authorityQuery = useQuery({
    queryKey: ['authority', incident.id],
    queryFn: ({ signal }) => fetchAuthorityCertificates(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const certificates =
    incident.mode === 'LIVE'
      ? (authorityQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoAuthorityCertificates
        : [];
  const sourceCertificate = certificates.find((cert) => cert.name === 'source_localization');

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
          <dt>{candidateCoverageLabel(incident)}</dt>
          <dd>{candidateCoverageValueText(incident)}</dd>
        </div>
        <div>
          <dt>Held-out measured coverage</dt>
          <dd>{measuredCoverageValueText(incident)}</dd>
        </div>
        <div>
          <dt>Calibration</dt>
          <dd>{calibrationStatusText(incident)}</dd>
        </div>
        <div>
          <dt>Disagreement</dt>
          <dd>
            {typeof incident.disagreement === 'number'
              ? `${(incident.disagreement * 100).toFixed(1)}%`
              : 'not measured'}
          </dd>
        </div>
        <div>
          <dt>OOD</dt>
          <dd>{incident.ood}</dd>
        </div>
      </dl>
      {sourceCertificate && (
        <div className="decision-badges">
          <AuthorityBadge authority={sourceCertificate.authority} />
          <ApplicabilityBadge applicability={sourceCertificate.applicability} />
        </div>
      )}
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

/** ui-work.txt "UI-10.5" 2 RESPONSE: this becomes the PRIMARY selected-plan
 * decision panel (identity, decision, CURRENT/STALE, simulator identity,
 * margins, numerical sensitivity, rejection/abstention reason, compact
 * action count) -- the full ordered action sequence, plan comparison
 * table, and Pareto frontier stay PRIMARY in the Response workspace body,
 * and the full forensic verification record (every hash, verified-at,
 * worst-case consequences) stays PRIMARY in TechnicalDock > Verification
 * (ui-work.txt "UI-10.5" 8's duplication rule). */
function ResponseSummary({ incident }: { incident: IncidentView }) {
  const { selectedPlanId } = useConsoleStore();
  const plan =
    incident.plans.find((item) => item.id === selectedPlanId) ??
    incident.plans.find((item) => item.id === incident.recommendedPlanId) ??
    incident.plans[0];
  if (!plan) {
    return <EmptyState title="No response plans available for this incident." />;
  }
  const verification = plan.verification;
  const consequences = verification?.consequences ?? null;
  return (
    <div className="inspector-stack">
      <p className="supporting">
        <span title={plan.id}>{formatDisplayId(plan.id)}</span> · {plan.name}
      </p>
      <div className="decision-badges">
        <StatusBadge
          tone={
            verification
              ? verification.decision === 'VERIFIED'
                ? 'good'
                : verification.decision === 'REJECTED'
                  ? 'danger'
                  : 'warn'
              : plan.status === 'REJECTED'
                ? 'danger'
                : 'warn'
          }
        >
          {verification ? verification.decision : plan.status}
        </StatusBadge>
        {verification && (
          <StatusBadge tone={verification.verificationStatus === 'CURRENT' ? 'good' : 'danger'}>
            {verification.verificationStatus}
          </StatusBadge>
        )}
      </div>
      {verification && (
        <KeyValueGrid
          entries={[
            {
              key: 'simulator',
              label: 'Simulator',
              value: `${verification.simulator} ${verification.simulatorVersion}`,
            },
          ]}
        />
      )}
      {consequences && (
        <KeyValueGrid
          entries={[
            {
              key: 'pressure-margin',
              label: 'Pressure margin',
              value:
                consequences.pressureMarginM === null
                  ? null
                  : `${consequences.pressureMarginM.toFixed(1)} m`,
            },
            {
              key: 'service-margin',
              label: 'Service margin',
              value:
                consequences.serviceAvailabilityMargin === null
                  ? null
                  : `${(consequences.serviceAvailabilityMargin * 100).toFixed(1)} pp`,
            },
            {
              key: 'sensitive',
              label: 'Numerically sensitive',
              value: consequences.numericallySensitive ? 'yes' : 'no',
            },
          ]}
        />
      )}
      {verification && verification.rejectionCodes.length > 0 && (
        <p className="supporting">Rejection codes: {verification.rejectionCodes.join(', ')}</p>
      )}
      {verification?.abstentionReason && (
        <p className="supporting">Abstention reason: {verification.abstentionReason}</p>
      )}
      <p className="supporting">{plan.actions.length} action(s) in this plan.</p>
      {incident.simulatorBudget && (
        <KeyValueGrid
          entries={[
            {
              key: 'remaining-budget',
              label: 'Remaining exact simulation budget',
              value: String(incident.simulatorBudget.remainingEpanetBudget),
            },
            {
              key: 'exact-sims-used',
              label: 'Exact simulations used',
              value: String(incident.simulatorBudget.exactSimulationsUsed),
            },
            {
              key: 'plans-verified',
              label: 'Plans exactly verified',
              value: String(incident.simulatorBudget.plansExactlyVerified),
            },
            {
              key: 'cache-hits',
              label: 'Exact simulation cache hits',
              value: String(incident.simulatorBudget.exactSimulationCacheHits),
            },
          ]}
        />
      )}
    </div>
  );
}

/** ui-work.txt 13.5: the guarded approval form itself lives in the
 * Approval workspace body -- this is the always-available real-time
 * summary of where the incident stands. */
function ApprovalSummary({ incident }: { incident: IncidentView }) {
  // Keep the same selected/recommended fallback as ApprovalWorkspace so the
  // permanent inspector never summarizes a different plan than the decision
  // boundary it accompanies.
  const plan =
    incident.plans.find((item) => item.id === incident.selectedPlanId) ??
    incident.plans.find((item) => item.id === incident.recommendedPlanId) ??
    incident.plans[0];
  if (!plan) {
    return <EmptyState title="No plan is awaiting approval." />;
  }

  const verification = plan.verification;
  const consequences = verification?.consequences;
  const approved = incident.selectedPlanId === plan.id && !incident.approvalPending;

  return (
    <div className="inspector-stack">
      <StatusBadge tone={approved ? 'good' : incident.approvalPending ? 'warn' : 'info'}>
        {approved ? 'APPROVED' : incident.approvalPending ? 'PENDING' : 'NO APPROVAL PENDING'}
      </StatusBadge>
      <KeyValueGrid
        entries={[
          {
            key: 'plan',
            label: 'Plan',
            value: `${formatDisplayId(plan.id)} · ${plan.name}`,
          },
          {
            key: 'verification',
            label: 'Exact verification',
            value: verification?.decision ?? null,
          },
          {
            key: 'verification-context',
            label: 'Verification context',
            value: verification?.verificationStatus ?? null,
          },
          {
            key: 'pressure-margin',
            label: 'Pressure margin',
            value:
              consequences?.pressureMarginM === null || consequences?.pressureMarginM === undefined
                ? null
                : `${consequences.pressureMarginM >= 0 ? '+' : ''}${consequences.pressureMarginM.toFixed(1)} m`,
          },
          {
            key: 'service-availability',
            label: 'Service availability',
            value:
              consequences?.serviceAvailability === null ||
              consequences?.serviceAvailability === undefined
                ? null
                : `${(consequences.serviceAvailability * 100).toFixed(1)}%`,
          },
          {
            key: 'operator-decision',
            label: 'Operator decision',
            value: approved ? 'APPROVED' : incident.approvalPending ? 'PENDING' : 'NOT REQUESTED',
          },
          {
            key: 'infrastructure-actuation',
            label: 'Infrastructure actuation',
            value: 'NONE',
          },
        ]}
      />
      <span className="sr-only">Full plan ID {plan.id}</span>
      <p className="supporting">
        {approved
          ? 'An operator approval is recorded. HydroSwarm still performs no infrastructure actuation.'
          : 'A human operator decision remains required. HydroSwarm does not actuate infrastructure.'}
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

/** Secondary utilities waste horizontal space with empty inspector content. */
const SECONDARY_WORKSPACES: ReadonlySet<string> = new Set([
  'network', 'validation', 'authority', 'benchmarks',
]);

export function DecisionInspector({ incident }: { incident: IncidentView }) {
  const { workspace, inspectorCollapsed, toggleInspector } = useConsoleStore();

  // Secondary utility pages: no Decision Inspector rendered.
  if (SECONDARY_WORKSPACES.has(workspace)) {
    return null;
  }

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
