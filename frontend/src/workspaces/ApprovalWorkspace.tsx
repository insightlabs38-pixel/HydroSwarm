import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { IncidentView, Plan } from '../types';
import { approvePlan } from '../api/approval';
import { ApiError } from '../api/client';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { PlanActionSequence } from '../components/plans/PlanActionSequence';
import { EmptyState } from '../components/common/EmptyState';
import { KeyValueGrid } from '../components/common/KeyValueGrid';
import { formatDisplayId } from '../displayId';

type HierarchyStep =
  'SIMULATOR_VERIFIED' | 'CURRENT_CONTEXT' | 'OPERATOR_REVIEW' | 'HUMAN_APPROVED';

const HIERARCHY: { id: HierarchyStep; label: string }[] = [
  { id: 'SIMULATOR_VERIFIED', label: 'Simulator verified' },
  { id: 'CURRENT_CONTEXT', label: 'Current context' },
  { id: 'OPERATOR_REVIEW', label: 'Operator review' },
  { id: 'HUMAN_APPROVED', label: 'Human approved' },
];

function HierarchyLadder({ reached }: { reached: HierarchyStep | null }) {
  const reachedIndex = reached ? HIERARCHY.findIndex((step) => step.id === reached) : -1;
  return (
    <ol className="approval-hierarchy" aria-label="Approval authority hierarchy">
      {HIERARCHY.map((step, index) => (
        <li key={step.id} className={index <= reachedIndex ? 'reached' : ''}>
          {step.label}
        </li>
      ))}
    </ol>
  );
}

function ApprovalDecisionGate({
  incident,
  verification,
  alreadyApproved,
}: {
  incident: IncidentView;
  verification: NonNullable<Plan['verification']>;
  alreadyApproved: boolean;
}) {
  const consequences = verification.consequences;
  const pressureMargin = consequences?.pressureMarginM;
  const eligibilityEntries = [
    { key: 'exact-verification', label: 'Exact verification', value: verification.decision },
    {
      key: 'verification-context',
      label: 'Verification context',
      value: verification.verificationStatus,
    },
    ...(pressureMargin === null || pressureMargin === undefined
      ? []
      : [
          {
            key: 'pressure-margin',
            label: 'Pressure margin',
            value: `${pressureMargin > 0 ? '+' : ''}${pressureMargin.toFixed(1)} m`,
          },
        ]),
    ...(consequences
      ? [
          {
            key: 'service-availability',
            label: 'Service availability',
            value: `${(consequences.serviceAvailability * 100).toFixed(1)}%`,
          },
        ]
      : []),
    {
      key: 'operator-decision',
      label: 'Operator decision',
      value: alreadyApproved ? 'APPROVED' : incident.approvalPending ? 'PENDING' : 'NOT REQUESTED',
    },
    { key: 'infrastructure-actuation', label: 'Infrastructure actuation', value: 'NONE' },
  ];

  return (
    <section className="approval-decision-gate" aria-labelledby="approval-decision-gate-title">
      <div className="approval-decision-gate-heading">
        <div>
          <p className="eyebrow">READY FOR HUMAN REVIEW</p>
          <h3 id="approval-decision-gate-title">Decision gate</h3>
        </div>
        <div className="approval-decision-gate-statuses" aria-label="Review eligibility">
          <StatusBadge tone="good">VERIFIED</StatusBadge>
          <StatusBadge tone="good">CURRENT</StatusBadge>
        </div>
      </div>
      <KeyValueGrid entries={eligibilityEntries} />
      <div className="approval-decision-effect">
        <p className="eyebrow">WHAT THIS DECISION DOES</p>
        <KeyValueGrid
          entries={[
            { key: 'records', label: 'Records', value: 'Human operator decision' },
            { key: 'does-not', label: 'Does not', value: 'Execute infrastructure' },
            {
              key: 'reference-mode',
              label: 'Reference mode',
              value: 'Replays the checksummed recorded approval transition only',
            },
          ]}
        />
      </div>
    </section>
  );
}

/**
 * ui-work.txt UI-6 / 13.5: guarded human plan-approval workflow. Never
 * enables approval unless the active plan is VERIFIED and CURRENT;
 * requires an operator ID and an explicit review checkbox; POSTs the
 * real approval and re-fetches authoritative state afterward. Fails
 * closed (never silently retries or infers success) on a 409 -- most
 * commonly a stale-verification race the backend itself detects.
 * DEMO_FALLBACK never performs a real approval mutation: there is no
 * live incident/plan UUID to record a decision against, so the action
 * is disabled with an explicit reason rather than faking a receipt.
 */
export function ApprovalWorkspace({ incident }: { incident: IncidentView }) {
  const queryClient = useQueryClient();
  const [operatorId, setOperatorId] = useState('');
  const [reviewed, setReviewed] = useState(false);

  const activePlan: Plan | null =
    incident.plans.find((plan) => plan.id === incident.selectedPlanId) ??
    incident.plans.find((plan) => plan.id === incident.recommendedPlanId) ??
    incident.plans[0] ??
    null;

  const approveMutation = useMutation({
    mutationFn: () => approvePlan(incident.id, activePlan!.id, operatorId.trim()),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['active-incident'] });
    },
  });

  if (!activePlan) {
    return (
      <div className="approval-workspace">
        <Panel title="No plan to approve" eyebrow="APPROVAL" className="wide-panel">
          <EmptyState title="No response plans available for this incident." />
        </Panel>
      </div>
    );
  }

  const verification = activePlan.verification;
  const isVerified = verification?.decision === 'VERIFIED';
  const isCurrent = verification?.verificationStatus === 'CURRENT';
  const canReview = isVerified && isCurrent;
  const alreadyApproved = incident.selectedPlanId === activePlan.id && !incident.approvalPending;

  let reachedStep: HierarchyStep | null = null;
  if (alreadyApproved) reachedStep = 'HUMAN_APPROVED';
  else if (canReview && reviewed && operatorId.trim().length > 0) reachedStep = 'OPERATOR_REVIEW';
  else if (isVerified && isCurrent) reachedStep = 'CURRENT_CONTEXT';
  else if (isVerified) reachedStep = 'SIMULATOR_VERIFIED';

  const canApprove =
    incident.mode === 'LIVE' &&
    canReview &&
    !alreadyApproved &&
    operatorId.trim().length > 0 &&
    reviewed &&
    !approveMutation.isPending;

  return (
    <div
      className={`approval-workspace${
        incident.mode === 'REFERENCE' ? ' approval-workspace-reference' : ''
      }`}
    >
      <Panel
        title={`${formatDisplayId(activePlan.id)} · ${activePlan.name}`}
        eyebrow="PLAN UNDER REVIEW"
        className="approval-plan-summary"
      >
        <HierarchyLadder reached={reachedStep} />
        <span className="sr-only">Full plan ID {activePlan.id}</span>
        <p className="supporting">
          HydroSwarm does not actuate infrastructure. Approval records an operator decision only.
        </p>
        {!isVerified && (
          <EmptyState
            title="This plan cannot be approved."
            detail={
              verification
                ? `Verification decision is ${verification.decision}, not VERIFIED.`
                : 'This plan has not yet been submitted for exact verification.'
            }
          />
        )}
        {isVerified && !isCurrent && (
          <EmptyState
            title="Verification is stale."
            detail="Verification is stale because incident evidence or verification context changed. Re-verify before approval."
          />
        )}
      </Panel>
      <section className="approval-decision-boundary" aria-labelledby="approval-boundary-title">
        <Panel
          title="Operator approval"
          eyebrow="HUMAN DECISION BOUNDARY"
          className="approval-primary-panel"
        >
          <p id="approval-boundary-title" className="approval-boundary-statement">
            HydroSwarm does not actuate infrastructure. A human operator records the decision only
            after reviewing a current, exactly verified plan.
          </p>
          {alreadyApproved ? (
            <StatusBadge tone="good">APPROVED</StatusBadge>
          ) : incident.mode === 'REFERENCE' ? (
            <>
              {canReview && verification && (
                <ApprovalDecisionGate
                  incident={incident}
                  verification={verification}
                  alreadyApproved={alreadyApproved}
                />
              )}
              <div className="reference-approval-boundary">
                <strong>Reference replay paused at the human decision boundary</strong>
                <p>
                  This is a checksummed replay of a previously generated workflow. No infrastructure
                  action is executed.
                </p>
                <p>
                  Advance the replay from the reference banner to reproduce the recorded
                  operator-approval transition.
                </p>
              </div>
            </>
          ) : incident.mode !== 'LIVE' ? (
            <EmptyState
              title="Approval is unavailable outside a live incident."
              detail="Illustrative and replay views never record an infrastructure decision."
            />
          ) : !canReview ? (
            <EmptyState title="Resolve the verification issue above before approval can be reviewed." />
          ) : (
            <form
              className="approval-form"
              onSubmit={(event) => {
                event.preventDefault();
                if (canApprove) approveMutation.mutate();
              }}
            >
              <label className="approval-field">
                Operator ID
                <input
                  type="text"
                  value={operatorId}
                  onChange={(event) => setOperatorId(event.target.value)}
                  required
                  maxLength={80}
                  autoComplete="off"
                />
              </label>
              <label className="approval-checkbox">
                <input
                  type="checkbox"
                  checked={reviewed}
                  onChange={(event) => setReviewed(event.target.checked)}
                />
                I reviewed the verified actions and consequences.
              </label>
              <button type="submit" className="primary-action" disabled={!canApprove}>
                {approveMutation.isPending ? 'Recording approval…' : 'Approve verified plan'}
              </button>
              {approveMutation.isError && (
                <p className="supporting" role="alert">
                  {approveMutation.error instanceof ApiError
                    ? approveMutation.error.message
                    : 'Approval failed. Re-fetch and re-verify before trying again.'}
                </p>
              )}
              {approveMutation.isSuccess && (
                <div className="approval-receipt" role="status">
                  <strong>Approval receipt</strong>
                  <KeyValueGrid
                    entries={[
                      {
                        key: 'operator',
                        label: 'Operator',
                        value: approveMutation.data.operatorId,
                      },
                      {
                        key: 'approved-at',
                        label: 'Approved at',
                        value: approveMutation.data.approvedAt,
                      },
                      {
                        key: 'plan',
                        label: 'Plan',
                        value: approveMutation.data.planId,
                        hash: true,
                      },
                    ]}
                  />
                </div>
              )}
            </form>
          )}
        </Panel>
      </section>
      <aside className="approval-evidence-stack" aria-label="Verified actions and consequences">
        <Panel title="Every action in this plan" eyebrow="FULL PLAN">
          <PlanActionSequence plan={activePlan} />
        </Panel>
        {verification?.consequences && (
          <Panel title="Safety margins and consequences" eyebrow="EXPECTED">
            <KeyValueGrid
              entries={[
                {
                  key: 'pressure-margin',
                  label: 'Pressure margin',
                  value:
                    verification.consequences.pressureMarginM === null
                      ? null
                      : `${verification.consequences.pressureMarginM.toFixed(1)} m`,
                },
                {
                  key: 'service-margin',
                  label: 'Service margin',
                  value:
                    verification.consequences.serviceAvailabilityMargin === null
                      ? null
                      : `${(verification.consequences.serviceAvailabilityMargin * 100).toFixed(1)} pp`,
                },
                {
                  key: 'sensitive',
                  label: 'Numerically sensitive',
                  value: verification.consequences.numericallySensitive ? 'yes' : 'no',
                },
                {
                  key: 'service-availability',
                  label: 'Service availability',
                  value: `${(verification.consequences.serviceAvailability * 100).toFixed(1)}%`,
                },
              ]}
            />
          </Panel>
        )}
        {verification?.worstCaseConsequences && (
          <Panel title="Worst case" eyebrow="ACROSS EVALUATED HYPOTHESES">
            <KeyValueGrid
              entries={[
                {
                  key: 'worst-pressure',
                  label: 'Minimum pressure',
                  value: `${verification.worstCaseConsequences.minimumPressureM.toFixed(1)} m`,
                },
                {
                  key: 'worst-service',
                  label: 'Service availability',
                  value: `${(verification.worstCaseConsequences.serviceAvailability * 100).toFixed(1)}%`,
                },
              ]}
            />
          </Panel>
        )}
        {verification && (
          <Panel title="Verification context" eyebrow="PROVENANCE">
            <KeyValueGrid
              entries={[
                {
                  key: 'simulator',
                  label: 'Simulator',
                  value: `${verification.simulator} ${verification.simulatorVersion}`,
                },
                { key: 'verified-at', label: 'Verified at', value: verification.verifiedAt },
                {
                  key: 'context-hash',
                  label: 'Context hash',
                  value: verification.contextHash,
                  hash: true,
                },
              ]}
            />
          </Panel>
        )}
      </aside>
    </div>
  );
}
