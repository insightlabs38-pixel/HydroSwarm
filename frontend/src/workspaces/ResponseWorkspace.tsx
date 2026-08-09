import { lazy, Suspense, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { IncidentView } from '../types';
import { fetchParetoFrontier } from '../api/planning';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates, demoParetoFrontier } from '../demoFixture';
import { useConsoleStore } from '../store';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { PlanTable } from '../components/PlanTable';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { PlanActionSequence } from '../components/plans/PlanActionSequence';
import { ParetoFrontier } from '../components/plans/ParetoFrontier';
import { EmptyState } from '../components/common/EmptyState';
import { KeyValueGrid } from '../components/common/KeyValueGrid';

const OperationalMap = lazy(() =>
  import('../components/OperationalMap').then((module) => ({ default: module.OperationalMap })),
);

/**
 * ui-work.txt UI-5: verified response-plan decision workspace
 * (Strategist). Full plan/action display, CURRENT/STALE verification,
 * expected/worst-case consequences, numerical sensitivity, and the
 * verified Pareto frontier -- consuming the real GET
 * /incidents/{id}/frontier endpoint and the plan_consequence:{plan_id}
 * authority certificates from the same /authority endpoint UI-3 wired.
 */
export function ResponseWorkspace({ incident }: { incident: IncidentView }) {
  const { selectedPlanId, selectPlan, frontierMode } = useConsoleStore();
  const activePlan =
    incident.plans.find((plan) => plan.id === selectedPlanId) ??
    incident.plans.find((plan) => plan.id === incident.recommendedPlanId) ??
    incident.plans[0] ??
    null;

  const frontierQuery = useQuery({
    queryKey: ['frontier', incident.id, frontierMode],
    queryFn: ({ signal }) => fetchParetoFrontier(incident.id, frontierMode, signal),
    enabled: incident.mode === 'LIVE',
  });
  const authorityQuery = useQuery({
    queryKey: ['authority', incident.id],
    queryFn: ({ signal }) => fetchAuthorityCertificates(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const frontier =
    incident.mode === 'LIVE'
      ? (frontierQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoParetoFrontier
        : [];
  const certificates =
    incident.mode === 'LIVE'
      ? (authorityQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoAuthorityCertificates
        : [];
  const planCertificate = activePlan
    ? certificates.find((cert) => cert.name === `plan_consequence:${activePlan.id}`)
    : undefined;
  const whyRejected = useMemo(
    () => incident.explanations.find((item) => item.intent === 'WHY_PLAN_REJECTED'),
    [incident.explanations],
  );
  const comparePlans = useMemo(
    () => incident.explanations.find((item) => item.intent === 'COMPARE_PLANS'),
    [incident.explanations],
  );

  return (
    <div className="workspace-grid">
      <Panel title="Network" eyebrow="RESPONSE ACTION OVERLAY" className="map-panel">
        <Suspense
          fallback={
            <div className="visual-loading" role="status">
              Loading offline network renderer…
            </div>
          }
        >
          <OperationalMap incident={incident} />
        </Suspense>
      </Panel>
      <aside className="right-rail" aria-label="Selected plan verification">
        {activePlan ? (
          <Panel title={`${activePlan.id} · ${activePlan.name}`} eyebrow="VERIFICATION">
            {activePlan.verification ? (
              <>
                <div className="decision-badges">
                  <StatusBadge
                    tone={
                      activePlan.verification.decision === 'VERIFIED'
                        ? 'good'
                        : activePlan.verification.decision === 'REJECTED'
                          ? 'danger'
                          : 'warn'
                    }
                  >
                    {activePlan.verification.decision}
                  </StatusBadge>
                  <StatusBadge
                    tone={
                      activePlan.verification.verificationStatus === 'CURRENT' ? 'good' : 'danger'
                    }
                  >
                    {activePlan.verification.verificationStatus}
                  </StatusBadge>
                  {planCertificate && <AuthorityBadge authority={planCertificate.authority} />}
                </div>
                {activePlan.verification.verificationStatus === 'STALE' && (
                  <p className="supporting">
                    Verification is stale because incident evidence or verification context changed.
                    Re-verify before approval.
                  </p>
                )}
                <KeyValueGrid
                  entries={[
                    {
                      key: 'simulator',
                      label: 'Simulator',
                      value: `${activePlan.verification.simulator} ${activePlan.verification.simulatorVersion}`,
                    },
                    {
                      key: 'state-hash',
                      label: 'State hash',
                      value: activePlan.verification.stateHash,
                      hash: true,
                    },
                    {
                      key: 'context-hash',
                      label: 'Context hash',
                      value: activePlan.verification.contextHash,
                      hash: true,
                    },
                  ]}
                />
                {activePlan.verification.consequences && (
                  <>
                    <h3>Expected consequences</h3>
                    <KeyValueGrid
                      entries={[
                        {
                          key: 'pressure-margin',
                          label: 'Pressure margin',
                          value:
                            activePlan.verification.consequences.pressureMarginM === null
                              ? null
                              : `${activePlan.verification.consequences.pressureMarginM.toFixed(1)} m`,
                        },
                        {
                          key: 'service-margin',
                          label: 'Service margin',
                          value:
                            activePlan.verification.consequences.serviceAvailabilityMargin === null
                              ? null
                              : `${(activePlan.verification.consequences.serviceAvailabilityMargin * 100).toFixed(1)} pp`,
                        },
                        {
                          key: 'sensitive',
                          label: 'Numerically sensitive',
                          value: activePlan.verification.consequences.numericallySensitive
                            ? 'yes'
                            : 'no',
                        },
                      ]}
                    />
                  </>
                )}
                {activePlan.verification.worstCaseConsequences && (
                  <>
                    <h3>Worst case</h3>
                    <KeyValueGrid
                      entries={[
                        {
                          key: 'worst-pressure',
                          label: 'Minimum pressure',
                          value: `${activePlan.verification.worstCaseConsequences.minimumPressureM.toFixed(1)} m`,
                        },
                        {
                          key: 'worst-service',
                          label: 'Service availability',
                          value: `${(activePlan.verification.worstCaseConsequences.serviceAvailability * 100).toFixed(1)}%`,
                        },
                      ]}
                    />
                  </>
                )}
                {activePlan.verification.rejectionCodes.length > 0 && (
                  <p className="supporting">
                    Rejection codes: {activePlan.verification.rejectionCodes.join(', ')}
                  </p>
                )}
              </>
            ) : (
              <EmptyState title="This plan has not yet been submitted for exact verification." />
            )}
          </Panel>
        ) : (
          <Panel title="No plan selected" eyebrow="VERIFICATION">
            <EmptyState title="No response plans available for this incident." />
          </Panel>
        )}
        {activePlan && (
          <Panel title="Action sequence" eyebrow="FULL PLAN">
            <PlanActionSequence plan={activePlan} />
          </Panel>
        )}
      </aside>
      <Panel title="Verified plan comparison" eyebrow="WNTR CONSEQUENCES" className="wide-panel">
        <PlanTable plans={incident.plans} />
      </Panel>
      <Panel
        title="Verified response Pareto frontier"
        eyebrow={`${frontierMode === 'posterior_weighted' ? 'POSTERIOR-WEIGHTED' : 'WORST-CASE'} CONTEXT`}
        className="wide-panel"
      >
        {incident.mode === 'LIVE' && frontierQuery.isLoading ? (
          <p role="status">Loading Pareto frontier…</p>
        ) : incident.mode === 'LIVE' && frontierQuery.isError ? (
          <EmptyState
            title="Pareto frontier unavailable."
            detail={(frontierQuery.error as Error).message}
          />
        ) : (
          <ParetoFrontier
            entries={frontier}
            selectedPlanId={activePlan?.id ?? null}
            onSelectPlan={selectPlan}
          />
        )}
      </Panel>
      <Panel title="Compare plans" eyebrow="GROUNDED EXPLANATION" className="wide-panel">
        {comparePlans ? (
          <p>{comparePlans.text}</p>
        ) : (
          <EmptyState title="No grounded plan comparison available for this incident." />
        )}
      </Panel>
      {activePlan?.status === 'REJECTED' && (
        <Panel
          title={`Why was ${activePlan.name} rejected?`}
          eyebrow="GROUNDED EXPLANATION"
          className="wide-panel"
        >
          {whyRejected ? (
            <p>{whyRejected.text}</p>
          ) : (
            <EmptyState title="No grounded rejection explanation available for this plan." />
          )}
        </Panel>
      )}
    </div>
  );
}
