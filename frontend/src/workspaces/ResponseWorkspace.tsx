import { lazy, Suspense, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { IncidentView } from '../types';
import { fetchParetoFrontier } from '../api/planning';
import { demoParetoFrontier } from '../demoFixture';
import { useConsoleStore } from '../store';
import { Panel } from '../components/Panel';
import { PlanTable } from '../components/PlanTable';
import { Counterfactuals } from '../components/Counterfactuals';
import { PlanActionSequence } from '../components/plans/PlanActionSequence';
import { ParetoFrontier } from '../components/plans/ParetoFrontier';
import { EmptyState } from '../components/common/EmptyState';

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
 *
 * ui-work.txt "UI-10.5" 2: no local `.right-rail` beside the map. The
 * selected plan's decision state (identity, VERIFIED/REJECTED/ABSTAINED,
 * CURRENT/STALE, simulator identity, hashes, margins, numerical
 * sensitivity, rejection/abstention reason) moved PRIMARY to the global
 * DecisionInspector (see DecisionInspector.tsx ResponseSummary); the full
 * forensic verification record (every hash, worst-case consequences,
 * verified-at) stays PRIMARY in TechnicalDock > Verification, which
 * already covered every one of those fields before this phase. This
 * workspace keeps the map, plan comparison, Pareto frontier, the
 * no-response counterfactual comparator (moved here from the old Incident
 * "everything dashboard" -- "UI-10.5" 3.D -- since it is a plan-comparison
 * view, not incident-overview content), the full ordered action sequence,
 * and the grounded comparison/rejection explanations as its own PRIMARY
 * content, per "UI-10.5" 2's RESPONSE ordering.
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
  const frontier =
    incident.mode === 'LIVE'
      ? (frontierQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoParetoFrontier
        : [];
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
      <Panel title="Network" eyebrow="RESPONSE ACTION OVERLAY" className="map-panel wide-panel">
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
      <Panel
        title="Counterfactual consequence branches"
        eyebrow="NO-RESPONSE COMPARATOR"
        className="wide-panel"
      >
        <Counterfactuals plans={incident.plans} />
      </Panel>
      {activePlan && (
        <Panel
          title="Action sequence"
          eyebrow={`FULL PLAN · ${activePlan.id}`}
          className="wide-panel"
        >
          {activePlan.verification?.verificationStatus === 'STALE' && (
            <p className="supporting">
              Verification is stale because incident evidence or verification context changed.
              Re-verify before approval.
            </p>
          )}
          <PlanActionSequence plan={activePlan} />
        </Panel>
      )}
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
