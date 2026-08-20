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
import { PlanVerdictStrip } from '../components/plans/PlanVerdictStrip';
import { formatDisplayId } from '../displayId';

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
// UI-11.1 §3: when the incident genuinely has no response plans (planning
// suppressed -- e.g. invalid calibration or an out-of-validated-range
// reading), every panel below must say so honestly instead of drawing
// content that only makes sense once plans exist (a Pareto frontier or a
// "compare plans" explanation authored for a *different*, populated,
// scenario). This is the single source of truth for that state, and for
// the reason shown alongside it, reusing the same calibration/OOD signals
// ModeBanner already surfaces -- never a new suppression concept.
function planningSuppressionDetail(incident: IncidentView): string {
  if (!incident.calibrationValid) {
    return 'Calibration is invalid for this network. Planning is suppressed until a valid calibration artifact is available.';
  }
  if (incident.ood === 'OUTSIDE_VALIDATED_RANGE') {
    return "This incident is outside the model's validated operating range. Planning is suppressed pending additional evidence or operator review.";
  }
  return 'No response plans have been generated for this incident yet.';
}

export function ResponseWorkspace({ incident }: { incident: IncidentView }) {
  const { selectedPlanId, selectPlan, frontierMode } = useConsoleStore();
  const planningSuppressed = incident.plans.length === 0;
  const activePlan =
    incident.plans.find((plan) => plan.id === selectedPlanId) ??
    incident.plans.find((plan) => plan.id === incident.recommendedPlanId) ??
    incident.plans[0] ??
    null;

  const frontierQuery = useQuery({
    queryKey: ['frontier', incident.id, frontierMode],
    queryFn: ({ signal }) => fetchParetoFrontier(incident.id, frontierMode, signal),
    enabled: incident.mode === 'LIVE' && !planningSuppressed,
  });
  const frontier =
    incident.mode === 'LIVE'
      ? (frontierQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK' && !planningSuppressed
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
      {!planningSuppressed && (
        <PlanVerdictStrip
          plans={incident.plans}
          selectedPlanId={activePlan?.id ?? null}
          onSelect={selectPlan}
        />
      )}
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
        {planningSuppressed ? (
          <EmptyState
            title="Planning suppressed -- no response plans to compare."
            detail={planningSuppressionDetail(incident)}
          />
        ) : (
          <PlanTable plans={incident.plans} />
        )}
      </Panel>
      {incident.mode !== 'REFERENCE' && (
        <Panel
          title="Verified response Pareto frontier"
          eyebrow={`${frontierMode === 'posterior_weighted' ? 'POSTERIOR-WEIGHTED' : 'WORST-CASE'} CONTEXT`}
          className="wide-panel"
        >
          {planningSuppressed ? (
            <EmptyState
              title="Planning suppressed -- no Pareto frontier to display."
              detail={planningSuppressionDetail(incident)}
            />
          ) : incident.mode === 'LIVE' && frontierQuery.isLoading ? (
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
      )}
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
          eyebrow={`FULL PLAN · ${formatDisplayId(activePlan.id)}`}
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
        {planningSuppressed ? (
          <EmptyState
            title="Planning suppressed -- no plan comparison available."
            detail={planningSuppressionDetail(incident)}
          />
        ) : comparePlans ? (
          <p>{comparePlans.text}</p>
        ) : incident.mode === 'REFERENCE' ? (
          <div className="inspector-stack">
            <p className="eyebrow">REFERENCE NARRATIVE</p>
            <p>{incident.explanation || 'Deterministic reference workflow — no grounded plan explanation available.'}</p>
          </div>
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
