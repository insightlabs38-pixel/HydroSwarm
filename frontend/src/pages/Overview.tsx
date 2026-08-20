import { lazy, Suspense } from 'react';
import type { IncidentView } from '../types';
import { deriveDecisionGate } from '../decisionGate';
import {
  candidateCoverageLabel,
  candidateCoverageValueText,
  isCalibrationApplicable,
  measuredCoverageValueText,
} from '../calibrationDisplay';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';
import { formatDisplayId } from '../displayId';
import { useConsoleStore } from '../store';
import { deriveWorkflowProgression } from '../workflow';

const OperationalMap = lazy(() =>
  import('../components/OperationalMap').then((module) => ({ default: module.OperationalMap })),
);

/**
 * ui-work.txt "UI-10.5" 3: the Incident workspace is a concise mission
 * overview, not the old "everything dashboard" embedded inside the new
 * shell. It answers "what incident, what state, where's the suspected
 * problem, what's next, is anything blocked, is approval pending" via a
 * compact strip + a dominant map + three small at-a-glance panels that
 * link to their dedicated workspace rather than repeating its full
 * content. The full plan comparison table, the counterfactual comparator,
 * the long explanation section, and the per-sensor health list all moved
 * to (or already existed in) their dedicated workspaces -- see
 * ResponseWorkspace (plan table, counterfactuals), SourceWorkspace
 * (ranked candidates, grounded explanation), SamplingWorkspace (evidence
 * certificate), and the map itself / TechnicalDock > Evidence (per-sensor
 * health, already color/shape-coded on the map per ui-work.txt 11).
 */
export function Overview({ incident }: { incident: IncidentView }) {
  const { setWorkspace } = useConsoleStore();
  const leading = incident.candidates[0] ?? null;
  const pendingPlan = incident.plans.find((plan) => plan.status === 'RECOMMENDED');
  const activePlan =
    incident.plans.find((plan) => plan.id === incident.selectedPlanId) ??
    incident.plans.find((plan) => plan.id === incident.recommendedPlanId) ??
    incident.plans[0] ??
    null;
  const { nextStep } = deriveWorkflowProgression(incident);
  const gate = deriveDecisionGate(incident);

  return (
    <div className="overview-grid">
      <section className="decision-banner" aria-labelledby="incident-heading">
        <div>
          <p className="eyebrow">INCIDENT DECISION STATE</p>
          <h1 id="incident-heading">
            {incident.status === 'APPROVAL'
              ? 'Verified response awaiting approval'
              : incident.status}
          </h1>
          <p>
            {leading ? (
              <>
                Leading source region <strong>{leading.nodeId}</strong> · candidate set size{' '}
                <strong>{incident.candidates.length}</strong> ·{' '}
                {candidateCoverageLabel(incident).toLowerCase()}{' '}
                {candidateCoverageValueText(incident)}
                {isCalibrationApplicable(incident) && !incident.calibrationValid
                  ? ' (calibration invalid for this network)'
                  : ''}
                {isCalibrationApplicable(incident) &&
                  typeof incident.measuredCoverage === 'number' && (
                    <> · held-out measured coverage {measuredCoverageValueText(incident)}</>
                  )}
              </>
            ) : (
              'No source candidates for this incident.'
            )}
            {' · next: '}
            <strong>{nextStep}</strong>
          </p>
        </div>
        <div className="decision-badges">
          <StatusBadge tone={gate.tone} label={gate.accessibleDetail}>
            {gate.pathLabel}
          </StatusBadge>
          <StatusBadge
            tone={incident.ood === 'NORMAL' ? 'good' : 'warn'}
            label={`OOD state ${incident.ood}`}
          >
            OOD {incident.ood}
          </StatusBadge>
          <StatusBadge
            tone={
              !isCalibrationApplicable(incident)
                ? 'info'
                : incident.calibrationValid
                  ? 'good'
                  : 'warn'
            }
          >
            {isCalibrationApplicable(incident)
              ? `CALIBRATION ${incident.calibrationValid ? 'VALID' : 'INVALID'}`
              : 'CALIBRATION N/A'}
          </StatusBadge>
          <StatusBadge tone={incident.approvalPending ? 'warn' : 'good'}>
            {incident.approvalPending ? 'HUMAN APPROVAL PENDING' : 'NO APPROVAL PENDING'}
          </StatusBadge>
          {incident.approvalPending && pendingPlan && (
            <button
              type="button"
              className="primary-action"
              onClick={() => setWorkspace('approval')}
            >
              Review {pendingPlan.name} approval
            </button>
          )}
        </div>
      </section>

      <Panel title="Live network" eyebrow="2D HYDRAULIC STATE" className="map-panel wide-panel">
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

      <div className="incident-summary-row wide-panel">
        <Panel title="Source" eyebrow="SENTINEL">
          {leading ? (
            <>
              <div className="candidate-hero">
                <strong>{leading.nodeId}</strong>
                <span>{Math.round(leading.probability * 100)}%</span>
              </div>
              <p className="supporting">
                {incident.candidates.length} candidate(s) · disagreement{' '}
                {typeof incident.disagreement === 'number'
                  ? `${(incident.disagreement * 100).toFixed(1)}%`
                  : 'not measured'}
              </p>
            </>
          ) : (
            <EmptyState title="No source candidates for this incident." />
          )}
          <button type="button" className="panel-nav-link" onClick={() => setWorkspace('source')}>
            Open Source workspace →
          </button>
        </Panel>

        <Panel title="Evidence / sampling" eyebrow="DETERMINISTIC SCOUT">
          {incident.recommendedSample ? (
            <>
              <div className="candidate-hero">
                <strong>{incident.recommendedSample.nodeId}</strong>
                <span>{incident.recommendedSample.informationGain.toFixed(2)} bits</span>
              </div>
              <p className="supporting">{incident.recommendedSample.rationale}</p>
            </>
          ) : (
            <EmptyState
              title="No further sampling recommended."
              detail="Sampling budget exhausted, or active sampling found no further useful measurement."
            />
          )}
          <button type="button" className="panel-nav-link" onClick={() => setWorkspace('sampling')}>
            Open Sampling workspace →
          </button>
        </Panel>

        <Panel title="Response" eyebrow="DETERMINISTIC PLANNER">
          {activePlan ? (
            <>
              <p className="supporting">
                <span title={activePlan.id}>{formatDisplayId(activePlan.id)}</span> ·{' '}
                {activePlan.name}
              </p>
              <StatusBadge
                tone={
                  activePlan.verification
                    ? activePlan.verification.decision === 'VERIFIED'
                      ? 'good'
                      : activePlan.verification.decision === 'REJECTED'
                        ? 'danger'
                        : 'warn'
                    : 'info'
                }
              >
                {activePlan.verification ? activePlan.verification.decision : activePlan.status}
              </StatusBadge>
            </>
          ) : (
            <EmptyState title="No response plans available for this incident." />
          )}
          <button type="button" className="panel-nav-link" onClick={() => setWorkspace('response')}>
            Open Response workspace →
          </button>
        </Panel>
      </div>
    </div>
  );
}
