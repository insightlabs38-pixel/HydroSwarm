import { lazy, Suspense } from 'react';
import type { IncidentView } from '../types';
import { Counterfactuals } from '../components/Counterfactuals';
import { EvidencePanel } from '../components/EvidencePanel';
import { Panel } from '../components/Panel';
import { PlanTable } from '../components/PlanTable';
import { StatusBadge } from '../components/StatusBadge';
import { Timeline } from '../components/Timeline';

const OperationalMap = lazy(() =>
  import('../components/OperationalMap').then((module) => ({ default: module.OperationalMap })),
);
const HydraulicChart = lazy(() =>
  import('../components/HydraulicChart').then((module) => ({ default: module.HydraulicChart })),
);

export function Overview({ incident }: { incident: IncidentView }) {
  const leading = incident.candidates[0];
  const sensors = incident.nodes.flatMap((node) =>
    node.sensor ? [{ ...node.sensor, nodeId: node.id }] : [],
  );
  const pendingPlan = incident.plans.find((plan) => plan.status === 'RECOMMENDED');
  const rejectedPlan = incident.plans.find((plan) => plan.status === 'REJECTED');
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
            Leading source region <strong>{leading.nodeId}</strong> · candidate set size{' '}
            <strong>{incident.candidates.length}</strong> · conformal target{' '}
            {Math.round(incident.candidateCoverage * 100)}%
            {incident.calibrationValid ? '' : ' (calibration invalid for this network)'}
            {typeof incident.measuredCoverage === 'number' && (
              <> · held-out measured coverage {Math.round(incident.measuredCoverage * 100)}%</>
            )}
          </p>
        </div>
        <div className="decision-badges">
          <StatusBadge
            tone={incident.ood === 'NORMAL' ? 'good' : 'warn'}
            label={`OOD state ${incident.ood}`}
          >
            OOD {incident.ood}
          </StatusBadge>
          <StatusBadge tone={incident.approvalPending ? 'warn' : 'good'}>
            {incident.approvalPending ? 'HUMAN APPROVAL PENDING' : 'NO APPROVAL PENDING'}
          </StatusBadge>
          {incident.approvalPending && pendingPlan && (
            <button
              type="button"
              className="primary-action"
              disabled
              title="Plan approval is not yet connected to the live API"
            >
              Review {pendingPlan.name} approval
            </button>
          )}
        </div>
      </section>

      <Panel title="Live network" eyebrow="2D HYDRAULIC STATE" className="map-panel">
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
      <aside className="right-rail" aria-label="Incident evidence and actions">
        <Panel title="Source candidates" eyebrow="SENTINEL">
          <div className="candidate-hero">
            <strong>{leading.nodeId}</strong>
            <span>{Math.round(leading.probability * 100)}%</span>
          </div>
          {incident.candidates.slice(1).map((candidate) => (
            <div className="compact-row" key={candidate.nodeId}>
              <span>{candidate.nodeId}</span>
              <strong>{Math.round(candidate.probability * 100)}%</strong>
            </div>
          ))}
          <p className="supporting">
            Classical ↔ neural disagreement: {(incident.disagreement * 100).toFixed(1)}% · low
          </p>
        </Panel>
        {incident.recommendedSample ? (
          <Panel
            title={`Collect sample at ${incident.recommendedSample.nodeId}`}
            eyebrow="RECOMMENDED NEXT ACTION"
          >
            <div className="sample-metrics">
              <span>
                <strong>{incident.recommendedSample.informationGain.toFixed(2)}</strong>{' '}
                information gain
              </span>
              <span>
                <strong>{incident.recommendedSample.delayMinutes}m</strong> delay
              </span>
              <span>
                <strong>{incident.recommendedSample.cost.toFixed(1)}</strong> cost
              </span>
            </div>
            <p>{incident.recommendedSample.rationale}</p>
            <button
              type="button"
              className="primary-action"
              disabled
              title="Sample-request review is not yet connected to the live API"
            >
              Review sample request
            </button>
          </Panel>
        ) : (
          <Panel title="No further sampling recommended" eyebrow="RECOMMENDED NEXT ACTION">
            <p className="supporting">
              The sampling budget is exhausted, or active sampling found no further useful
              measurement for this incident.
            </p>
          </Panel>
        )}
        <Panel title="Sensor health" eyebrow="DATA QUALITY">
          {sensors.map((sensor) => (
            <div className="sensor-row" key={sensor.id}>
              <div>
                <strong>
                  {sensor.id} · {sensor.nodeId}
                </strong>
                <small>
                  {sensor.ageMinutes} min old · quality {Math.round(sensor.quality * 100)}%
                </small>
              </div>
              <StatusBadge tone={sensor.health === 'HEALTHY' ? 'good' : 'warn'}>
                {sensor.health}
              </StatusBadge>
            </div>
          ))}
        </Panel>
      </aside>
      <Panel
        title="Sensor and hydraulic profile"
        eyebrow="OBSERVED EVIDENCE"
        className="hydraulic-panel"
      >
        <Suspense
          fallback={
            <div className="chart visual-loading" role="status">
              Loading hydraulic chart…
            </div>
          }
        >
          <HydraulicChart />
        </Suspense>
      </Panel>
      <Panel title="Incident replay" eyebrow="AUDITABLE SEQUENCE" className="timeline-panel">
        <Timeline events={incident.audit} />
      </Panel>
      <Panel
        title="What changed after the sample?"
        eyebrow="EVIDENCE CONTRACTION"
        className="wide-panel"
      >
        <EvidencePanel incident={incident} />
      </Panel>
      <Panel
        title="Counterfactual consequence branches"
        eyebrow="SYNCHRONIZED AT 08:40"
        className="wide-panel"
      >
        <Counterfactuals plans={incident.plans} />
      </Panel>
      <Panel title="Verified plan comparison" eyebrow="WNTR CONSEQUENCES" className="wide-panel">
        <PlanTable plans={incident.plans} />
      </Panel>
      <Panel
        title={pendingPlan ? `Why ${pendingPlan.name}?` : 'Verified explanation'}
        eyebrow="VERIFIED EXPLANATION"
        className="wide-panel explanation-panel"
      >
        <p>{incident.explanation}</p>
        <div
          className="explanation-actions"
          role="group"
          aria-label="Available explanation questions"
        >
          {/* Not yet wired to the explanation API (overnight-plan.txt Task
              3.4): disabled with an explicit reason rather than appearing
              functional while doing nothing on click. */}
          <button type="button" disabled title="Explanation Q&A is not yet connected to the live API">
            Why this source?
          </button>
          <button type="button" disabled title="Explanation Q&A is not yet connected to the live API">
            Why this sample?
          </button>
          {rejectedPlan && (
            <button type="button" disabled title="Explanation Q&A is not yet connected to the live API">
              Why was {rejectedPlan.name} rejected?
            </button>
          )}
          <button type="button" disabled title="Explanation Q&A is not yet connected to the live API">
            What uncertainty remains?
          </button>
        </div>
      </Panel>
    </div>
  );
}
