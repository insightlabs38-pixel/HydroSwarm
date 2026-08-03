import type { IncidentView } from '../types';
import { Counterfactuals } from '../components/Counterfactuals';
import { EvidencePanel } from '../components/EvidencePanel';
import { HydraulicChart } from '../components/HydraulicChart';
import { OperationalMap } from '../components/OperationalMap';
import { Panel } from '../components/Panel';
import { PlanTable } from '../components/PlanTable';
import { StatusBadge } from '../components/StatusBadge';
import { Timeline } from '../components/Timeline';

export function Overview({ incident }: { incident: IncidentView }) {
  const leading = incident.candidates[0];
  const sensors = incident.nodes.flatMap((node) =>
    node.sensor ? [{ ...node.sensor, nodeId: node.id }] : [],
  );
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
            Leading source region <strong>{leading.nodeId}</strong> · calibrated candidate coverage{' '}
            {Math.round(incident.candidateCoverage * 100)}%
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
          {incident.approvalPending && (
            <button type="button" className="primary-action">
              Review Plan B approval
            </button>
          )}
        </div>
      </section>

      <Panel title="Live network" eyebrow="2D HYDRAULIC STATE" className="map-panel">
        <OperationalMap incident={incident} />
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
        <Panel
          title={`Collect sample at ${incident.recommendedSample.nodeId}`}
          eyebrow="RECOMMENDED NEXT ACTION"
        >
          <div className="sample-metrics">
            <span>
              <strong>{incident.recommendedSample.informationGain.toFixed(2)}</strong> information
              gain
            </span>
            <span>
              <strong>{incident.recommendedSample.delayMinutes}m</strong> delay
            </span>
            <span>
              <strong>{incident.recommendedSample.cost.toFixed(1)}</strong> cost
            </span>
          </div>
          <p>{incident.recommendedSample.rationale}</p>
          <button type="button" className="primary-action">
            Review sample request
          </button>
        </Panel>
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
        <HydraulicChart />
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
        title="Why Plan B?"
        eyebrow="VERIFIED EXPLANATION"
        className="wide-panel explanation-panel"
      >
        <p>{incident.explanation}</p>
        <div
          className="explanation-actions"
          role="group"
          aria-label="Available explanation questions"
        >
          <button type="button">Why this source?</button>
          <button type="button">Why this sample?</button>
          <button type="button">Why was Plan A rejected?</button>
          <button type="button">What uncertainty remains?</button>
        </div>
      </Panel>
    </div>
  );
}
