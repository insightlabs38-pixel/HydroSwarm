import type { IncidentView } from '../types';

function CandidateBars({ title, values }: { title: string; values: IncidentView['candidates'] }) {
  return (
    <div className="evidence-column">
      <h3>{title}</h3>
      {values.map((candidate) => (
        <div className="probability-row" key={candidate.nodeId}>
          <span>{candidate.nodeId}</span>
          <div className="probability-track">
            <i style={{ width: `${candidate.probability * 100}%` }} />
          </div>
          <strong>{Math.round(candidate.probability * 100)}%</strong>
        </div>
      ))}
    </div>
  );
}

export function EvidencePanel({ incident }: { incident: IncidentView }) {
  return (
    <div>
      <div className="evidence-grid">
        <CandidateBars title="Before J123" values={incident.evidence.before} />
        <div className="evidence-arrow" aria-hidden="true">
          →
        </div>
        <CandidateBars title="After J123" values={incident.evidence.after} />
      </div>
      <div className="metric-strip" aria-label="Evidence change summary">
        <span>
          <strong>{Math.round(incident.evidence.uncertaintyReduction * 100)}%</strong> uncertainty
          reduction
        </span>
        <span>
          <strong>{incident.evidence.nodesRemoved}</strong> candidate nodes removed
        </span>
        <span>
          <strong>{incident.recommendedSample.delayMinutes} min</strong> sample delay
        </span>
      </div>
    </div>
  );
}
