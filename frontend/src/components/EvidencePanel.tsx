import type { IncidentView } from '../types';

function shortHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 8)}…` : hash;
}

/**
 * Renders real observed-evidence rounds and the current calibrated
 * candidate set. Previously this rendered a "before/after" candidate
 * probability comparison seeded from an always-empty `before` array and a
 * hard-coded-zero uncertainty-reduction metric -- fabricated-looking
 * content built from real (but absent) data. ui-work.txt 8.4: "Do not
 * render fake before/after candidate probabilities from empty arrays. Use
 * actual evidence history / posterior history. If only counts/hashes are
 * available, show those."
 */
export function EvidencePanel({ incident }: { incident: IncidentView }) {
  return (
    <div>
      <div className="table-scroll">
        <table className="benchmark-table">
          <caption>Observed evidence rounds</caption>
          <thead>
            <tr>
              <th scope="col">Round</th>
              <th scope="col">Observations</th>
              <th scope="col">Valid concentration readings</th>
              <th scope="col">Sensor nodes</th>
              <th scope="col">Evidence hash</th>
            </tr>
          </thead>
          <tbody>
            {incident.evidenceHistory.length === 0 ? (
              <tr>
                <td colSpan={5}>No sampling rounds recorded for this incident yet.</td>
              </tr>
            ) : (
              incident.evidenceHistory.map((round) => (
                <tr key={round.roundIndex}>
                  <td>{round.roundIndex}</td>
                  <td>{round.observationCount}</td>
                  <td>{round.validConcentrationCount}</td>
                  <td>{round.sensorNodes.join(', ') || '—'}</td>
                  <td>
                    <code title={round.evidenceHash}>{shortHash(round.evidenceHash)}</code>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <div className="evidence-column">
        <h3>Current calibrated candidate set</h3>
        {incident.candidates.map((candidate) => (
          <div className="probability-row" key={candidate.nodeId}>
            <span>{candidate.nodeId}</span>
            <div className="probability-track">
              <i style={{ width: `${candidate.probability * 100}%` }} />
            </div>
            <strong>{Math.round(candidate.probability * 100)}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}
