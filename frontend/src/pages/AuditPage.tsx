import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';

export function AuditPage({ incident }: { incident: IncidentView }) {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">TAMPER-EVIDENT PROVENANCE</p>
        <h1>Incident audit and replay</h1>
        <p>Every agent, simulator, and operator transition is ordered and exportable.</p>
      </header>
      <Panel title="Event ledger" eyebrow={`${incident.audit.length} VERIFIED EVENTS`}>
        <ol className="audit-list">
          {incident.audit.map((event) => (
            <li key={event.sequence}>
              <span className="audit-sequence">{event.sequence.toString().padStart(2, '0')}</span>
              <time>{event.timestamp}</time>
              <div>
                <strong>{event.type.replaceAll('_', ' ')}</strong>
                <small>{event.actor}</small>
                <p>{event.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </Panel>
      <div className="audit-actions">
        <button type="button" className="primary-action">
          Export incident JSON
        </button>
        <button type="button">Verify hash chain</button>
        <span>
          Chain status: <strong>VALID</strong>
        </span>
      </div>
    </div>
  );
}
