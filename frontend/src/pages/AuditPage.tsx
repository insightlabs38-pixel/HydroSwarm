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
        {/* Not yet wired to the export/replay API (overnight-plan.txt Task
            3.4): disabled with an explicit reason rather than appearing
            functional while doing nothing on click. Chain validity is not
            asserted here since this console does not yet independently
            verify the hash chain client-side -- asserting VALID
            unconditionally would be exactly the false-VERIFIED-state
            problem the plan prohibits. */}
        <button type="button" className="primary-action" disabled title="Export is not yet connected to the live API">
          Export incident JSON
        </button>
        <button type="button" disabled title="Hash-chain verification is not yet connected to the live API">
          Verify hash chain
        </button>
        <span>Chain status: not independently verified by this console yet</span>
      </div>
    </div>
  );
}
