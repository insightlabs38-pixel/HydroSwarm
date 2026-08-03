import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';

export function ValidationPage({ incident }: { incident: IncidentView }) {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">SCIENTIFIC VALIDATION</p>
        <h1>Benchmarks and operating range</h1>
        <p>
          Held-out incidents, robustness conditions, and unseen-network transfer. No
          production-safety claim.
        </p>
      </header>
      <div className="validation-summary">
        <article>
          <strong>6,400</strong>
          <span>test incidents</span>
        </article>
        <article>
          <strong>6</strong>
          <span>networks tested</span>
        </article>
        <article>
          <strong>C-Town</strong>
          <span>unseen network</span>
        </article>
        <article>
          <strong>1.8 GB</strong>
          <span>peak RAM</span>
        </article>
      </div>
      <Panel title="Operational benchmark table" eyebrow="HYBRID HYDROSWARM-M">
        <div className="table-scroll">
          <table className="benchmark-table">
            <caption>Current frozen evaluation results</caption>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Result</th>
                <th>Comparison</th>
                <th>Gate</th>
              </tr>
            </thead>
            <tbody>
              {incident.benchmarks.map((benchmark) => (
                <tr key={benchmark.metric}>
                  <th scope="row">{benchmark.metric}</th>
                  <td>{benchmark.value}</td>
                  <td>{benchmark.comparison}</td>
                  <td>
                    <StatusBadge tone={benchmark.status === 'PASS' ? 'good' : 'warn'}>
                      {benchmark.status}
                    </StatusBadge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="validation-grid">
        <Panel title="Robustness sweep" eyebrow="IMPERFECT EVIDENCE">
          <ul className="check-list">
            <li>Missing sensors — evaluated</li>
            <li>Noise and drift — evaluated</li>
            <li>Delayed observations — evaluated</li>
            <li>Demand uncertainty — evaluated</li>
            <li>Flow reversal — evaluated</li>
            <li>OOD incident — abstained</li>
          </ul>
        </Panel>
        <Panel title="Known limitations" eyebrow="HONEST BOUNDARY">
          <ul className="warning-list">
            <li>Single-species simulated incidents only</li>
            <li>Unseen-network gap remains 13.2 points</li>
            <li>No autonomous actuator control</li>
            <li>Candidate coverage is marginal, not per-incident</li>
            <li>Human engineering review is mandatory</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
