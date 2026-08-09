import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { ModelGovernanceTable } from '../components/ModelGovernanceTable';

export function ValidationPage({ incident }: { incident: IncidentView }) {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">SCIENTIFIC VALIDATION</p>
        <h1>Benchmarks and operating range</h1>
        <p>
          Frozen regression, deterministic replay, scientific unit tests, and governed trained-
          checkpoint evaluation. No production-safety claim.
        </p>
      </header>
      <div className="validation-summary">
        <article>
          <strong>3</strong>
          <span>seeded golden runs</span>
        </article>
        <article>
          <strong>1</strong>
          <span>frozen network</span>
        </article>
        <article>
          <strong>96.0%</strong>
          <span>promoted checkpoint top-1 (HydroCore-S hybrid)</span>
        </article>
        <article>
          <strong>1.17 MB</strong>
          <span>Python allocation peak only</span>
        </article>
      </div>
      <Panel title="Model evaluation and promotion" eyebrow="GOVERNED RESULTS">
        <ModelGovernanceTable />
      </Panel>
      <Panel title="Operational benchmark table" eyebrow="FROZEN WNTR REGRESSION">
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
            <li>Missing sensors - scientific tests pass</li>
            <li>Noise and drift - scientific tests pass</li>
            <li>Delayed observations - controller tests pass</li>
            <li>Demand uncertainty - reconciliation tests pass</li>
            <li>Flow reversal - graph regression tests pass</li>
            <li>OOD evidence - abstention tests pass</li>
          </ul>
        </Panel>
        <Panel title="Known limitations" eyebrow="HONEST BOUNDARY">
          <ul className="warning-list">
            <li>Single-species simulated incidents only</li>
            <li>
              Unseen-topology transfer is measured but weak (a genuinely different network correctly
              triggers CAUTION and suppresses planning rather than demonstrating strong transfer
              accuracy)
            </li>
            <li>Frozen proof uses one compact reference network</li>
            <li>No autonomous actuator control</li>
            <li>Candidate coverage is a conformal target, not measured per-incident coverage</li>
            <li>Human engineering review is mandatory</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}
