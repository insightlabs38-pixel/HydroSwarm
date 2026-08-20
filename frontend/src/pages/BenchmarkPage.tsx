import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';

export function BenchmarkPage({ incident }: { incident: IncidentView }) {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">SYSTEM REGRESSION &amp; PERFORMANCE</p>
        <h1>Regression and runtime benchmarks</h1>
        <p>
          Golden/reference WNTR regression, deterministic replay/regression facts,
          and runtime measurements. This is regression evidence, not final-model accuracy.
        </p>
      </header>
      <Panel title="Reference fixture / regression evidence" eyebrow="FROZEN WNTR REGRESSION">
        <div className="table-scroll">
          <table className="benchmark-table">
            <caption>Measured frozen WNTR regression metrics</caption>
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
      <Panel title="Runtime & system characteristics" eyebrow="PERFORMANCE">
        <ul className="check-list">
          <li>Exact verification: WNTR / EPANET simulator</li>
          <li>Offline/local architecture — no network dependencies</li>
          <li>Human approval required before any action</li>
          <li>No autonomous actuation</li>
          <li>Deterministic OOD detection (OODDetector)</li>
          <li>Deterministic scout (rank_sample_locations)</li>
          <li>Deterministic planner (generate_response_plans)</li>
        </ul>
      </Panel>
    </div>
  );
}
