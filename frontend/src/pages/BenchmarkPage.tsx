import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';

export function BenchmarkPage({ incident }: { incident: IncidentView }) {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">FROZEN EVALUATION RUN</p>
        <h1>Operational benchmarks</h1>
        <p>
          End-to-end outcomes regenerated from three seeded runs of the checked-in WNTR golden
          fixture. This is regression evidence, not field performance.
        </p>
      </header>
      <div className="validation-summary">
        <article>
          <strong>1.51 s</strong>
          <span>mean golden runtime</span>
        </article>
        <article>
          <strong>99.41%</strong>
          <span>true-source posterior</span>
        </article>
        <article>
          <strong>1.0 bit</strong>
          <span>sample information gain</span>
        </article>
        <article>
          <strong>14,723 mg</strong>
          <span>modeled reduction</span>
        </article>
      </div>
      <Panel title="Benchmark evidence" eyebrow="MEASURED + MISSING MADE EXPLICIT">
        <div className="table-scroll">
          <table className="benchmark-table">
            <caption>Measured frozen WNTR metrics; neural checkpoints explicitly not run</caption>
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
    </div>
  );
}
