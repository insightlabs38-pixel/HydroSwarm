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
          End-to-end outcomes for localization, evidence efficiency, plan safety, transfer, runtime,
          and memory.
        </p>
      </header>
      <div className="validation-summary">
        <article>
          <strong>438 ms</strong>
          <span>CPU inference</span>
        </article>
        <article>
          <strong>90.8%</strong>
          <span>candidate coverage</span>
        </article>
        <article>
          <strong>1.8</strong>
          <span>median samples</span>
        </article>
        <article>
          <strong>7.1%</strong>
          <span>invalid plans</span>
        </article>
      </div>
      <Panel title="Benchmark evidence" eyebrow="SEEN + UNSEEN NETWORKS">
        <div className="table-scroll">
          <table className="benchmark-table">
            <caption>Frozen simulator-derived metrics; not live incident values</caption>
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
