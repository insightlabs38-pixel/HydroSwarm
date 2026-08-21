import { useEffect, useState } from 'react';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';

interface RegressionMetric {
  metric: string;
  value: string;
  comparison: string;
  status: 'PASS' | 'FAIL' | 'MEASURED';
}

interface RegressionEvidenceDocument {
  schemaVersion: string;
  generatedFrom: { path: string; sha256: string }[];
  note: string;
  metrics: RegressionMetric[];
  limitations: string[];
}

function statusTone(status: RegressionMetric['status']): 'good' | 'warn' | 'info' {
  if (status === 'PASS') return 'good';
  if (status === 'FAIL') return 'warn';
  return 'info';
}

/**
 * ui-improvements.txt: the reference/live IncidentView never carries
 * measured regression evidence (incident.benchmarks is always []) -- that
 * would require a live-computed regression suite this console does not
 * have. Rendering an empty "measured results" table under that heading
 * would look broken during the primary demo. Instead this page reads one
 * provenance-backed static system-level artifact
 * (public/system-regression-evidence.json, itself copied from the real
 * committed reports/results/*.json evaluation reports with hashes
 * recorded), independent of which incident happens to be open. See
 * ValidationPage for the separate M11.6 final-model scientific evaluation.
 */
export function BenchmarkPage() {
  const [doc, setDoc] = useState<RegressionEvidenceDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/system-regression-evidence.json')
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<RegressionEvidenceDocument>;
      })
      .then((data) => {
        if (!cancelled) setDoc(data);
      })
      .catch((fetchError: Error) => {
        if (!cancelled) setError(fetchError.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">SYSTEM REGRESSION &amp; PERFORMANCE</p>
        <h1>Regression and runtime benchmarks</h1>
        <p>
          Golden/reference WNTR regression, deterministic replay/regression facts, and runtime
          measurements. This is regression evidence, not final-model accuracy.
        </p>
      </header>
      <Panel title="Frozen golden-fixture regression evidence" eyebrow="FROZEN WNTR REGRESSION">
        {error ? (
          <EmptyState title="Regression evidence unavailable." detail={error} />
        ) : !doc ? (
          <p role="status">Loading regression evidence…</p>
        ) : (
          <>
            <div className="table-scroll">
              <table className="benchmark-table">
                <caption>Measured frozen WNTR regression and runtime metrics</caption>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Result</th>
                    <th>Comparison</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {doc.metrics.map((metric) => (
                    <tr key={metric.metric}>
                      <th scope="row">{metric.metric}</th>
                      <td>{metric.value}</td>
                      <td>{metric.comparison}</td>
                      <td>
                        <StatusBadge tone={statusTone(metric.status)}>{metric.status}</StatusBadge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="supporting">
              Source:{' '}
              {doc.generatedFrom.map((source, index) => (
                <span key={source.path}>
                  {index > 0 && ', '}
                  <span title={source.sha256} className="mono">
                    {source.path}
                  </span>
                </span>
              ))}
              .
            </p>
            {doc.limitations.length > 0 && (
              <ul className="warning-list">
                {doc.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            )}
          </>
        )}
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
