import { useEffect, useState } from 'react';
import { StatusBadge } from './StatusBadge';

interface GovernanceModel {
  id: string;
  name: string;
  parameters: number | null;
  top1: number | null;
  ece: number | null;
  latencyMs: number | null;
  decision: 'promoted' | 'rejected' | 'ineligible' | 'fallback';
}

interface GovernanceDocument {
  schemaVersion: string;
  models: GovernanceModel[];
}

function decisionTone(decision: GovernanceModel['decision']): 'good' | 'warn' | 'danger' | 'info' {
  switch (decision) {
    case 'promoted':
      return 'good';
    case 'fallback':
      return 'info';
    case 'rejected':
      return 'warn';
    case 'ineligible':
      return 'danger';
  }
}

function formatParameters(value: number | null): string {
  if (value === null) return '—';
  return `${(value / 1_000_000).toFixed(2)}M`;
}

function formatPercent(value: number | null): string {
  return value === null ? 'not evaluated' : `${(value * 100).toFixed(1)}%`;
}

function formatEce(value: number | null): string {
  return value === null ? '—' : value.toFixed(4);
}

function formatLatency(value: number | null): string {
  return value === null ? 'profile only' : `${value.toFixed(2)} ms`;
}

/**
 * Renders the governed model-evaluation/promotion table (overnight-plan.txt
 * Task 3.6). Values are read from a versioned static artifact
 * (public/model-governance.json, itself copied from the real governed
 * reports/results/*.json evaluation reports with provenance hashes
 * recorded) rather than hard-coded in this component.
 */
export function ModelGovernanceTable() {
  const [doc, setDoc] = useState<GovernanceDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/model-governance.json')
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<GovernanceDocument>;
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

  if (error) {
    return <p role="alert">Model governance data unavailable: {error}</p>;
  }
  if (!doc) {
    return <p role="status">Loading governed model evaluation…</p>;
  }

  return (
    <div className="table-scroll">
      <table className="benchmark-table">
        <caption>Governed model evaluation and promotion decisions</caption>
        <thead>
          <tr>
            <th>Model</th>
            <th>Parameters</th>
            <th>Top-1</th>
            <th>ECE</th>
            <th>Latency</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {doc.models.map((model) => (
            <tr key={model.id}>
              <th scope="row">{model.name}</th>
              <td>{formatParameters(model.parameters)}</td>
              <td>{formatPercent(model.top1)}</td>
              <td>{formatEce(model.ece)}</td>
              <td>{formatLatency(model.latencyMs)}</td>
              <td>
                <StatusBadge tone={decisionTone(model.decision)}>
                  {model.decision.toUpperCase()}
                </StatusBadge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
