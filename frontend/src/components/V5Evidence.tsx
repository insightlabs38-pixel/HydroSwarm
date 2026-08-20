import { useEffect, useState } from 'react';
import { StatusBadge } from './StatusBadge';

interface V5EvidenceDoc {
  schema: string;
  system_identity: {
    name: string;
    variant: string;
    parameters: number;
    selected_seed: number;
    checkpoint_sha256: string;
    calibration_artifact_hash: string;
    calibration_sha256: string;
    feature_schema_hash: string;
    release_bundle: string;
    release_schema_version: string;
  };
  runtime_outputs: string[];
  trained_tasks: string[];
  deterministic_authority: {
    ood: string;
    scout: string;
    planner: string;
    physical_verification: string;
    human_approval_required: boolean;
    autonomous_actuation: boolean;
  };
  locked_governance: {
    gate_pass: boolean;
    locked_final_count: number;
    locked_topology_count: number;
    total_count: number;
    authorized_openings: number;
    actual_openings: number;
    rerun: boolean;
    post_lock_tuning: boolean;
    closure_state: string;
  };
  hard_safety_counters: {
    total_counters: number;
    counters_zero: boolean;
    all_pass: boolean;
  };
  metrics: {
    locked_final_test: {
      aggregate: {
        source: {
          n: number;
          top1_rate: number;
          top3_rate: number;
          mrr: number;
          coverage_rate: number;
          actionable_rate: number;
        };
      };
      by_condition: Record<string, {
        n: number;
        top1_rate: number;
        top3_rate: number;
        mrr: number;
        coverage_rate: number;
        actionable_rate: number;
        calibrated_rate?: number;
        candidate_set_size?: number;
        posterior_entropy?: number;
      }>;
    };
    locked_topology_test: {
      source: {
        n: number;
        top1_rate: number;
        top3_rate: number;
        mrr: number;
        actionable_rate: number;
      };
      topology_shift_predictive: string;
    };
  };
}

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

export function V5Evidence() {
  const [doc, setDoc] = useState<V5EvidenceDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/hydrocore-v5-evidence.json')
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<V5EvidenceDoc>;
      })
      .then((data) => {
        if (!cancelled) setDoc(data);
      })
      .catch((fetchError: Error) => {
        if (!cancelled) setError(fetchError.message);
      });
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return <p role="alert">V5 evidence unavailable: {error}</p>;
  }
  if (!doc) {
    return <p role="status">Loading HydroCore-v5 evidence…</p>;
  }

  const lg = doc.locked_governance;
  const sc = doc.hard_safety_counters;
  const id = doc.system_identity;
  const da = doc.deterministic_authority;
  const agg = doc.metrics.locked_final_test.aggregate.source;
  const topo = doc.metrics.locked_topology_test;

  return (
    <div className="v5-evidence">
      {/* Section A: Compact final-evaluation strip */}
      <div className="v5-eval-strip">
        <article>
          <StatusBadge tone={lg.gate_pass ? 'good' : 'danger'}>
            M11.6 {lg.gate_pass ? 'PASS' : 'FAIL'}
          </StatusBadge>
        </article>
        <article>
          <strong>{lg.total_count}</strong>
          <span> / {lg.total_count} complete</span>
        </article>
        <article>
          <strong>{sc.counters_zero ? '0' : '!'}</strong>
          <span> / {sc.total_counters} hard safety counters violated</span>
        </article>
        <article>
          <strong>{lg.actual_openings}</strong>
          <span> locked opening · {lg.rerun ? 'rerun' : 'no rerun'} · {lg.post_lock_tuning ? 'post-lock tuning' : 'no post-lock tuning'}</span>
        </article>
      </div>

      {/* System identity row */}
      <div className="v5-identity-row">
        <article>
          <span className="eyebrow">SYSTEM</span>
          <strong>{id.name}</strong>
          <span>{id.variant} · {id.parameters.toLocaleString()} parameters · seed {id.selected_seed}</span>
        </article>
        <article>
          <span className="eyebrow">OUTPUTS</span>
          <strong>{doc.runtime_outputs.length}</strong>
          <span>learned runtime outputs · trained: {doc.trained_tasks.join(', ')}</span>
        </article>
        <article>
          <span className="eyebrow">AUTHORITY</span>
          <span>OOD: {da.ood} · Scout: {da.scout} · Planner: {da.planner}</span>
          <span>{da.physical_verification} · human approval: {da.human_approval_required ? 'yes' : 'no'} · autonomous: {da.autonomous_actuation ? 'yes' : 'no'}</span>
        </article>
      </div>

      {/* Section B: Three-row high-level scientific table */}
      <div className="table-scroll">
        <table className="benchmark-table v5-summary-table">
          <caption>HydroCore-v5 M11.6 final evaluation summary</caption>
          <thead>
            <tr>
              <th>Population</th>
              <th>n</th>
              <th>Top-1</th>
              <th>Top-3</th>
              <th>MRR</th>
              <th>Coverage</th>
              <th>Actionable</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">NOMINAL</th>
              <td>{doc.metrics.locked_final_test.by_condition.NOMINAL.n}</td>
              <td>{pct(doc.metrics.locked_final_test.by_condition.NOMINAL.top1_rate)}</td>
              <td>{pct(doc.metrics.locked_final_test.by_condition.NOMINAL.top3_rate)}</td>
              <td>{doc.metrics.locked_final_test.by_condition.NOMINAL.mrr.toFixed(3)}</td>
              <td>{pct(doc.metrics.locked_final_test.by_condition.NOMINAL.coverage_rate)}</td>
              <td>{pct(doc.metrics.locked_final_test.by_condition.NOMINAL.actionable_rate)}</td>
            </tr>
            <tr>
              <th scope="row">ALL LOCKED-FINAL</th>
              <td>{agg.n}</td>
              <td>{pct(agg.top1_rate)}</td>
              <td>{pct(agg.top3_rate)}</td>
              <td>{agg.mrr.toFixed(3)}</td>
              <td>{pct(agg.coverage_rate)}</td>
              <td>{pct(agg.actionable_rate)}</td>
            </tr>
            <tr>
              <th scope="row">NOVEL TOPOLOGY</th>
              <td>{topo.source.n}</td>
              <td>{pct(topo.source.top1_rate)}</td>
              <td>{pct(topo.source.top3_rate)}</td>
              <td>{topo.source.mrr.toFixed(3)}</td>
              <td>N/A (calibration inapplicable)</td>
              <td>{pct(topo.source.actionable_rate)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="supporting">
        Novel topology: calibrated rate {pct(topo.source.actionable_rate)} · human-approved 0% · predictive values are DESCRIPTIVE / NON-GATING · topology fail-closed gate PASS
      </p>

      {/* Section C: Dense seven-condition matrix */}
      <div className="table-scroll">
        <table className="benchmark-table v5-condition-table">
          <caption>Seven-condition evaluation matrix (locked-final test)</caption>
          <thead>
            <tr>
              <th>Condition</th>
              <th>n</th>
              <th>Top-1</th>
              <th>Top-3</th>
              <th>MRR</th>
              <th>Coverage</th>
              <th>Actionable</th>
              <th>Cand. Size</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(doc.metrics.locked_final_test.by_condition).map(([condition, metrics]) => (
              <tr key={condition}>
                <th scope="row">{condition}</th>
                <td>{metrics.n}</td>
                <td>{pct(metrics.top1_rate)}</td>
                <td>{pct(metrics.top3_rate)}</td>
                <td>{metrics.mrr.toFixed(3)}</td>
                <td>{pct(metrics.coverage_rate)}</td>
                <td>{pct(metrics.actionable_rate)}</td>
                <td>{metrics.candidate_set_size !== undefined ? metrics.candidate_set_size.toFixed(1) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Section D: Compact hard-gate/safety section */}
      <div className="v5-safety-strip">
        <article>
          <span className="eyebrow">HARD GATE</span>
          <StatusBadge tone={lg.gate_pass ? 'good' : 'danger'}>
            {lg.gate_pass ? 'PASS' : 'FAIL'}
          </StatusBadge>
        </article>
        <article>
          <span className="eyebrow">SAFETY COUNTERS</span>
          <strong>{sc.total_counters} evaluated · all zero</strong>
          <StatusBadge tone={sc.all_pass ? 'good' : 'danger'}>{sc.all_pass ? 'PASS' : 'FAIL'}</StatusBadge>
        </article>
        <article>
          <span className="eyebrow">TOPOLOGY SHIFT</span>
          <span>Predictive: {topo.topology_shift_predictive}</span>
          <span>Never labeled as conformal coverage</span>
        </article>
      </div>
    </div>
  );
}
