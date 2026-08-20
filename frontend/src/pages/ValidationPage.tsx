import { Panel } from '../components/Panel';
import { V5Evidence } from '../components/V5Evidence';

export function ValidationPage() {
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">SCIENTIFIC VALIDATION</p>
        <h1>HydroCore-v5 final evaluation evidence</h1>
        <p>
          M11.6 locked evaluation. 125 incidents · 105 locked-final + 20 locked-topology.
          No production-safety claim.
        </p>
      </header>
      <V5Evidence />
      <Panel title="Known limitations" eyebrow="HONEST BOUNDARY">
        <ul className="warning-list">
          <li>Single-species simulated incidents only</li>
          <li>
            Sensor-dropout condition shows degraded coverage (66.7%) and actionable rate (60.0%) —
            source localization weakens under incomplete sensor data
          </li>
          <li>
            Measurement-noise condition degrades top-1 to 40.0% and actionable to 33.3% —
            noisy inputs meaningfully reduce prediction accuracy
          </li>
          <li>
            Ambiguity/disagreement condition degrades top-1 to 40.0% and MRR to 0.567 —
            conflicting classical/neural signatures impair ranking
          </li>
          <li>Unseen-topology transfer is measured but weak: predictive values are DESCRIPTIVE / NON-GATING</li>
          <li>Novel topology calibrated rate 0% — no operational authority granted</li>
          <li>Frozen proof uses one compact reference network</li>
          <li>No autonomous actuator control</li>
          <li>Candidate coverage is a conformal target, not measured per-incident coverage</li>
          <li>Human engineering review is mandatory</li>
        </ul>
      </Panel>
    </div>
  );
}
