import { lazy, Suspense, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { IncidentView } from '../types';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates } from '../demoFixture';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { ApplicabilityBadge } from '../components/status/ApplicabilityBadge';
import { EmptyState } from '../components/common/EmptyState';
import { KeyValueGrid } from '../components/common/KeyValueGrid';

const OperationalMap = lazy(() =>
  import('../components/OperationalMap').then((module) => ({ default: module.OperationalMap })),
);

/**
 * ui-work.txt UI-3: governed source-localization workspace. Every value
 * here is either already on IncidentView (candidates, coverage,
 * calibration, disagreement, OOD, grounded WHY_SOURCE explanation) or
 * comes from the real GET /incidents/{id}/authority endpoint in LIVE
 * mode -- never inferred or fabricated (ui-work.txt 9.2).
 */
export function SourceWorkspace({ incident }: { incident: IncidentView }) {
  const authorityQuery = useQuery({
    queryKey: ['authority', incident.id],
    queryFn: ({ signal }) => fetchAuthorityCertificates(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const certificates =
    incident.mode === 'LIVE'
      ? (authorityQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoAuthorityCertificates
        : [];
  const sourceCertificate = certificates.find((cert) => cert.name === 'source_localization');
  const whySource = useMemo(
    () => incident.explanations.find((item) => item.intent === 'WHY_SOURCE'),
    [incident.explanations],
  );

  return (
    <div className="workspace-grid">
      <Panel title="Network" eyebrow="2D HYDRAULIC STATE" className="map-panel">
        <Suspense
          fallback={
            <div className="visual-loading" role="status">
              Loading offline network renderer…
            </div>
          }
        >
          <OperationalMap incident={incident} />
        </Suspense>
      </Panel>
      <aside className="right-rail" aria-label="Calibration and disagreement">
        <Panel title="Calibrated candidate set" eyebrow="CONFORMAL CALIBRATION">
          <dl className="key-value-grid">
            <div>
              <dt>Candidate-set size</dt>
              <dd>{incident.candidates.length}</dd>
            </div>
            <div>
              <dt>Conformal target</dt>
              <dd>{Math.round(incident.candidateCoverage * 100)}%</dd>
            </div>
            <div>
              <dt>Held-out measured coverage</dt>
              <dd>
                {typeof incident.measuredCoverage === 'number'
                  ? `${Math.round(incident.measuredCoverage * 100)}%`
                  : 'not measured'}
              </dd>
            </div>
            <div>
              <dt>Calibration state</dt>
              <dd>{incident.calibrationValid ? 'valid' : 'invalid'}</dd>
            </div>
          </dl>
          <p className="supporting">
            The conformal target sizes the calibrated candidate set for this network; it is not a
            per-incident correctness probability.
          </p>
        </Panel>
        <Panel title="Classical / neural disagreement" eyebrow="FUSION DIAGNOSTICS">
          <p className="supporting">
            Jensen-Shannon disagreement:{' '}
            <strong>{(incident.disagreement * 100).toFixed(1)}%</strong>
          </p>
          <StatusBadge tone={incident.ood === 'NORMAL' ? 'good' : 'warn'}>
            OOD {incident.ood}
          </StatusBadge>
          {incident.runtimeAnalysisMode === 'CLASSICAL_SAFE' && (
            <p className="supporting">
              Running in CLASSICAL_SAFE mode (the neural pipeline did not run for this incident);
              the exact reason is not yet exposed to the console.
            </p>
          )}
        </Panel>
      </aside>
      <Panel title="Ranked source candidates" eyebrow="SENTINEL" className="wide-panel">
        {incident.candidates.length === 0 ? (
          <EmptyState title="No source candidates for this incident." />
        ) : (
          <div className="evidence-column">
            {incident.candidates.map((candidate, index) => (
              <div className="probability-row" key={candidate.nodeId}>
                <span>
                  {index + 1}. {candidate.nodeId}
                </span>
                <div className="probability-track">
                  <i style={{ width: `${candidate.probability * 100}%` }} />
                </div>
                <strong>{Math.round(candidate.probability * 100)}%</strong>
              </div>
            ))}
          </div>
        )}
      </Panel>
      <Panel
        title="Source-localization authority"
        eyebrow="DECISION CERTIFICATE"
        className="wide-panel"
      >
        {incident.mode === 'LIVE' && authorityQuery.isLoading ? (
          <p role="status">Loading authority certificate…</p>
        ) : incident.mode === 'LIVE' && authorityQuery.isError ? (
          <EmptyState
            title="Authority certificate unavailable."
            detail={(authorityQuery.error as Error).message}
          />
        ) : sourceCertificate ? (
          <div className="inspector-stack">
            <div className="decision-badges">
              <AuthorityBadge authority={sourceCertificate.authority} />
              <ApplicabilityBadge applicability={sourceCertificate.applicability} />
              <StatusBadge tone={sourceCertificate.enabled ? 'good' : 'warn'}>
                {sourceCertificate.enabled ? 'ENABLED' : 'DISABLED'}
              </StatusBadge>
              <StatusBadge tone={sourceCertificate.calibrated ? 'good' : 'info'}>
                {sourceCertificate.calibrated ? 'CALIBRATED' : 'UNCALIBRATED'}
              </StatusBadge>
            </div>
            {sourceCertificate.suppressionReasons.length > 0 && (
              <p className="supporting">
                Suppression reasons: {sourceCertificate.suppressionReasons.join(', ')}
              </p>
            )}
            <KeyValueGrid
              entries={[
                { key: 'source', label: 'Source', value: sourceCertificate.source },
                {
                  key: 'model',
                  label: 'Model provenance',
                  value: sourceCertificate.provenance.model,
                  hash: true,
                },
                {
                  key: 'calibration',
                  label: 'Calibration provenance',
                  value: sourceCertificate.provenance.calibration,
                  hash: true,
                },
                {
                  key: 'network',
                  label: 'Network provenance',
                  value: sourceCertificate.provenance.network,
                  hash: true,
                },
              ]}
            />
          </div>
        ) : (
          <EmptyState title="No authority certificate available for this incident." />
        )}
      </Panel>
      <Panel title="Why this source?" eyebrow="GROUNDED EXPLANATION" className="wide-panel">
        {whySource ? (
          <>
            <p>{whySource.text}</p>
            {whySource.limitations.length > 0 && (
              <ul className="warning-list">
                {whySource.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <EmptyState title="No grounded source explanation available for this incident." />
        )}
      </Panel>
    </div>
  );
}
