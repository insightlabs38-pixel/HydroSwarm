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
 *
 * ui-work.txt "UI-10.5" 2: no local `.right-rail` beside the map -- the
 * compact calibration/disagreement/authority summary now lives PRIMARY in
 * the global DecisionInspector (see DecisionInspector.tsx SourceSummary).
 * This workspace stacks its full-detail panels (map, ranked candidates,
 * authority certificate, grounded explanation) below the large map
 * instead, per "UI-10.5" 2's "1. large operational network map; 2. ranked
 * candidate distribution; 3. Decision Certificate / authority detail;
 * 4. grounded explanation" ordering.
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
      <Panel title="Network" eyebrow="2D HYDRAULIC STATE" className="map-panel wide-panel">
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
        {incident.runtimeAnalysisMode === 'CLASSICAL_SAFE' && (
          <p className="supporting">
            Learned localization was unavailable for this incident; deterministic classical
            localization remains authoritative.
          </p>
        )}
      </Panel>
      <Panel
        title="Source-localization authority"
        eyebrow="DECISION CERTIFICATE"
        className="wide-panel"
      >
        {incident.mode === 'REFERENCE' ? (
          <div className="inspector-stack">
            <dl className="key-value-grid">
              <div>
                <dt>Mode</dt>
                <dd>Deterministic reference workflow</dd>
              </div>
              <div>
                <dt>Calibration</dt>
                <dd>Not applicable to reference replay</dd>
              </div>
              {incident.provenance.networkHash && (
                <div>
                  <dt>Network provenance</dt>
                  <dd title={incident.provenance.networkHash} className="mono">
                    {incident.provenance.networkHash.length > 16
                      ? `${incident.provenance.networkHash.slice(0, 8)}…${incident.provenance.networkHash.slice(-6)}`
                      : incident.provenance.networkHash}
                  </dd>
                </div>
              )}
              <div>
                <dt>Learned checkpoint</dt>
                <dd>Not claimed for reference replay</dd>
              </div>
              <div>
                <dt>Live V5 DecisionCertificate</dt>
                <dd>Not applicable — deterministic reference replay</dd>
              </div>
            </dl>
          </div>
        ) : incident.mode === 'LIVE' && authorityQuery.isLoading ? (
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
        ) : incident.mode === 'REFERENCE' ? (
          <div className="inspector-stack">
            <p className="eyebrow">REFERENCE NARRATIVE</p>
            <p>{incident.explanation || 'Deterministic reference workflow — no grounded model explanation available.'}</p>
          </div>
        ) : (
          <EmptyState title="No grounded source explanation available for this incident." />
        )}
      </Panel>
    </div>
  );
}
