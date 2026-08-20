import { lazy, Suspense, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { EvidenceCertificate, IncidentView } from '../types';
import { fetchEvidenceCertificate } from '../api/sampling';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates, demoEvidenceCertificate } from '../demoFixture';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { EmptyState } from '../components/common/EmptyState';

const OperationalMap = lazy(() =>
  import('../components/OperationalMap').then((module) => ({ default: module.OperationalMap })),
);

const STATUS_LABEL: Record<EvidenceCertificate['status'], string> = {
  EVIDENCE_SUFFICIENT: 'EVIDENCE SUFFICIENT',
  CONTINUE_SAMPLING: 'CONTINUE SAMPLING',
  STOP_BUDGET_EXHAUSTED: 'STOP: SAMPLE BUDGET EXHAUSTED',
  STOP_NO_USEFUL_CANDIDATE: 'STOP: NO USEFUL CANDIDATE',
  STOP_ABSTAIN: 'STOP: ABSTAIN',
};

/**
 * ui-work.txt UI-4: deterministic evidence-value sampling workspace
 * (Scout). Every value here comes from the real GET
 * /incidents/{id}/evidence-certificate endpoint in LIVE mode -- "no
 * sample recommended" is rendered as an explicit stop certificate, not
 * an empty card (ui-work.txt 13.3).
 */
export function SamplingWorkspace({ incident }: { incident: IncidentView }) {
  const evidenceQuery = useQuery({
    queryKey: ['evidence-certificate', incident.id],
    queryFn: ({ signal }) => fetchEvidenceCertificate(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const authorityQuery = useQuery({
    queryKey: ['authority', incident.id],
    queryFn: ({ signal }) => fetchAuthorityCertificates(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const certificate: EvidenceCertificate | null =
    incident.mode === 'LIVE'
      ? (evidenceQuery.data ?? null)
      : incident.mode === 'DEMO_FALLBACK'
        ? demoEvidenceCertificate
        : null;
  const scoutCertificates =
    incident.mode === 'LIVE'
      ? (authorityQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoAuthorityCertificates
        : [];
  const scoutCertificate = scoutCertificates.find((cert) => cert.name === 'scout_recommendation');
  const whySample = useMemo(
    () => incident.explanations.find((item) => item.intent === 'WHY_SAMPLE'),
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
      <aside className="right-rail" aria-label="Evidence stop certificate">
        <Panel title="Evidence status" eyebrow="STOP CERTIFICATE">
          {incident.mode === 'REFERENCE' ? (
            <div className="inspector-stack">
              <StatusBadge tone="info">DETERMINISTIC REFERENCE SAMPLE</StatusBadge>
              <p className="supporting">
                Reference replay uses a deterministic classical workflow.
                The live EvidenceCertificate contract does not apply.
              </p>
              {scoutCertificate && (
                <p className="supporting">
                  Deterministic authority: <AuthorityBadge authority={scoutCertificate.authority} />
                </p>
              )}
            </div>
          ) : incident.mode === 'LIVE' && evidenceQuery.isLoading ? (
            <p role="status">Loading evidence certificate…</p>
          ) : incident.mode === 'LIVE' && evidenceQuery.isError ? (
            <EmptyState
              title="Evidence certificate unavailable."
              detail={(evidenceQuery.error as Error).message}
            />
          ) : certificate ? (
            <>
              <StatusBadge
                tone={
                  certificate.stop
                    ? certificate.status === 'EVIDENCE_SUFFICIENT'
                      ? 'good'
                      : 'warn'
                    : 'info'
                }
              >
                {STATUS_LABEL[certificate.status]}
              </StatusBadge>
              <p className="supporting">{certificate.message}</p>
              {scoutCertificate && (
                <p className="supporting">
                  Deterministic authority: <AuthorityBadge authority={scoutCertificate.authority} />
                </p>
              )}
            </>
          ) : (
            <EmptyState title="No evidence certificate available for this incident." />
          )}
        </Panel>
        {certificate && (
          <Panel title="Sample budget" eyebrow="SAMPLING STATE">
            <dl className="key-value-grid">
              <div>
                <dt>Posterior entropy</dt>
                <dd>{certificate.posteriorEntropyBits.toFixed(2)} bits</dd>
              </div>
              <div>
                <dt>Candidate-set size</dt>
                <dd>
                  {certificate.candidateSetSize} (
                  {certificate.candidateRegionCalibrated ? 'calibrated' : 'uncalibrated'})
                </dd>
              </div>
              <div>
                <dt>Remaining sample budget</dt>
                <dd>{certificate.sampleBudgetRemaining}</dd>
              </div>
              <div>
                <dt>Already sampled</dt>
                <dd>{certificate.alreadySampledNodes.join(', ') || 'none'}</dd>
              </div>
            </dl>
          </Panel>
        )}
      </aside>
      <Panel title="Next sample recommendation" eyebrow={incident.mode === 'REFERENCE' ? 'DETERMINISTIC REFERENCE SAMPLE' : 'DETERMINISTIC SCOUT'} className="wide-panel">
        {certificate?.recommendedSampleNode ? (
          <>
            <div className="candidate-hero">
              <strong>{certificate.recommendedSampleNode}</strong>
              {certificate.expectedInformationGainBits !== null && (
                <span>{certificate.expectedInformationGainBits.toFixed(2)} bits</span>
              )}
            </div>
            <dl className="key-value-grid">
              <div>
                <dt>Expected information gain</dt>
                <dd>
                  {certificate.expectedInformationGainBits === null
                    ? 'not evaluated'
                    : `${certificate.expectedInformationGainBits.toFixed(2)} bits`}
                </dd>
              </div>
              <div>
                <dt>Expected candidate reduction</dt>
                <dd>
                  {certificate.expectedCandidateReduction === null
                    ? 'not evaluated'
                    : certificate.expectedCandidateReduction}
                </dd>
              </div>
              <div>
                <dt>Accessible</dt>
                <dd>
                  {certificate.recommendedNodeAccessible === null
                    ? 'unknown'
                    : certificate.recommendedNodeAccessible
                      ? 'yes'
                      : 'no'}
                </dd>
              </div>
            </dl>
            <p className="supporting">
              Alternatives (remaining candidate nodes):{' '}
              {certificate.candidateNodes
                .filter((node) => node !== certificate.recommendedSampleNode)
                .join(', ') || 'none'}
            </p>
          </>
        ) : (
          <EmptyState
            title="No further sampling recommended."
            detail="This is a real decision state -- the sampling budget is exhausted, or active sampling found no further useful measurement for this incident."
          />
        )}
      </Panel>
      <Panel title="Why this sample?" eyebrow="GROUNDED EXPLANATION" className="wide-panel">
        {whySample ? (
          <>
            <p>{whySample.text}</p>
            {whySample.limitations.length > 0 && (
              <ul className="warning-list">
                {whySample.limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <EmptyState title="No grounded sample explanation available for this incident." />
        )}
      </Panel>
    </div>
  );
}
