import { useQuery } from '@tanstack/react-query';
import type { DecisionCertificate, IncidentView } from '../types';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates } from '../demoFixture';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { ApplicabilityBadge } from '../components/status/ApplicabilityBadge';
import { EmptyState } from '../components/common/EmptyState';

const AUTHORITY_LADDER = [
  'UNAVAILABLE',
  'SUPPRESSED',
  'ADVISORY',
  'CALIBRATED_ADVISORY',
  'DETERMINISTIC',
  'SIMULATOR_VERIFIED',
  'HUMAN_APPROVED',
] as const;

function certificateLabel(name: string): string {
  if (name === 'source_localization') return 'Source localization';
  if (name === 'scout_recommendation') return 'Sample recommendation';
  if (name === 'ood_state') return 'OOD decision';
  if (name.startsWith('plan_consequence:')) {
    return `Plan verification: ${name.slice('plan_consequence:'.length)}`;
  }
  return name;
}

function resultSummary(certificate: DecisionCertificate): string {
  const value = certificate.value;
  if (value === null || value === undefined) return '—';
  if (certificate.name === 'source_localization' && typeof value === 'object') {
    const topNode = (value as { top_node?: unknown }).top_node;
    return typeof topNode === 'string' ? topNode : '—';
  }
  if (typeof value === 'string') return value;
  if (typeof value === 'object') return 'see plan verification detail';
  return String(value);
}

/**
 * ui-work.txt UI-9 / 16: "Which subsystem is allowed to be authoritative
 * for which decision?" -- governance, not a generic ML metric dashboard.
 * Every row is a real Decision Certificate from the same GET
 * /incidents/{id}/authority endpoint UI-3 wired for the Source workspace;
 * this workspace just shows all of them together instead of only
 * source_localization.
 */
export function AuthorityWorkspace({ incident }: { incident: IncidentView }) {
  const authorityQuery = useQuery({
    queryKey: ['authority', incident.id],
    queryFn: ({ signal }) => fetchAuthorityCertificates(incident.id, signal),
    enabled: incident.mode === 'LIVE',
  });
  const certificates: DecisionCertificate[] =
    incident.mode === 'LIVE'
      ? (authorityQuery.data ?? [])
      : incident.mode === 'DEMO_FALLBACK'
        ? demoAuthorityCertificates
        : [];

  return (
    <div className="page-stack">
      <Panel title="Authority ladder" eyebrow="GOVERNANCE">
        <p className="supporting">
          Ordered from least to most operationally binding. A UI must never let a lower authority
          result visually outrank a higher one.
        </p>
        <ol className="approval-hierarchy" aria-label="Authority ladder">
          {AUTHORITY_LADDER.map((level) => (
            <li key={level}>{level.replaceAll('_', ' ')}</li>
          ))}
        </ol>
      </Panel>
      <Panel title="Decision certificates" eyebrow="CURRENT INCIDENT">
        {incident.mode === 'LIVE' && authorityQuery.isLoading ? (
          <p role="status">Loading decision certificates…</p>
        ) : incident.mode === 'LIVE' && authorityQuery.isError ? (
          <EmptyState
            title="Decision certificates unavailable."
            detail={(authorityQuery.error as Error).message}
          />
        ) : certificates.length === 0 ? (
          <EmptyState title="No decision certificates available for this incident." />
        ) : (
          <div className="table-scroll">
            <table className="benchmark-table">
              <caption>Decision Authority / Applicability Certificates</caption>
              <thead>
                <tr>
                  <th>Decision</th>
                  <th>Result</th>
                  <th>Source</th>
                  <th>Authority</th>
                  <th>Applicability</th>
                  <th>Enabled</th>
                  <th>Calibrated</th>
                  <th>Suppression reasons</th>
                </tr>
              </thead>
              <tbody>
                {certificates.map((certificate) => (
                  <tr key={certificate.name}>
                    <th scope="row">{certificateLabel(certificate.name)}</th>
                    <td>{resultSummary(certificate)}</td>
                    <td>{certificate.source}</td>
                    <td>
                      <AuthorityBadge authority={certificate.authority} />
                    </td>
                    <td>
                      <ApplicabilityBadge applicability={certificate.applicability} />
                    </td>
                    <td>
                      <StatusBadge tone={certificate.enabled ? 'good' : 'warn'}>
                        {certificate.enabled ? 'YES' : 'NO'}
                      </StatusBadge>
                    </td>
                    <td>
                      <StatusBadge tone={certificate.calibrated ? 'good' : 'info'}>
                        {certificate.calibrated ? 'YES' : 'NO'}
                      </StatusBadge>
                    </td>
                    <td>{certificate.suppressionReasons.join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
