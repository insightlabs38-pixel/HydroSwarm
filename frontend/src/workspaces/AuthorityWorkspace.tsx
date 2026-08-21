import { useQuery } from '@tanstack/react-query';
import type { DecisionCertificate, IncidentView } from '../types';
import { fetchAuthorityCertificates } from '../api/authority';
import { demoAuthorityCertificates } from '../demoFixture';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { AuthorityBadge } from '../components/status/AuthorityBadge';
import { ApplicabilityBadge } from '../components/status/ApplicabilityBadge';
import { EmptyState } from '../components/common/EmptyState';

/** Frozen SYSTEM authority path for HydroCore-v5. Not per-incident. */
const SYSTEM_AUTHORITY_PATH = [
  { label: 'HydroCore-v5 Sentinel', level: 'ADVISORY' },
  { label: 'Calibrated fusion', level: 'CALIBRATED ADVISORY' },
  { label: 'OODDetector', level: 'DETERMINISTIC' },
  { label: 'Deterministic Scout', level: 'DETERMINISTIC' },
  { label: 'Deterministic Plan Generator', level: 'DETERMINISTIC' },
  { label: 'WNTR / EPANET', level: 'SIMULATOR VERIFIED' },
  { label: 'Human operator', level: 'HUMAN APPROVED' },
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
      <Panel title="Decision authority path" eyebrow="SYSTEM AUTHORITY">
        <p className="supporting">
          Frozen HydroCore-v5 system authority path. Not per-incident. Source fusion combines
          deterministic hydraulic/signature evidence with HydroCore-v5 Sentinel evidence before
          downstream deterministic authority checks.
        </p>
        <div className="table-scroll">
          <ol className="approval-hierarchy authority-path" aria-label="System authority path">
            {SYSTEM_AUTHORITY_PATH.map((step, index) => (
              <li key={step.label}>
                <span className="authority-path-label">{step.label}</span>
                <span className="authority-path-level">{step.level}</span>
                {index < SYSTEM_AUTHORITY_PATH.length - 1 && (
                  <span className="authority-path-arrow" aria-hidden="true">
                    →
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>
      </Panel>
      <Panel title="Current incident certificates" eyebrow="CURRENT INCIDENT CERTIFICATE">
        {incident.mode === 'LIVE' && authorityQuery.isLoading ? (
          <p role="status">Loading decision certificates…</p>
        ) : incident.mode === 'LIVE' && authorityQuery.isError ? (
          <EmptyState
            title="Decision certificates unavailable."
            detail={(authorityQuery.error as Error).message}
          />
        ) : certificates.length === 0 && incident.mode === 'REFERENCE' ? (
          <EmptyState
            title="Deterministic Reference replay does not claim live DecisionCertificates."
            detail="The deterministic Reference replay does not claim live HydroCore-v5 DecisionCertificates. Run Live Example to inspect incident-native authority certificates."
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
