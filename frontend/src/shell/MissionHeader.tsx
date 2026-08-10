import type { IncidentView } from '../types';
import { isCalibrationApplicable } from '../calibrationDisplay';
import { StatusBadge } from '../components/StatusBadge';

function modeTone(mode: IncidentView['mode']): 'good' | 'warn' | 'danger' | 'info' {
  switch (mode) {
    case 'LIVE':
      return 'good';
    case 'REFERENCE':
      return 'info';
    case 'REPLAY':
      return 'info';
    case 'DEMO_FALLBACK':
      return 'warn';
    case 'ERROR':
      return 'danger';
  }
}

const controllerStageLabel: Record<IncidentView['status'], string> = {
  SAMPLING: 'Sampling',
  PLANNING: 'Planning',
  APPROVAL: 'Awaiting approval',
  CLOSED: 'Closed',
};

/**
 * Derived, never fabricated: computed only from fields the console
 * already has for real (calibrationValid, ood, mode) -- not a live
 * `/api/readiness` poll (that belongs to the Validation workspace,
 * ui-work.txt 18, which can afford a dedicated network round trip; the
 * header must stay cheap and synchronous with the incident it is
 * already rendering).
 */
function readinessLabel(incident: IncidentView): {
  label: string;
  tone: 'good' | 'warn' | 'danger';
} {
  if (incident.mode === 'ERROR') return { label: 'NOT READY', tone: 'danger' };
  if (
    (isCalibrationApplicable(incident) && !incident.calibrationValid) ||
    incident.ood !== 'NORMAL'
  )
    return { label: 'DEGRADED', tone: 'warn' };
  return { label: 'READY', tone: 'good' };
}

function formatTimestamp(iso: string | null): string | null {
  if (!iso) return null;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString().slice(11, 19) + ' UTC';
}

export function MissionHeader({ incident }: { incident: IncidentView }) {
  const readiness = readinessLabel(incident);
  const timestamp = formatTimestamp(incident.generatedAt);
  return (
    <header className="mission-header">
      <div className="mission-header-brand">
        <span className="brand-mark" aria-hidden="true">
          H
        </span>
        <div>
          <strong>HydroSwarm</strong>
          <small>Offline Neuro-Hydraulic Incident Intelligence</small>
        </div>
      </div>
      <div className="mission-header-context">
        {incident.id ? (
          <>
            <span>
              Incident <strong>{incident.id}</strong>
            </span>
            <span>{incident.networkId}</span>
            <span>{controllerStageLabel[incident.status]}</span>
            {timestamp && <span className="mono">{timestamp}</span>}
          </>
        ) : (
          <span>No incident loaded</span>
        )}
      </div>
      <div className="mission-header-status">
        <StatusBadge tone="good">OFFLINE · LOCAL</StatusBadge>
        <StatusBadge tone={modeTone(incident.mode)}>
          <span aria-label={`Data mode ${incident.mode}`}>{incident.mode}</span>
        </StatusBadge>
        {incident.runtimeAnalysisMode && (
          <StatusBadge tone={incident.runtimeAnalysisMode === 'FULL_HYBRID' ? 'good' : 'warn'}>
            {incident.runtimeAnalysisMode}
          </StatusBadge>
        )}
        <StatusBadge tone={readiness.tone}>{readiness.label}</StatusBadge>
        {incident.ood !== 'NORMAL' && (
          <StatusBadge tone={incident.ood === 'OUTSIDE_VALIDATED_RANGE' ? 'danger' : 'warn'}>
            OOD {incident.ood}
          </StatusBadge>
        )}
        {incident.mode !== 'ERROR' && incident.runtimeMs > 0 && (
          <span
            className="runtime mono"
            aria-label={`Inference runtime ${incident.runtimeMs} milliseconds`}
          >
            {incident.runtimeMs} ms
          </span>
        )}
      </div>
    </header>
  );
}
