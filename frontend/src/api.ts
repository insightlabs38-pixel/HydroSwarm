import { demoIncident } from './demoFixture';
import type { AuditEvent, IncidentView } from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';
const INCIDENT_ID = import.meta.env.VITE_INCIDENT_ID as string | undefined;

interface ApiIncidentState {
  incident_id: string;
  network_id: string;
  status: IncidentView['status'];
  ood_level: IncidentView['ood'];
  approval_pending: boolean;
  disagreement_js: number | null;
  candidates: { node_probabilities: Record<string, number>; coverage_target: number } | null;
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) throw new Error(`HydroSwarm API ${response.status}: ${response.statusText}`);
  return (await response.json()) as T;
}

function eventFromApi(event: Record<string, unknown>): AuditEvent {
  return {
    sequence: Number(event.sequence),
    timestamp: String(event.timestamp).slice(11, 19),
    type: String(event.event_type),
    actor: String(event.actor),
    detail: JSON.stringify(event.payload),
  };
}

export async function fetchIncident(signal?: AbortSignal): Promise<IncidentView> {
  await request<{ status: string }>('/health', signal);
  if (!INCIDENT_ID) throw new Error('No active incident configured');
  const [state, events] = await Promise.all([
    request<ApiIncidentState>(`/incidents/${INCIDENT_ID}`, signal),
    request<Record<string, unknown>[]>(`/incidents/${INCIDENT_ID}/events`, signal),
  ]);
  const candidates = Object.entries(state.candidates?.node_probabilities ?? {})
    .map(([nodeId, probability]) => ({ nodeId, probability }))
    .sort((a, b) => b.probability - a.probability);
  return {
    ...demoIncident,
    id: state.incident_id,
    networkId: state.network_id,
    status: state.status,
    source: 'api',
    ood: state.ood_level,
    approvalPending: state.approval_pending,
    candidateCoverage: state.candidates?.coverage_target ?? 0,
    disagreement: state.disagreement_js ?? 0,
    candidates: candidates.length ? candidates : demoIncident.candidates,
    audit: events.map(eventFromApi),
  };
}

export async function fetchIncidentWithFallback(signal?: AbortSignal): Promise<IncidentView> {
  try {
    return await fetchIncident(signal);
  } catch {
    return demoIncident;
  }
}
