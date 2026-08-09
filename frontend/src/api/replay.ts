import type { AuditEvent, ReplayResult } from '../types';
import { requestJson } from './client';

/** Mirrors only the IncidentState fields this console renders (see
 * types.ts's ReplayIncidentState doc comment) -- not the full backend
 * model, since observations/candidates/disagreement_js have no current
 * consumer here. */
interface ApiIncidentState {
  incident_id: string;
  network_id: string;
  status: string;
  sample_count: number;
  approval_pending: boolean;
  ood_level: string;
  exact_simulations_used: number;
  plans_exactly_verified: number;
  exact_simulation_cache_hits: number;
  remaining_epanet_budget: number;
}

interface ApiAuditEvent {
  sequence: number;
  timestamp: string;
  event_type: string;
  actor: string;
  payload: Record<string, unknown>;
}

/** Mirrors hydroswarm.api.state.ReplayResponse field-for-field, as
 * returned by POST /incidents/{id}/replay (ui-work.txt 9.8). */
interface ApiReplayResponse {
  state: ApiIncidentState;
  events: ApiAuditEvent[];
  chain_valid: boolean;
}

function eventFromApi(raw: ApiAuditEvent): AuditEvent {
  return {
    sequence: raw.sequence,
    timestamp: raw.timestamp.slice(11, 19),
    type: raw.event_type,
    actor: raw.actor,
    detail: JSON.stringify(raw.payload),
  };
}

/**
 * Request the real audit-ledger replay for this incident: its complete
 * event history plus a real hash-chain integrity check (ui-work.txt 9.8).
 * This is NOT a historical state snapshot -- `state` is the incident's
 * *current* raw state, never a point-in-time replay frame (ui-work.txt
 * 20: "Never pretend the current map is a historical snapshot"). Only
 * meaningful for a LIVE incident with a real backend.
 */
export async function replayIncident(
  incidentId: string,
  signal?: AbortSignal,
): Promise<ReplayResult> {
  const raw = await requestJson<ApiReplayResponse>(
    `/incidents/${incidentId}/replay`,
    undefined,
    signal,
  );
  return {
    state: {
      incidentId: raw.state.incident_id,
      networkId: raw.state.network_id,
      status: raw.state.status,
      sampleCount: raw.state.sample_count,
      approvalPending: raw.state.approval_pending,
      oodLevel: raw.state.ood_level,
      exactSimulationsUsed: raw.state.exact_simulations_used,
      plansExactlyVerified: raw.state.plans_exactly_verified,
      exactSimulationCacheHits: raw.state.exact_simulation_cache_hits,
      remainingEpanetBudget: raw.state.remaining_epanet_budget,
    },
    events: raw.events.map(eventFromApi),
    chainValid: raw.chain_valid,
  };
}
