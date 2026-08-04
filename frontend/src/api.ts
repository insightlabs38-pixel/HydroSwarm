import { demoIncident } from './demoFixture';
import type { AuditEvent, IncidentView } from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';
const INCIDENT_ID = import.meta.env.VITE_INCIDENT_ID as string | undefined;

/**
 * Controlled failure-injection demonstration (overnight-plan.txt Task 3.8).
 * Selected via the `?failure=<category>` query parameter so an operator or
 * judge can deterministically see the console's fail-closed behavior for
 * each governed failure category without needing a live backend in that
 * state. Every category renders as ERROR mode with the exact reason named
 * -- never a false VERIFIED/LIVE state.
 */
export const FAILURE_INJECTION_CATEGORIES = [
  'missing_checkpoint',
  'corrupt_checkpoint_hash',
  'incompatible_feature_schema',
  'corrupt_calibration',
  'unknown_topology',
  'wntr_timeout',
  'incomplete_simulator_output',
  'severe_sensor_dropout',
  'no_valid_plan',
] as const;

export type FailureInjectionCategory = (typeof FAILURE_INJECTION_CATEGORIES)[number];

const FAILURE_INJECTION_REASONS: Record<FailureInjectionCategory, string> = {
  missing_checkpoint:
    'No trained checkpoint is present at the configured path. Falling back to classical-only mode is not simulated here; this demonstration shows the fail-closed ERROR state instead of a false LIVE result.',
  corrupt_checkpoint_hash:
    "The checkpoint's SHA-256 does not match its recorded metadata. Refusing to load a checkpoint that may have been tampered with or corrupted.",
  incompatible_feature_schema:
    "The checkpoint's feature schema hash does not match this build's DEFAULT_FEATURE_SCHEMA. Refusing to run inference with mismatched feature semantics.",
  corrupt_calibration:
    'The calibration artifact failed hash or schema validation against the loaded checkpoint. Refusing to report a calibrated candidate set without valid calibration.',
  unknown_topology:
    "This network's topology hash is not in the validated set and no broader validated calibration artifact covers it. Calibration is invalid; planning is suppressed (CAUTION).",
  wntr_timeout:
    'The exact WNTR/EPANET verification simulation exceeded its timeout. No plan may be labeled VERIFIED without a completed authoritative simulation.',
  incomplete_simulator_output:
    'WNTR returned incomplete results for this scenario (missing timesteps or nodes). Refusing to compute consequences from a partial simulation.',
  severe_sensor_dropout:
    'Too few sensors are reporting valid observations to form a trustworthy candidate set. Evidence is insufficient; planning is suppressed.',
  no_valid_plan:
    'Every generated response plan was rejected by exact WNTR verification. Correct behavior here is abstention, not forcing a plan through.',
};

function injectedFailure(): FailureInjectionCategory | null {
  if (typeof window === 'undefined') return null;
  const requested = new URLSearchParams(window.location.search).get('failure');
  return (FAILURE_INJECTION_CATEGORIES as readonly string[]).includes(requested ?? '')
    ? (requested as FailureInjectionCategory)
    : null;
}

interface ApiIncidentState {
  incident_id: string;
  network_id: string;
  status: IncidentView['status'];
  ood_level: IncidentView['ood'];
  approval_pending: boolean;
  disagreement_js: number | null;
  candidates: {
    node_probabilities: Record<string, number>;
    coverage_target: number;
    calibrated: boolean;
    measured_coverage: number | null;
  } | null;
}

/**
 * Raised when the live API is reachable and returned a well-formed
 * response, but that response does not yet cover every field a complete
 * IncidentView requires (map topology, plans, sample recommendation,
 * evidence contraction, benchmarks, explanation text -- see
 * overnight-plan.txt Task 3.2, "add a complete incident-view API
 * contract", not yet implemented server-side). Callers must not paper
 * over this by silently substituting fixture content; it must surface as
 * a distinct mode, never as LIVE.
 */
export class LiveViewIncompleteError extends Error {
  constructor(public readonly missingFields: string[]) {
    super(`live API does not yet provide: ${missingFields.join(', ')}`);
    this.name = 'LiveViewIncompleteError';
  }
}

/** Raised when the API is reachable but this specific incident cannot be
 * safely rendered (e.g. a configured incident ID that does not exist). */
export class IncidentUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IncidentUnavailableError';
  }
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

/**
 * Fetch the live incident view. Throws LiveViewIncompleteError rather than
 * ever silently blending in demo-fixture content: today's API surface
 * (/incidents/{id}, /incidents/{id}/events) covers only a subset of
 * IncidentView (id, networkId, status, ood, approvalPending, disagreement,
 * candidates, candidateCoverage/calibration, audit). Map topology, plans,
 * sample recommendation, evidence contraction, benchmarks, and explanation
 * text are not yet exposed by a live endpoint (Task 3.2 tracks adding a
 * single complete `/incidents/{id}/view` contract). Until that lands, this
 * function refuses to claim a LIVE mode that isn't actually complete.
 */
export async function fetchIncident(signal?: AbortSignal): Promise<IncidentView> {
  await request<{ status: string }>('/health', signal);
  if (!INCIDENT_ID) {
    throw new IncidentUnavailableError('No active incident configured (VITE_INCIDENT_ID unset)');
  }
  let state: ApiIncidentState;
  let events: Record<string, unknown>[];
  try {
    [state, events] = await Promise.all([
      request<ApiIncidentState>(`/incidents/${INCIDENT_ID}`, signal),
      request<Record<string, unknown>[]>(`/incidents/${INCIDENT_ID}/events`, signal),
    ]);
  } catch (error) {
    throw new IncidentUnavailableError(
      `configured incident ${INCIDENT_ID} could not be loaded: ${(error as Error).message}`,
    );
  }

  // The two live calls above already succeeded and are correctly typed --
  // this validates connectivity and response shape even in this interim
  // state. What's not yet available is everything else a complete
  // IncidentView needs (overnight-plan.txt Task 3.2, "add a complete
  // incident-view API contract", not yet implemented server-side): map
  // topology, plans, sample recommendation, evidence contraction,
  // benchmarks, explanation text. Rather than silently substituting
  // fixture content for those, refuse to claim LIVE until they exist.
  void state;
  void events;
  void eventFromApi;
  throw new LiveViewIncompleteError([
    'nodes',
    'links',
    'recommendedSample',
    'evidence',
    'plans',
    'benchmarks',
    'explanation',
  ]);
}

export async function fetchIncidentWithFallback(signal?: AbortSignal): Promise<IncidentView> {
  const failure = injectedFailure();
  if (failure) {
    return {
      ...demoIncident,
      mode: 'ERROR',
      modeReason: `[Failure injection: ${failure}] ${FAILURE_INJECTION_REASONS[failure]}`,
    };
  }
  try {
    return await fetchIncident(signal);
  } catch (error) {
    if (error instanceof IncidentUnavailableError) {
      return {
        ...demoIncident,
        mode: 'ERROR',
        modeReason: error.message,
      };
    }
    // Network/health failure, or a live response that is well-formed but
    // structurally incomplete: fall back to the clearly-labeled demo
    // fixture rather than a blank screen. This mirrors the plan's product
    // requirement that HydroSwarm remain usable offline; it must never be
    // confused with LIVE, and demoIncident.mode is already DEMO_FALLBACK.
    const reason =
      error instanceof LiveViewIncompleteError
        ? `Live incident data is incomplete (${error.missingFields.join(', ')} not yet available from the API). Showing the frozen demo fixture instead.`
        : demoIncident.modeReason;
    return { ...demoIncident, modeReason: reason };
  }
}
