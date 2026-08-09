import { demoIncident } from '../demoFixture';
import type {
  AuditEvent,
  Candidate,
  ConsequenceView,
  ExplanationIntent,
  GroundedExplanation,
  IncidentView,
  NetworkLink,
  NetworkNode,
  Plan,
  PlanStatus,
  SensorState,
} from '../types';
import { request } from './client';

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

/** Raised when the API is reachable but this specific incident cannot be
 * safely rendered (e.g. a configured incident ID that does not exist). */
export class IncidentUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IncidentUnavailableError';
  }
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

/** Mirrors hydroswarm.api.state.IncidentView (Pydantic) field-for-field, as
 * returned by GET /incidents/{id}/view -- overnight-plan.txt Task 3.2. Kept
 * snake_case/untransformed here; viewFromApi() below does the one-time
 * mapping into the UI-shaped, camelCase IncidentView. */
interface ApiIncidentView {
  schema_version: 1;
  incident_id: string;
  network_id: string;
  runtime_mode: 'FULL_HYBRID' | 'CLASSICAL_SAFE';
  data_mode: 'LIVE';
  controller_state: 'DETECTED' | 'ANALYZING' | 'SAMPLING' | 'PLANNING' | 'APPROVAL' | 'CLOSED';
  generated_at: string;
  provenance: {
    network_hash: string;
    feature_schema_hash: string;
    model_version: string;
    model_checkpoint_hash: string;
    calibration_version: string;
    calibration_hash: string;
    simulator: string | null;
    simulator_version: string | null;
  };
  candidates: {
    node_probabilities: Record<string, number>;
    node_ids: string[];
    coverage_target: number;
    measured_coverage: number | null;
    calibrated: boolean;
  };
  disagreement_js: number | null;
  ood_level: string;
  calibration_alpha: number | null;
  nodes: {
    node_id: string;
    node_type: string;
    elevation_m: number;
    coordinates: [number, number];
    probability: number;
    concentration_mg_l: number | null;
    candidate: boolean;
  }[];
  links: { link_id: string; link_type: string; start_node: string; end_node: string }[];
  sensor_health: {
    sensor_id: string;
    node_id: string;
    health: 'HEALTHY' | 'DRIFT' | 'MISSING';
    quality: number;
    observed_at: string;
    received_at: string;
    pressure_m: number | null;
    concentration_mg_l: number | null;
  }[];
  sample_recommendation: {
    node_id: string;
    expected_information_gain: number;
    alternatives: string[];
  } | null;
  evidence_history: {
    round_index: number;
    observation_count: number;
    valid_concentration_count: number;
    sensor_nodes: string[];
    evidence_hash: string;
  }[];
  plans: {
    plan: {
      plan_id: string;
      name: string;
      actions: {
        action_type: string;
        target_id: string | null;
        start_minute: number;
        duration_minutes: number;
        flow_rate_lps: number | null;
      }[];
    };
    verification: {
      decision: 'VERIFIED' | 'REJECTED' | 'ABSTAINED' | 'PENDING_APPROVAL';
      simulator: string;
      simulator_version: string;
      state_hash: string;
      consequences: ApiConsequenceMetrics | null;
      worst_case_consequences: ApiConsequenceMetrics | null;
      evaluation_provenance: Record<string, unknown> | null;
      rejection_codes: string[];
      abstention_reason: string | null;
      verified_at: string;
      context_hash: string | null;
      verification_status: 'CURRENT' | 'STALE';
    } | null;
  }[];
  selected_plan_id: string | null;
  recommended_plan_id: string | null;
  counterfactual_consequences: Record<string, ApiConsequenceMetrics>;
  explanations: {
    intent: string;
    text: string;
    facts: Record<string, unknown>;
    limitations: string[];
  }[];
  audit_events: Record<string, unknown>[];
  runtime_metrics_ms: Record<string, number>;
}

/** Mirrors hydroswarm.domain.schemas.ConsequenceMetrics field-for-field.
 * Shared by both counterfactual_consequences and plans[].verification,
 * since the backend populates the former directly from the latter's
 * `consequences` (overnight-plan.txt Task 3.2). */
interface ApiConsequenceMetrics {
  population_impacted: number;
  contaminant_mass_consumed_mg: number;
  volume_above_threshold_l: number;
  contaminated_pipe_extent_m: number;
  minimum_pressure_m: number;
  pressure_violation_minutes: number;
  unserved_demand_l: number;
  service_availability: number;
  operation_count: number;
  containment_time_minutes: number | null;
  exposure_evaluated: boolean;
  pressure_margin_m: number | null;
  service_availability_margin: number | null;
  numerically_sensitive: boolean;
}

function planStatusFromApi(
  planId: string,
  verification: ApiIncidentView['plans'][number]['verification'],
  recommendedPlanId: string | null,
): PlanStatus {
  if (verification?.decision === 'REJECTED') return 'REJECTED';
  if (!verification || verification.decision !== 'VERIFIED') return 'PENDING';
  return planId === recommendedPlanId ? 'RECOMMENDED' : 'VALID';
}

function consequenceFromApi(raw: ApiConsequenceMetrics): ConsequenceView {
  return {
    populationImpacted: raw.population_impacted,
    contaminantMassConsumedMg: raw.contaminant_mass_consumed_mg,
    volumeAboveThresholdL: raw.volume_above_threshold_l,
    contaminatedPipeExtentM: raw.contaminated_pipe_extent_m,
    minimumPressureM: raw.minimum_pressure_m,
    pressureViolationMinutes: raw.pressure_violation_minutes,
    unservedDemandL: raw.unserved_demand_l,
    serviceAvailability: raw.service_availability,
    operationCount: raw.operation_count,
    containmentTimeMinutes: raw.containment_time_minutes,
    exposureEvaluated: raw.exposure_evaluated,
    pressureMarginM: raw.pressure_margin_m,
    serviceAvailabilityMargin: raw.service_availability_margin,
    numericallySensitive: raw.numerically_sensitive,
  };
}

/** Maps the live /view response into the UI-shaped IncidentView. Every
 * field is sourced from `raw`; nothing here substitutes demo content. */
export function viewFromApi(raw: ApiIncidentView): IncidentView {
  const sensorByNode = new Map(raw.sensor_health.map((entry) => [entry.node_id, entry]));
  const nowMs = Date.now();

  const nodes: NetworkNode[] = raw.nodes.map((node) => {
    const sensor = sensorByNode.get(node.node_id);
    const sensorState: SensorState | undefined = sensor
      ? {
          id: sensor.sensor_id,
          health: sensor.health,
          quality: sensor.quality,
          ageMinutes: Math.max(0, (nowMs - new Date(sensor.observed_at).getTime()) / 60_000),
          pressure: sensor.pressure_m ?? 0,
          concentration: sensor.concentration_mg_l ?? 0,
        }
      : undefined;
    return {
      id: node.node_id,
      // EPANET networks only ever have these three node types.
      kind: node.node_type as NetworkNode['kind'],
      coordinates: node.coordinates,
      probability: node.probability,
      concentration: node.concentration_mg_l ?? 0,
      candidate: node.candidate,
      sensor: sensorState,
    };
  });

  const links: NetworkLink[] = raw.links.map((link) => ({
    id: link.link_id,
    source: link.start_node,
    target: link.end_node,
    // Per-link flow is not yet threaded through IncidentAnalysisResult
    // (HydraulicSimulator computes it, but it isn't carried onto the
    // result today); null is honest "not measured" (ui-work.txt 8.1),
    // never a fabricated 0 that a map layer could draw as if measured.
    flow: null,
    // NetworkLinkView carries no per-link concentration field today.
    concentration: null,
  }));

  const candidates: Candidate[] = raw.candidates.node_ids
    .map((nodeId) => ({ nodeId, probability: raw.candidates.node_probabilities[nodeId] ?? 0 }))
    .sort((a, b) => b.probability - a.probability);

  const plans: Plan[] = raw.plans.map(({ plan, verification }) => ({
    id: plan.plan_id,
    name: plan.name,
    actions: plan.actions.map((action) => ({
      actionType: action.action_type,
      targetId: action.target_id,
      startMinute: action.start_minute,
      durationMinutes: action.duration_minutes,
      flowRateLps: action.flow_rate_lps,
    })),
    status: planStatusFromApi(plan.plan_id, verification, raw.recommended_plan_id),
    verification: verification
      ? {
          decision: verification.decision,
          simulator: verification.simulator,
          simulatorVersion: verification.simulator_version,
          stateHash: verification.state_hash,
          consequences: verification.consequences
            ? consequenceFromApi(verification.consequences)
            : null,
          worstCaseConsequences: verification.worst_case_consequences
            ? consequenceFromApi(verification.worst_case_consequences)
            : null,
          evaluationProvenance: verification.evaluation_provenance,
          rejectionCodes: [...verification.rejection_codes],
          abstentionReason: verification.abstention_reason,
          verifiedAt: verification.verified_at,
          contextHash: verification.context_hash,
          verificationStatus: verification.verification_status,
        }
      : null,
    // Not yet computed server-side against a no-response baseline (see
    // hydroswarm.explanation.EvidenceBundle.exposure_reduction_mg, also
    // always None today) -- null means "not measured", not "no benefit"
    // (ui-work.txt 8.2: render "-- / Not evaluated").
    exposureReduction: null,
  }));

  const sample = raw.sample_recommendation;
  // core-issues.txt: "no sample recommended" (sampling budget exhausted,
  // or active sampling found nothing further worth measuring) is a real,
  // legitimate state -- represented as null, not a zero-filled
  // placeholder object that a consumer could mistake for an actual
  // recommendation of zero information gain at an empty node id.
  const recommendedSample = sample
    ? {
        nodeId: sample.node_id,
        informationGain: sample.expected_information_gain,
        // SampleRecommendationView carries no delay/cost fields today --
        // null, never a fabricated 0 (ui-work.txt 8.3).
        delayMinutes: null,
        cost: null,
        rationale: `Expected information gain ${sample.expected_information_gain.toFixed(2)} bits; alternatives: ${sample.alternatives.join(', ') || 'none'}.`,
      }
    : null;

  const explanations: GroundedExplanation[] = raw.explanations.map((item) => ({
    intent: item.intent as ExplanationIntent,
    text: item.text,
    facts: item.facts,
    limitations: [...item.limitations],
  }));
  const whySource = explanations.find((item) => item.intent === 'WHY_SOURCE');

  return {
    id: raw.incident_id,
    networkId: raw.network_id,
    status: raw.controller_state as IncidentView['status'],
    mode: 'LIVE',
    offline: true,
    runtimeMs: Object.values(raw.runtime_metrics_ms).reduce((total, value) => total + value, 0),
    modelVersion: raw.provenance.model_version,
    generatedAt: raw.generated_at,
    runtimeAnalysisMode: raw.runtime_mode,
    provenance: {
      networkHash: raw.provenance.network_hash,
      featureSchemaHash: raw.provenance.feature_schema_hash,
      modelCheckpointHash: raw.provenance.model_checkpoint_hash,
      calibrationVersion: raw.provenance.calibration_version,
      calibrationHash: raw.provenance.calibration_hash,
      simulator: raw.provenance.simulator,
      simulatorVersion: raw.provenance.simulator_version,
    },
    ood: raw.ood_level as IncidentView['ood'],
    approvalPending: raw.controller_state === 'APPROVAL',
    candidateCoverage: raw.candidates.coverage_target,
    calibrationValid: raw.candidates.calibrated,
    measuredCoverage: raw.candidates.measured_coverage ?? undefined,
    disagreement: raw.disagreement_js ?? 0,
    nodes,
    links,
    candidates,
    recommendedSample,
    evidenceHistory: raw.evidence_history.map((entry) => ({
      roundIndex: entry.round_index,
      observationCount: entry.observation_count,
      validConcentrationCount: entry.valid_concentration_count,
      sensorNodes: [...entry.sensor_nodes],
      evidenceHash: entry.evidence_hash,
    })),
    // No backend endpoint exposes a live per-incident hydraulic time
    // series yet -- null here, never a fabricated series (ui-work.txt 8.5).
    hydraulicSeries: null,
    plans,
    selectedPlanId: raw.selected_plan_id,
    recommendedPlanId: raw.recommended_plan_id,
    counterfactuals: Object.fromEntries(
      Object.entries(raw.counterfactual_consequences).map(([planId, value]) => [
        planId,
        consequenceFromApi(value),
      ]),
    ),
    audit: raw.audit_events.map(eventFromApi),
    // Model-level benchmark summaries are a distinct, not-yet-live data
    // source (see ModelGovernanceTable, which does not read from
    // IncidentView at all) -- genuinely out of scope for Task 3.2's field
    // list. An empty array is the honest "not available live" state.
    benchmarks: [],
    explanations,
    explanation: whySource?.text ?? '',
  };
}

/**
 * Fetch the live, complete incident view (overnight-plan.txt Task 3.2).
 */
export async function fetchIncident(signal?: AbortSignal): Promise<IncidentView> {
  await request<{ status: string }>('/health', signal);
  if (!INCIDENT_ID) {
    throw new IncidentUnavailableError('No active incident configured (VITE_INCIDENT_ID unset)');
  }
  let raw: ApiIncidentView;
  try {
    raw = await request<ApiIncidentView>(`/incidents/${INCIDENT_ID}/view`, signal);
  } catch (error) {
    throw new IncidentUnavailableError(
      `configured incident ${INCIDENT_ID} could not be loaded: ${(error as Error).message}`,
    );
  }
  return viewFromApi(raw);
}

/** A genuinely empty, safe IncidentView for the ERROR mode -- core-issues.txt:
 * "Ensure ERROR mode cannot render demo plans, candidates, or operational
 * recommendations." Spreading `demoIncident` here (as this used to do) and
 * merely overriding `mode` left every one of demoIncident's fabricated
 * plans/candidates/sample-recommendation/benchmarks intact underneath the
 * error banner -- an operator who missed the banner could act on fake
 * content. Every substantive field here is empty/null/zero-length, never a
 * fabricated value, so there is nothing but the error reason itself to
 * render. */
function errorIncidentView(reason: string): IncidentView {
  return {
    id: '',
    networkId: '',
    status: 'SAMPLING',
    mode: 'ERROR',
    modeReason: reason,
    offline: true,
    runtimeMs: 0,
    modelVersion: '',
    generatedAt: null,
    runtimeAnalysisMode: null,
    provenance: {
      networkHash: '',
      featureSchemaHash: '',
      modelCheckpointHash: '',
      calibrationVersion: '',
      calibrationHash: '',
      simulator: null,
      simulatorVersion: null,
    },
    ood: 'NORMAL',
    approvalPending: false,
    candidateCoverage: 0,
    calibrationValid: false,
    disagreement: 0,
    nodes: [],
    links: [],
    candidates: [],
    recommendedSample: null,
    evidenceHistory: [],
    hydraulicSeries: null,
    plans: [],
    selectedPlanId: null,
    recommendedPlanId: null,
    counterfactuals: {},
    audit: [],
    benchmarks: [],
    explanations: [],
    explanation: '',
  };
}

export async function fetchIncidentWithFallback(signal?: AbortSignal): Promise<IncidentView> {
  const failure = injectedFailure();
  if (failure) {
    return errorIncidentView(
      `[Failure injection: ${failure}] ${FAILURE_INJECTION_REASONS[failure]}`,
    );
  }
  try {
    return await fetchIncident(signal);
  } catch (error) {
    if (error instanceof IncidentUnavailableError) {
      return errorIncidentView(error.message);
    }
    // Network/health failure, or a live response that failed to parse into
    // a valid IncidentView: fall back to the clearly-labeled demo fixture
    // rather than a blank screen. This mirrors the plan's product
    // requirement that HydroSwarm remain usable offline; it must never be
    // confused with LIVE, and demoIncident.mode is already DEMO_FALLBACK.
    return { ...demoIncident };
  }
}
