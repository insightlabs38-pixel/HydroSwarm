export type OodLevel = 'NORMAL' | 'CAUTION' | 'OUTSIDE_VALIDATED_RANGE';
/** PENDING: generated but not yet run through WNTR/EPANET verification. */
export type PlanStatus = 'PENDING' | 'REJECTED' | 'RECOMMENDED' | 'VALID';

/**
 * Explicit runtime data provenance (overnight-plan.txt Task 3.1).
 *
 * LIVE: every field was derived from the live API for this exact incident.
 * REPLAY: every field was derived from a selected stored trajectory, not
 *   the live API and not the demo fixture.
 * DEMO_FALLBACK: the live API is unavailable (or no incident is
 *   configured); every field is the frozen, simulator-derived demo
 *   fixture. Must be visibly labeled throughout the page.
 * ERROR: the API was reachable but returned a state this console cannot
 *   safely render (e.g. a configured incident ID that does not exist).
 *   Shows a recoverable error screen instead of any incident data.
 *
 * No mode may mix sources: it is invalid to label demo-fixture-derived
 * content as LIVE, and no API error may silently produce a page that
 * appears LIVE.
 */
export type RuntimeMode = 'LIVE' | 'REPLAY' | 'DEMO_FALLBACK' | 'ERROR';

export interface NetworkNode {
  id: string;
  kind: 'junction' | 'reservoir' | 'tank';
  coordinates: [number, number];
  probability: number;
  concentration: number;
  candidate: boolean;
  sensor?: SensorState;
}

export interface NetworkLink {
  id: string;
  source: string;
  target: string;
  flow: number;
  concentration: number;
  action?: 'CLOSE' | 'FLUSH';
}

export interface SensorState {
  id: string;
  health: 'HEALTHY' | 'DRIFT' | 'MISSING';
  quality: number;
  ageMinutes: number;
  pressure: number;
  concentration: number;
}

export interface Candidate {
  nodeId: string;
  probability: number;
}

export interface Plan {
  id: string;
  name: string;
  exposureReduction: number;
  pressureViolations: number;
  serviceAvailability: number;
  actions: number;
  containmentMinutes: number;
  status: PlanStatus;
  rejectionReason?: string;
}

export interface AuditEvent {
  sequence: number;
  timestamp: string;
  type: string;
  actor: string;
  detail: string;
}

export interface Benchmark {
  metric: string;
  value: string;
  comparison: string;
  status: 'PASS' | 'WATCH' | 'NOT RUN';
}

/** Mirrors the backend's ConsequenceMetrics (Pydantic) field-for-field,
 * camelCased -- the exact simulated outcome of running one specific plan
 * (overnight-plan.txt Task 3.2's "counterfactual consequences"). */
export interface ConsequenceView {
  populationImpacted: number;
  contaminantMassConsumedMg: number;
  volumeAboveThresholdL: number;
  contaminatedPipeExtentM: number;
  minimumPressureM: number;
  pressureViolationMinutes: number;
  unservedDemandL: number;
  serviceAvailability: number;
  operationCount: number;
  containmentTimeMinutes: number | null;
}

/** Mirrors the backend's ProvenanceView -- every hash/version an operator
 * needs to trust an IncidentView response (overnight-plan.txt Task 3.2). */
export interface Provenance {
  networkHash: string;
  featureSchemaHash: string;
  modelCheckpointHash: string;
  calibrationVersion: string;
  calibrationHash: string;
  simulator: string | null;
  simulatorVersion: string | null;
}

export interface IncidentView {
  id: string;
  networkId: string;
  status: 'SAMPLING' | 'PLANNING' | 'APPROVAL' | 'CLOSED';
  mode: RuntimeMode;
  /** Human-readable reason for the current mode, shown in the mode banner
   * (e.g. why the live API was not used). Required whenever mode is not
   * LIVE, since a DEMO_FALLBACK/ERROR page must explain itself. */
  modeReason?: string;
  offline: boolean;
  runtimeMs: number;
  modelVersion: string;
  provenance: Provenance;
  ood: OodLevel;
  approvalPending: boolean;
  /** Conformal target used to size the calibrated candidate set (e.g. 0.9
   * for a 90% target). This is NOT measured per-incident coverage -- see
   * calibrationValid/measuredCoverage for that. Task 3.5. */
  candidateCoverage: number;
  /** Whether the calibration artifact backing this incident's candidate
   * set validated (matching checkpoint/feature-schema hashes) for the
   * current network/topology. When false, candidateCoverage is a stale
   * target, not a trustworthy one. */
  calibrationValid: boolean;
  /** Held-out marginal coverage actually measured for the calibration
   * artifact in use, if known -- distinct from the per-incident
   * candidateCoverage target. */
  measuredCoverage?: number;
  disagreement: number;
  nodes: NetworkNode[];
  links: NetworkLink[];
  candidates: Candidate[];
  recommendedSample: {
    nodeId: string;
    informationGain: number;
    delayMinutes: number;
    cost: number;
    rationale: string;
  };
  evidence: {
    before: Candidate[];
    after: Candidate[];
    uncertaintyReduction: number;
    nodesRemoved: number;
  };
  plans: Plan[];
  /** Operator-approved plan id, sourced from the audit ledger's
   * PLAN_APPROVED event -- null until a human has actually approved one. */
  selectedPlanId: string | null;
  /** The top strategist-ranked proposal, before any approval decision. May
   * differ from selectedPlanId, and may itself end up REJECTED on
   * verification -- see Plan.status, not this field, for that outcome. */
  recommendedPlanId: string | null;
  /** Simulated outcome of every plan that underwent WNTR/EPANET
   * verification, keyed by plan id (overnight-plan.txt Task 3.2). */
  counterfactuals: Record<string, ConsequenceView>;
  audit: AuditEvent[];
  benchmarks: Benchmark[];
  explanation: string;
}
