export type OodLevel = 'NORMAL' | 'CAUTION' | 'OUTSIDE_VALIDATED_RANGE';
export type PlanStatus = 'REJECTED' | 'RECOMMENDED' | 'VALID';

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
  audit: AuditEvent[];
  benchmarks: Benchmark[];
  explanation: string;
}
