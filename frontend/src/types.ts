export type OodLevel = 'NORMAL' | 'CAUTION' | 'OUTSIDE_VALIDATED_RANGE';
/** PENDING: generated but not yet run through WNTR/EPANET verification. */
export type PlanStatus = 'PENDING' | 'REJECTED' | 'RECOMMENDED' | 'VALID';

/**
 * Mirrors hydroswarm.domain.schemas.AuthorityLevel exactly (core-issues5.txt
 * Section 13). Ordered from least to most operationally binding -- a UI
 * must never let a lower authority result visually outrank a higher one
 * (ui-work.txt 7's "Authority labels").
 */
export type AuthorityLevel =
  | 'UNAVAILABLE'
  | 'SUPPRESSED'
  | 'ADVISORY'
  | 'CALIBRATED_ADVISORY'
  | 'DETERMINISTIC'
  | 'SIMULATOR_VERIFIED'
  | 'HUMAN_APPROVED';

/** Mirrors hydroswarm.domain.schemas.ApplicabilityStatus exactly -- why a
 * result's authority is what it is, not just the resulting level. */
export type ApplicabilityStatus =
  | 'VALIDATED'
  | 'UNVALIDATED'
  | 'OOD'
  | 'CALIBRATION_UNAVAILABLE'
  | 'STALE'
  | 'SIMULATOR_SENSITIVE'
  | 'INSUFFICIENT_EVIDENCE'
  | 'DISABLED_BY_GOVERNANCE'
  | 'FAILED_PROMOTION_GATE';

export interface DecisionProvenance {
  model: string | null;
  calibration: string | null;
  network: string | null;
  evidence: string | null;
}

/** Mirrors hydroswarm.explanation.grounded.ExplanationIntent exactly. */
export type ExplanationIntent =
  | 'WHY_SOURCE'
  | 'WHY_SAMPLE'
  | 'WHAT_CHANGED'
  | 'WHY_PLAN_REJECTED'
  | 'COMPARE_PLANS'
  | 'UNCERTAINTY_REMAINS'
  | 'WHICH_SENSOR_MATTERED';

/** Mirrors hydroswarm.api.state.ExplanationPayload field-for-field -- a
 * real grounded explanation, not a generative chat response (ui-work.txt
 * 9.5). The backend computes every intent up front as part of /view, so
 * the frontend never needs a separate round trip per question. */
export interface GroundedExplanation {
  intent: ExplanationIntent;
  text: string;
  facts: Record<string, unknown>;
  limitations: string[];
}

/** Mirrors hydroswarm.domain.schemas.DecisionCertificate field-for-field.
 * `name` is a stable backend identifier, not a display label -- known
 * values today are "source_localization", "scout_recommendation",
 * "ood_state", and "plan_consequence:{plan_id}" (one per verified plan);
 * treat any other name as a certificate this build doesn't know how to
 * render specially, not an error. `value`'s shape depends on `name`. */
export interface DecisionCertificate {
  name: string;
  value: unknown;
  source: string;
  authority: AuthorityLevel;
  applicability: ApplicabilityStatus;
  enabled: boolean;
  calibrated: boolean;
  suppressionReasons: string[];
  provenance: DecisionProvenance;
}

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
  /** Null when this link's directed flow has not been made available to
   * the console. Currently always null when mode is LIVE:
   * HydraulicSimulator computes per-link flow, but it is not yet threaded
   * onto IncidentAnalysisResult / NetworkLinkView (a known, additive
   * backend gap -- see api.ts). Never a fabricated 0 (ui-work.txt 8.1). */
  flow: number | null;
  /** Null in LIVE mode: NetworkLinkView carries no per-link concentration
   * field today (only node-level concentration_mg_l exists). Never a
   * fabricated 0. */
  concentration: number | null;
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

/** Mirrors hydroswarm.domain.schemas.OperationalAction (Pydantic)
 * field-for-field, camelCased -- ui-work.txt 9.1 "plan action typing". */
export interface PlanAction {
  actionType: string;
  targetId: string | null;
  startMinute: number;
  durationMinutes: number;
  flowRateLps: number | null;
}

export type PlanDecision = 'VERIFIED' | 'REJECTED' | 'ABSTAINED' | 'PENDING_APPROVAL';
export type VerificationStatus = 'CURRENT' | 'STALE';

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
  /** False whenever the contamination-exposure fields above are Pydantic
   * defaults rather than measured from a real chemical-transport
   * simulation (e.g. the legacy hydraulic-only evaluation path). Callers
   * must never present exposure fields as measured when this is false. */
  exposureEvaluated: boolean;
  /** Signed margin to the configured safety threshold (measured -
   * required; positive means the plan passed with room to spare). Null
   * only for a verification predating this field. */
  pressureMarginM: number | null;
  serviceAvailabilityMargin: number | null;
  /** True when either margin above falls inside the conservative
   * sensitivity epsilon band -- a real VERIFIED/REJECTED decision this
   * close to the threshold could plausibly flip under a different (still
   * correct) EPANET build. Must never be hidden behind a plain VERIFIED
   * badge (ui-work.txt 9.1 "numerical sensitivity"). */
  numericallySensitive: boolean;
}

/** Mirrors hydroswarm.domain.schemas.PlanVerification field-for-field,
 * camelCased -- ui-work.txt 9.1 "full plan verification". Null when this
 * plan has not yet undergone WNTR/EPANET verification (PENDING). */
export interface PlanVerificationView {
  decision: PlanDecision;
  simulator: string;
  simulatorVersion: string;
  stateHash: string;
  consequences: ConsequenceView | null;
  worstCaseConsequences: ConsequenceView | null;
  evaluationProvenance: Record<string, unknown> | null;
  rejectionCodes: string[];
  abstentionReason: string | null;
  verifiedAt: string;
  /** One canonical hash over every behavior-critical piece of context this
   * verification was computed against. Null only for verifications
   * predating this field. */
  contextHash: string | null;
  /** CURRENT until a behavior-critical context component changes, at
   * which point this plan's prior verification becomes STALE and is no
   * longer approvable. ui-work.txt: "Stale verification unmistakable." */
  verificationStatus: VerificationStatus;
}

export interface Plan {
  id: string;
  name: string;
  actions: PlanAction[];
  status: PlanStatus;
  verification: PlanVerificationView | null;
  /** Modeled exposure reduction versus a no-response baseline. Null until
   * a backend endpoint computes a genuine no-response WNTR comparator (a
   * known gap -- see api.ts viewFromApi comment). Never a fabricated 0
   * (ui-work.txt 8.2: render "-- / Not evaluated" when absent). */
  exposureReduction: number | null;
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

/** Mirrors one EvidenceHistoryEntryView -- a real observed-evidence round,
 * not a fabricated before/after candidate-probability comparison
 * (ui-work.txt 8.4: "If only counts/hashes are available, show those"). */
export interface EvidenceHistoryEntry {
  roundIndex: number;
  observationCount: number;
  validConcentrationCount: number;
  sensorNodes: string[];
  evidenceHash: string;
}

/** A real recorded hydraulic time series. Only ever populated for
 * DEMO_FALLBACK/REPLAY content -- no backend endpoint exposes a live
 * per-incident hydraulic time series yet, so LIVE/ERROR must always carry
 * `null` here rather than a fabricated series (ui-work.txt 8.5). */
export interface HydraulicSeriesPoint {
  time: string;
  pressureM: number;
  concentrationMgL: number;
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
  /** ISO timestamp this view was generated, or null when unknown (ERROR
   * mode). Mirrors hydroswarm.api.state.IncidentView.generated_at. */
  generatedAt: string | null;
  /** Whether the hybrid neural pipeline ran (FULL_HYBRID) or the
   * deterministic classical-safe fallback ran instead (CLASSICAL_SAFE).
   * Null when unknown (non-LIVE modes carry no analysis-mode signal of
   * their own). Distinct from `mode` (LIVE/REPLAY/DEMO_FALLBACK/ERROR),
   * which is data provenance, not analysis-pipeline identity. */
  runtimeAnalysisMode: 'FULL_HYBRID' | 'CLASSICAL_SAFE' | null;
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
  /** Null is a real, legitimate state (sampling budget exhausted, or
   * active sampling found no further useful sample) -- represented as
   * null rather than a zero-filled placeholder object so no consumer can
   * mistake "nothing recommended" for "a recommendation of zero
   * information gain at an empty node id" (core-issues.txt). */
  recommendedSample: {
    nodeId: string;
    informationGain: number;
    /** Null in LIVE mode: SampleRecommendationView carries no delay/cost
     * fields today. Never a fabricated 0 (ui-work.txt 8.3). */
    delayMinutes: number | null;
    cost: number | null;
    rationale: string;
  } | null;
  /** Real observed-evidence rounds, in order. Empty when no sampling has
   * occurred yet -- never a fabricated before/after probability delta. */
  evidenceHistory: EvidenceHistoryEntry[];
  /** See HydraulicSeriesPoint: non-null only for DEMO_FALLBACK/REPLAY. */
  hydraulicSeries: HydraulicSeriesPoint[] | null;
  plans: Plan[];
  /** Operator-approved plan id, sourced from the audit ledger's
   * PLAN_APPROVED event -- null until a human has actually approved one. */
  selectedPlanId: string | null;
  /** The top strategist-ranked proposal, before any approval decision. May
   * differ from selectedPlanId, and may itself end up REJECTED on
   * verification -- see Plan.status, not this field, for that outcome. */
  recommendedPlanId: string | null;
  /** Simulated outcome of every plan that underwent WNTR/EPANET
   * verification, keyed by plan id (overnight-plan.txt Task 3.2). Equal to
   * each plan's own verification.consequences -- kept as a separate field
   * because that is the shape the backend /view response actually returns. */
  counterfactuals: Record<string, ConsequenceView>;
  audit: AuditEvent[];
  benchmarks: Benchmark[];
  /** All grounded explanations the backend computed for this incident,
   * one per ExplanationIntent (ui-work.txt 9.5). */
  explanations: GroundedExplanation[];
  /** WHY_SOURCE's text, kept as a convenience for the existing Overview
   * explanation panel -- equivalent to
   * explanations.find(e => e.intent === 'WHY_SOURCE')?.text. */
  explanation: string;
}
