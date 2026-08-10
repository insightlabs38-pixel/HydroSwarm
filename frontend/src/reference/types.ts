/** Raw shape of GET /api/reference-demo (SUB-4's
 * artifacts/reference-demo/reference-incident-v1.json), as the backend
 * actually serves it -- snake_case, mirroring
 * hydroswarm.evaluation.reference_demo.build_reference_incident_artifact.
 * mapMilestone.ts does the one-time camelCase mapping into IncidentView,
 * the same pattern api/incident.ts's viewFromApi already established for
 * the live endpoint. */

export interface ApiReferenceNetworkNode {
  node_id: string;
  node_type: string;
  elevation_m: number;
  coordinates: [number, number];
}

export interface ApiReferenceNetworkLink {
  link_id: string;
  link_type: string;
  start_node: string;
  end_node: string;
}

export interface ApiReferenceNetworkTopology {
  nodes: ApiReferenceNetworkNode[];
  links: ApiReferenceNetworkLink[];
}

export interface ApiReferencePlanAction {
  action_type: string;
  target_id: string | null;
  start_minute: number;
  duration_minutes: number;
  flow_rate_lps: number | null;
}

export interface ApiReferencePlan {
  plan_id: string;
  incident_id: string;
  name: string;
  actions: ApiReferencePlanAction[];
  created_at: string;
  model_version: string;
  requires_operator_approval: boolean;
}

export interface ApiReferenceConsequences {
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

export interface ApiReferenceVerification {
  decision: 'VERIFIED' | 'REJECTED' | 'ABSTAINED' | 'PENDING_APPROVAL';
  simulator: string;
  simulator_version: string;
  state_hash: string;
  consequences: ApiReferenceConsequences | null;
  worst_case_consequences: ApiReferenceConsequences | null;
  evaluation_provenance: Record<string, unknown> | null;
  rejection_codes: string[];
  abstention_reason: string | null;
  verified_at: string;
  context_hash: string | null;
  verification_status: 'CURRENT' | 'STALE' | null;
}

export interface ApiReferencePlanEntry {
  plan: ApiReferencePlan;
  verification: ApiReferenceVerification | null;
}

export interface ApiReferenceIncidentView {
  incident_id: string;
  network_id: string;
  ood_level: string;
  controller_state: string;
  candidates: Record<string, number> | null;
  candidate_region: string[] | null;
  evidence_sufficient: boolean | null;
  recommended_sample: { node_id: string; expected_information_gain_bits: number } | null;
  sample_observation: { node_id: string; concentration_mg_l: number } | null;
  plans: ApiReferencePlanEntry[] | null;
  selected_plan_id: string | null;
  approved_plan_id: string | null;
  approval_pending: boolean;
  final_event_hash: string | null;
}

export interface ApiReferenceMilestone {
  index: number;
  milestone_id: string;
  label: string;
  controller_state: string;
  event_sequence_start: number;
  event_sequence_end: number;
  auto_advance: boolean;
  pause_reason: string | null;
  incident_view: ApiReferenceIncidentView;
  highlight: string;
  narrative: string;
}

export interface ApiReferenceArtifact {
  schema_version: string;
  reference_id: string;
  title: string;
  description: string;
  generator: string;
  generated_at: string;
  source_commit: string;
  /** The real frozen network file's own SHA-256 -- distinct from
   * golden_result_hash below (a hash of the entire golden result payload,
   * not the network specifically). Use this for provenance.networkHash;
   * never golden_result_hash (submission.txt SUB-12.1 P0 #2A). */
  network_sha256: string;
  golden_result_hash: string;
  final_event_hash: string;
  event_count: number;
  network_topology: ApiReferenceNetworkTopology | null;
  milestones: ApiReferenceMilestone[];
  artifact_sha256: string;
}
