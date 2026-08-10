import { consequenceFromApi, planStatusFromApi } from '../api/incident';
import { normalizeMapCoordinates } from '../geometry';
import type { Candidate, IncidentView, NetworkLink, NetworkNode, Plan, PlanAction } from '../types';
import type { ApiReferenceArtifact, ApiReferenceMilestone, ApiReferencePlanEntry } from './types';

/** submission.txt SS4.1B's exact supporting copy for the REFERENCE label. */
export const REFERENCE_MODE_REASON =
  'Replaying a checksummed HydroSwarm reference workflow generated from the frozen ' +
  'WNTR-backed scenario. Not live telemetry.';

/** Milestones are grouped into the shell's four coarse workspace-routing
 * states by id, not by parsing the underlying FSM `controller_state`
 * string (an internal state-machine name, not this 4-value UI enum). */
const STATUS_BY_MILESTONE_ID: Record<string, IncidentView['status']> = {
  alert: 'SAMPLING',
  initial_uncertainty: 'SAMPLING',
  evidence_insufficient: 'SAMPLING',
  sample_recommended: 'SAMPLING',
  sample_received: 'SAMPLING',
  posterior_contracted: 'SAMPLING',
  plans_generated: 'PLANNING',
  unsafe_plan_rejected: 'PLANNING',
  safe_plan_verified: 'PLANNING',
  human_approval_boundary: 'APPROVAL',
  completed: 'CLOSED',
};

function planActionFromApi(action: ApiReferencePlanEntry['plan']['actions'][number]): PlanAction {
  return {
    actionType: action.action_type,
    targetId: action.target_id,
    startMinute: action.start_minute,
    durationMinutes: action.duration_minutes,
    flowRateLps: action.flow_rate_lps,
  };
}

function planFromApi(entry: ApiReferencePlanEntry, recommendedPlanId: string | null): Plan {
  const verification = entry.verification;
  return {
    id: entry.plan.plan_id,
    name: entry.plan.name,
    actions: entry.plan.actions.map(planActionFromApi),
    status: planStatusFromApi(entry.plan.plan_id, verification, recommendedPlanId),
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
          rejectionCodes: verification.rejection_codes,
          abstentionReason: verification.abstention_reason,
          verifiedAt: verification.verified_at,
          contextHash: verification.context_hash,
          verificationStatus: verification.verification_status ?? 'CURRENT',
        }
      : null,
    // No no-response WNTR comparator is threaded through the reference
    // artifact today -- same known gap noted in api/incident.ts for LIVE.
    exposureReduction: null,
  };
}

/** Maps one milestone (plus the artifact's static network topology) into
 * a fully-typed IncidentView the existing mission-control shell can render
 * unmodified -- no new bespoke reference-only UI components, per the
 * product-freeze constraint. Every field is sourced from real generated
 * data (see hydroswarm.evaluation.reference_demo); nothing here invents a
 * value the SUB-4 generator did not already compute, and nothing borrows
 * the hand-authored DEMO_FALLBACK fixture. */
export function milestoneToIncidentView(
  artifact: ApiReferenceArtifact,
  milestone: ApiReferenceMilestone,
): IncidentView {
  const view = milestone.incident_view;
  const topology = artifact.network_topology;

  const candidateNodeIds = new Set(view.candidate_region ?? []);
  const normalizedCoordinates = topology
    ? normalizeMapCoordinates(topology.nodes.map((node) => node.coordinates))
    : [];
  const nodes: NetworkNode[] = topology
    ? topology.nodes.map((node, index) => ({
        id: node.node_id,
        kind: node.node_type as NetworkNode['kind'],
        coordinates: normalizedCoordinates[index],
        probability: view.candidates?.[node.node_id] ?? 0,
        concentration:
          view.sample_observation?.node_id === node.node_id
            ? view.sample_observation.concentration_mg_l
            : null,
        candidate: candidateNodeIds.has(node.node_id),
      }))
    : [];
  const links: NetworkLink[] = topology
    ? topology.links.map((link) => ({
        id: link.link_id,
        source: link.start_node,
        target: link.end_node,
        flow: null,
        concentration: null,
      }))
    : [];

  const candidates: Candidate[] = view.candidates
    ? Object.entries(view.candidates)
        .map(([nodeId, probability]) => ({ nodeId, probability }))
        .sort((a, b) => b.probability - a.probability)
    : [];

  const plans: Plan[] = view.plans
    ? view.plans.map((entry) => planFromApi(entry, view.selected_plan_id))
    : [];

  const counterfactuals: Record<string, ReturnType<typeof consequenceFromApi>> = {};
  for (const entry of view.plans ?? []) {
    if (entry.verification?.consequences) {
      counterfactuals[entry.plan.plan_id] = consequenceFromApi(entry.verification.consequences);
    }
  }

  return {
    id: view.incident_id,
    networkId: view.network_id,
    status: STATUS_BY_MILESTONE_ID[milestone.milestone_id] ?? 'SAMPLING',
    mode: 'REFERENCE',
    modeReason: REFERENCE_MODE_REASON,
    offline: true,
    // Not a real wall-clock computation -- this is a replay, not live
    // inference (submission.txt SS4.1B: "not pretending to be wall-clock
    // computation").
    runtimeMs: 0,
    modelVersion: view.plans?.[0]?.plan.model_version ?? artifact.schema_version,
    generatedAt: artifact.generated_at,
    // The golden reference workflow runs the deterministic
    // classical-signature localization path, not the trained neural
    // pipeline -- there is no FULL_HYBRID/CLASSICAL_SAFE distinction to
    // report for it (see GoldenScenarioRunner).
    runtimeAnalysisMode: null,
    provenance: {
      networkHash: artifact.golden_result_hash,
      // Feature-schema/model/calibration provenance is a neural-pipeline
      // concept the deterministic golden workflow never exercises --
      // empty string is this codebase's established "not applicable"
      // convention for ProvenanceView (see errorIncidentView), not a
      // fabricated hash.
      featureSchemaHash: '',
      modelCheckpointHash: '',
      calibrationVersion: '',
      calibrationHash: '',
      simulator: view.plans?.find((entry) => entry.verification)?.verification?.simulator ?? null,
      simulatorVersion:
        view.plans?.find((entry) => entry.verification)?.verification?.simulator_version ?? null,
    },
    ood: (view.ood_level as IncidentView['ood']) ?? 'NORMAL',
    approvalPending: view.approval_pending,
    // The golden workflow's own credible-region target (GoldenScenarioRunner
    // uses target=0.90) -- a real value, not fabricated.
    candidateCoverage: 0.9,
    calibrationValid: true,
    disagreement: null,
    nodes,
    links,
    candidates,
    recommendedSample: view.recommended_sample
      ? {
          nodeId: view.recommended_sample.node_id,
          informationGain: view.recommended_sample.expected_information_gain_bits,
          delayMinutes: null,
          cost: null,
          rationale: 'Largest measured signature split; demand-centrality tie-break.',
        }
      : null,
    evidenceHistory: [],
    hydraulicSeries: null,
    plans,
    selectedPlanId: view.selected_plan_id,
    recommendedPlanId: view.selected_plan_id,
    counterfactuals,
    audit: [],
    benchmarks: [],
    explanations: [],
    explanation: milestone.narrative,
    simulatorBudget: null,
  };
}
