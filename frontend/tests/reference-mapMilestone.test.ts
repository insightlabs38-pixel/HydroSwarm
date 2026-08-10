import { milestoneToIncidentView, REFERENCE_MODE_REASON } from '../src/reference/mapMilestone';
import type { ApiReferenceArtifact, ApiReferenceMilestone } from '../src/reference/types';

function baseMilestone(
  overrides: Partial<ApiReferenceMilestone['incident_view']> = {},
): ApiReferenceMilestone {
  return {
    index: 0,
    milestone_id: 'alert',
    label: 'Incident detected',
    controller_state: 'NETWORK_VALIDATED',
    event_sequence_start: 0,
    event_sequence_end: 2,
    auto_advance: true,
    pause_reason: null,
    pause_action: null,
    pause_action_label: null,
    highlight: 'incident_opened',
    narrative: 'An incident opens.',
    incident_view: {
      incident_id: 'incident-1',
      network_id: 'golden-network-v1',
      ood_level: 'NORMAL',
      controller_state: 'NETWORK_VALIDATED',
      candidates: null,
      candidate_region: null,
      evidence_sufficient: null,
      recommended_sample: null,
      sample_observation: null,
      plans: null,
      selected_plan_id: null,
      approved_plan_id: null,
      approval_pending: false,
      final_event_hash: null,
      ...overrides,
    },
  };
}

function baseArtifact(milestones: ApiReferenceMilestone[]): ApiReferenceArtifact {
  return {
    schema_version: 'hydroswarm-reference-incident-v1',
    reference_id: 'reference-incident-v1',
    title: 'HydroSwarm reference incident',
    description: 'test',
    generator: 'test',
    generated_at: '2026-08-10T00:00:00+00:00',
    source_commit: 'deadbeef',
    network_sha256: 'network-hash',
    golden_result_hash: 'golden-hash',
    final_event_hash: 'final-hash',
    event_count: 21,
    network_topology: {
      nodes: [
        { node_id: 'R1', node_type: 'reservoir', elevation_m: 100, coordinates: [0, 0] },
        { node_id: 'J2', node_type: 'junction', elevation_m: 95, coordinates: [1, 1] },
      ],
      links: [{ link_id: 'P1', link_type: 'pipe', start_node: 'R1', end_node: 'J2' }],
    },
    milestones,
    artifact_sha256: 'artifact-hash',
  };
}

test('maps mode, offline, and the exact REFERENCE supporting copy', () => {
  const artifact = baseArtifact([baseMilestone()]);
  const view = milestoneToIncidentView(artifact, artifact.milestones[0]);

  expect(view.mode).toBe('REFERENCE');
  expect(view.offline).toBe(true);
  expect(view.modeReason).toBe(REFERENCE_MODE_REASON);
  expect(view.runtimeMs).toBe(0);
});

test('alert milestone has no candidates, plans, or approval state', () => {
  const artifact = baseArtifact([baseMilestone()]);
  const view = milestoneToIncidentView(artifact, artifact.milestones[0]);

  expect(view.candidates).toEqual([]);
  expect(view.plans).toEqual([]);
  expect(view.selectedPlanId).toBeNull();
  expect(view.approvalPending).toBe(false);
});

test('maps real network topology into nodes/links, marking candidates from candidate_region', () => {
  const milestone = baseMilestone({
    candidates: { R1: 0.1, J2: 0.9 },
    candidate_region: ['J2'],
  });
  const artifact = baseArtifact([milestone]);
  const view = milestoneToIncidentView(artifact, milestone);

  expect(view.nodes).toHaveLength(2);
  const j2 = view.nodes.find((node) => node.id === 'J2')!;
  expect(j2.probability).toBe(0.9);
  expect(j2.candidate).toBe(true);
  const r1 = view.nodes.find((node) => node.id === 'R1')!;
  expect(r1.candidate).toBe(false);
  expect(view.links).toHaveLength(1);
});

test('sample_observation only sets concentration on the sampled node, never fabricating others', () => {
  const milestone = baseMilestone({
    sample_observation: { node_id: 'J2', concentration_mg_l: 4.2 },
  });
  const artifact = baseArtifact([milestone]);
  const view = milestoneToIncidentView(artifact, milestone);

  const j2 = view.nodes.find((node) => node.id === 'J2')!;
  const r1 = view.nodes.find((node) => node.id === 'R1')!;
  expect(j2.concentration).toBe(4.2);
  expect(r1.concentration).toBeNull();
});

test('plan status is PENDING before verification and REJECTED/VERIFIED after, matching planStatusFromApi', () => {
  const pendingMilestone = baseMilestone({
    plans: [
      {
        plan: {
          plan_id: 'p1',
          incident_id: 'incident-1',
          name: 'Unsafe plan',
          actions: [],
          created_at: '2026-08-03T00:00:00Z',
          model_version: 'golden-deterministic-v1',
          requires_operator_approval: true,
        },
        verification: null,
      },
    ],
  });
  const artifact = baseArtifact([pendingMilestone]);
  const pendingView = milestoneToIncidentView(artifact, pendingMilestone);
  expect(pendingView.plans[0].status).toBe('PENDING');
  expect(pendingView.plans[0].verification).toBeNull();

  const rejectedMilestone = baseMilestone({
    plans: [
      {
        plan: pendingMilestone.incident_view.plans![0].plan,
        verification: {
          decision: 'REJECTED',
          simulator: 'WNTRSimulator',
          simulator_version: '1.5.0',
          state_hash: 'hash',
          consequences: null,
          worst_case_consequences: null,
          evaluation_provenance: null,
          rejection_codes: ['PRESSURE_BELOW_MINIMUM'],
          abstention_reason: null,
          verified_at: '2026-08-03T00:00:00Z',
          context_hash: null,
          verification_status: 'CURRENT',
        },
      },
    ],
  });
  const rejectedView = milestoneToIncidentView(artifact, rejectedMilestone);
  expect(rejectedView.plans[0].status).toBe('REJECTED');
  expect(rejectedView.plans[0].verification?.decision).toBe('REJECTED');
  expect(rejectedView.plans[0].verification?.rejectionCodes).toEqual(['PRESSURE_BELOW_MINIMUM']);
});

test('status is derived from milestone id, not the raw FSM controller_state string', () => {
  const artifact = baseArtifact([
    { ...baseMilestone(), milestone_id: 'plans_generated' },
    { ...baseMilestone(), milestone_id: 'human_approval_boundary' },
    { ...baseMilestone(), milestone_id: 'completed' },
  ]);
  expect(milestoneToIncidentView(artifact, artifact.milestones[0]).status).toBe('PLANNING');
  expect(milestoneToIncidentView(artifact, artifact.milestones[1]).status).toBe('APPROVAL');
  expect(milestoneToIncidentView(artifact, artifact.milestones[2]).status).toBe('CLOSED');
});

test('provenance.networkHash uses the real network hash, never the golden-result hash', () => {
  const artifact = baseArtifact([baseMilestone()]);
  artifact.network_sha256 = 'real-network-hash';
  artifact.golden_result_hash = 'real-golden-result-hash';
  const view = milestoneToIncidentView(artifact, artifact.milestones[0]);

  expect(view.provenance.networkHash).toBe('real-network-hash');
  expect(view.provenance.networkHash).not.toBe('real-golden-result-hash');
});

test('REFERENCE mode marks calibration as not applicable, never a fabricated production result', () => {
  const artifact = baseArtifact([baseMilestone()]);
  const view = milestoneToIncidentView(artifact, artifact.milestones[0]);

  expect(view.calibrationApplicable).toBe(false);
});
