import type {
  ConsequenceView,
  DecisionCertificate,
  EvidenceCertificate,
  GroundedExplanation,
  IncidentView,
  ParetoFrontierEntry,
} from './types';

// Illustrative required safety thresholds used only to compute the demo
// fixture's pressure/service-availability margins below (matches the
// rejection narrative: "Pressure below 15 m for 23 minutes"). Real margins
// come from hydroswarm.simulation.verifier at runtime; this fixture is
// hand-authored and clearly labeled DEMO_FALLBACK throughout.
const REQUIRED_MINIMUM_PRESSURE_M = 15;
const REQUIRED_SERVICE_AVAILABILITY = 0.95;

function consequence(
  values: Omit<
    ConsequenceView,
    'exposureEvaluated' | 'pressureMarginM' | 'serviceAvailabilityMargin' | 'numericallySensitive'
  >,
): ConsequenceView {
  return {
    ...values,
    exposureEvaluated: true,
    pressureMarginM: values.minimumPressureM - REQUIRED_MINIMUM_PRESSURE_M,
    serviceAvailabilityMargin: values.serviceAvailability - REQUIRED_SERVICE_AVAILABILITY,
    numericallySensitive: false,
  };
}

const consequenceA = consequence({
  populationImpacted: 210,
  contaminantMassConsumedMg: 4820,
  volumeAboveThresholdL: 1860,
  contaminatedPipeExtentM: 640,
  minimumPressureM: 9.4,
  pressureViolationMinutes: 23,
  unservedDemandL: 3100,
  serviceAvailability: 0.918,
  operationCount: 5,
  containmentTimeMinutes: 31,
});

const consequenceB = consequence({
  populationImpacted: 140,
  contaminantMassConsumedMg: 2510,
  volumeAboveThresholdL: 980,
  contaminatedPipeExtentM: 310,
  minimumPressureM: 24.1,
  pressureViolationMinutes: 0,
  unservedDemandL: 420,
  serviceAvailability: 0.987,
  operationCount: 3,
  containmentTimeMinutes: 44,
});

const consequenceC = consequence({
  populationImpacted: 190,
  contaminantMassConsumedMg: 3340,
  volumeAboveThresholdL: 1420,
  contaminatedPipeExtentM: 480,
  minimumPressureM: 27.8,
  pressureViolationMinutes: 0,
  unservedDemandL: 0,
  serviceAvailability: 1,
  operationCount: 1,
  containmentTimeMinutes: 68,
});

export const demoIncident: IncidentView = {
  id: 'HS-0041',
  networkId: 'Net3',
  status: 'APPROVAL',
  mode: 'DEMO_FALLBACK',
  modeReason:
    'Live API unavailable or no incident configured. Values shown are a frozen, simulator-derived fixture, not live telemetry.',
  offline: true,
  runtimeMs: 438,
  modelVersion: 'HydroSwarm-M 0.9.2',
  generatedAt: '2026-08-03T08:40:05Z',
  runtimeAnalysisMode: 'FULL_HYBRID',
  provenance: {
    networkHash: '3f9a1c7e8b2d4560a9f1e3c7b5d8024f6a1c9e3b7d5f0824a6c1e9b3d7f50281',
    featureSchemaHash: '8b2d4560a9f1e3c73f9a1c7eb5d8024f6a1c9e3b7d5f0824a6c1e9b3d7f5081c',
    modelCheckpointHash: 'c9e3b7d5f0824a6c1e9b3d7f50813f9a1c7e8b2d4560a9f1e3c7b5d8024f6a1',
    calibrationVersion: 'hydroswarm-calibration-v1',
    calibrationHash: '1e9b3d7f50813f9a1c7e8b2d4560a9f1e3c7b5d8024f6a1c9e3b7d5f0824a6c',
    simulator: 'wntr-epanet',
    simulatorVersion: '1.2.0',
  },
  ood: 'NORMAL',
  approvalPending: true,
  candidateCoverage: 0.9,
  calibrationValid: true,
  measuredCoverage: 0.91,
  disagreement: 0.08,
  nodes: [
    {
      id: 'R1',
      kind: 'reservoir',
      coordinates: [-80.01, 35.01],
      probability: 0,
      concentration: 0,
      candidate: false,
    },
    {
      id: 'J104',
      kind: 'junction',
      coordinates: [-80.005, 35.008],
      probability: 0.02,
      concentration: 0,
      candidate: false,
    },
    {
      id: 'J109',
      kind: 'junction',
      coordinates: [-80, 35.012],
      probability: 0.05,
      concentration: 0.1,
      candidate: false,
      sensor: {
        id: 'S4',
        health: 'HEALTHY',
        quality: 0.98,
        ageMinutes: 2,
        pressure: 28.4,
        concentration: 0.1,
      },
    },
    {
      id: 'J117',
      kind: 'junction',
      coordinates: [-79.995, 35.015],
      probability: 0.76,
      concentration: 0.78,
      candidate: true,
    },
    {
      id: 'J121',
      kind: 'junction',
      coordinates: [-79.989, 35.011],
      probability: 0.11,
      concentration: 0.46,
      candidate: true,
    },
    {
      id: 'J123',
      kind: 'junction',
      coordinates: [-79.984, 35.006],
      probability: 0.03,
      concentration: 0.31,
      candidate: false,
      sensor: {
        id: 'S7',
        health: 'DRIFT',
        quality: 0.72,
        ageMinutes: 14,
        pressure: 26.9,
        concentration: 0.31,
      },
    },
    {
      id: 'J131',
      kind: 'junction',
      coordinates: [-79.978, 35.002],
      probability: 0.03,
      concentration: 0.18,
      candidate: false,
    },
    {
      id: 'T1',
      kind: 'tank',
      coordinates: [-79.992, 35.003],
      probability: 0,
      concentration: 0,
      candidate: false,
    },
  ],
  links: [
    { id: 'P1', source: 'R1', target: 'J104', flow: 32, concentration: 0 },
    { id: 'P2', source: 'J104', target: 'J109', flow: 27, concentration: 0.1 },
    { id: 'P3', source: 'J109', target: 'J117', flow: 22, concentration: 0.72 },
    { id: 'P4', source: 'J117', target: 'J121', flow: 19, concentration: 0.61 },
    { id: 'P5', source: 'J121', target: 'J123', flow: 13, concentration: 0.42 },
    { id: 'P6', source: 'J123', target: 'J131', flow: 9, concentration: 0.2 },
    { id: 'P7', source: 'T1', target: 'J123', flow: -6, concentration: 0.04 },
  ],
  candidates: [
    { nodeId: 'J117', probability: 0.76 },
    { nodeId: 'J121', probability: 0.11 },
    { nodeId: 'J109', probability: 0.05 },
  ],
  recommendedSample: {
    nodeId: 'J123',
    informationGain: 0.37,
    delayMinutes: 14,
    cost: 1.4,
    rationale: 'Best separation between the two remaining upstream hypotheses.',
  },
  evidenceHistory: [
    {
      roundIndex: 0,
      observationCount: 6,
      validConcentrationCount: 5,
      sensorNodes: ['S4'],
      evidenceHash: 'e1a9c7d5f3b8024f6a1c9e3b7d5f0824a6c1e9b3d7f50824a6c1e9b3d7f5082',
    },
    {
      roundIndex: 1,
      observationCount: 9,
      validConcentrationCount: 8,
      sensorNodes: ['S4', 'S7'],
      evidenceHash: 'e2b8d6c4f2a7913e5b0d8f6a1c9e3b7d5f0824a6c1e9b3d7f50824a6c1e9b3d',
    },
  ],
  hydraulicSeries: [
    { time: '08:00', pressureM: 31, concentrationMgL: 0 },
    { time: '08:10', pressureM: 29, concentrationMgL: 0.12 },
    { time: '08:20', pressureM: 27, concentrationMgL: 0.78 },
    { time: '08:30', pressureM: 28, concentrationMgL: 0.44 },
    { time: '08:40', pressureM: 30, concentrationMgL: 0.16 },
  ],
  plans: [
    {
      id: 'A',
      name: 'Aggressive isolation',
      exposureReduction: 0.61,
      actions: [
        {
          actionType: 'ISOLATE_ZONE',
          targetId: 'J117',
          startMinute: 0,
          durationMinutes: 60,
          flowRateLps: null,
        },
        {
          actionType: 'CLOSE_PIPE',
          targetId: 'P4',
          startMinute: 0,
          durationMinutes: 60,
          flowRateLps: null,
        },
        {
          actionType: 'CLOSE_PIPE',
          targetId: 'P6',
          startMinute: 0,
          durationMinutes: 60,
          flowRateLps: null,
        },
        {
          actionType: 'MONITOR_NODE',
          targetId: 'J121',
          startMinute: 0,
          durationMinutes: 60,
          flowRateLps: null,
        },
        {
          actionType: 'END_PLAN',
          targetId: null,
          startMinute: 60,
          durationMinutes: 0,
          flowRateLps: null,
        },
      ],
      status: 'REJECTED',
      verification: {
        decision: 'REJECTED',
        simulator: 'wntr-epanet',
        simulatorVersion: '1.2.0',
        stateHash: 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9',
        consequences: consequenceA,
        worstCaseConsequences: null,
        evaluationProvenance: { aggregation_policy: 'posterior_weighted', hypotheses_evaluated: 1 },
        rejectionCodes: ['PRESSURE_BELOW_MINIMUM'],
        abstentionReason: null,
        verifiedAt: '2026-08-03T08:28:03Z',
        contextHash: 'ctx-a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8',
        verificationStatus: 'CURRENT',
      },
    },
    {
      id: 'B',
      name: 'Isolate + controlled flush',
      exposureReduction: 0.48,
      actions: [
        {
          actionType: 'CLOSE_PIPE',
          targetId: 'P4',
          startMinute: 0,
          durationMinutes: 60,
          flowRateLps: null,
        },
        {
          actionType: 'FLUSH_NODE',
          targetId: 'J123',
          startMinute: 5,
          durationMinutes: 20,
          flowRateLps: 4.2,
        },
        {
          actionType: 'END_PLAN',
          targetId: null,
          startMinute: 44,
          durationMinutes: 0,
          flowRateLps: null,
        },
      ],
      status: 'RECOMMENDED',
      verification: {
        decision: 'VERIFIED',
        simulator: 'wntr-epanet',
        simulatorVersion: '1.2.0',
        stateHash: 'b1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9',
        consequences: consequenceB,
        worstCaseConsequences: null,
        evaluationProvenance: { aggregation_policy: 'posterior_weighted', hypotheses_evaluated: 1 },
        rejectionCodes: [],
        abstentionReason: null,
        verifiedAt: '2026-08-03T08:28:05Z',
        contextHash: 'ctx-b1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8',
        verificationStatus: 'CURRENT',
      },
    },
    {
      id: 'C',
      name: 'Monitor + flush only',
      exposureReduction: 0.33,
      actions: [
        {
          actionType: 'MONITOR_NODE',
          targetId: 'J123',
          startMinute: 0,
          durationMinutes: 68,
          flowRateLps: null,
        },
        {
          actionType: 'FLUSH_NODE',
          targetId: 'J123',
          startMinute: 10,
          durationMinutes: 15,
          flowRateLps: 3.1,
        },
        {
          actionType: 'END_PLAN',
          targetId: null,
          startMinute: 68,
          durationMinutes: 0,
          flowRateLps: null,
        },
      ],
      status: 'VALID',
      verification: {
        decision: 'VERIFIED',
        simulator: 'wntr-epanet',
        simulatorVersion: '1.2.0',
        stateHash: 'c1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9',
        consequences: consequenceC,
        worstCaseConsequences: null,
        evaluationProvenance: { aggregation_policy: 'posterior_weighted', hypotheses_evaluated: 1 },
        rejectionCodes: [],
        abstentionReason: null,
        verifiedAt: '2026-08-03T08:28:06Z',
        contextHash: 'ctx-c1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8',
        verificationStatus: 'CURRENT',
      },
    },
  ],
  // Null, not 'B': this fixture's own headline is "Verified response
  // awaiting approval" and approvalPending is true below -- selectedPlanId
  // (which types.ts documents as "null until a human has actually
  // approved one") must stay null until that approval actually happens,
  // never pre-filled with the plan that's merely recommended.
  selectedPlanId: null,
  recommendedPlanId: 'B',
  counterfactuals: {
    A: consequenceA,
    B: consequenceB,
    C: consequenceC,
  },
  audit: [
    {
      sequence: 1,
      timestamp: '08:14:02',
      type: 'INCIDENT_DETECTED',
      actor: 'SENSOR S4',
      detail: 'Concentration crossed 0.001 mg/L.',
    },
    {
      sequence: 2,
      timestamp: '08:14:05',
      type: 'SOURCE_LOCALIZED',
      actor: 'SENTINEL',
      detail: 'Candidate region contains 11 nodes.',
    },
    {
      sequence: 3,
      timestamp: '08:14:08',
      type: 'SAMPLE_RECOMMENDED',
      actor: 'SCOUT',
      detail: 'J123; expected information gain 0.37.',
    },
    {
      sequence: 4,
      timestamp: '08:28:00',
      type: 'SAMPLE_RECEIVED',
      actor: 'OPERATOR',
      detail: 'Candidate region contracted to 3 nodes.',
    },
    {
      sequence: 5,
      timestamp: '08:28:03',
      type: 'PLAN_REJECTED',
      actor: 'WNTR VERIFIER',
      detail: 'Plan A caused 4 pressure violations.',
    },
    {
      sequence: 6,
      timestamp: '08:28:05',
      type: 'PLAN_VERIFIED',
      actor: 'WNTR VERIFIER',
      detail: 'Plan B preserved 98.7% service availability.',
    },
  ],
  benchmarks: [
    {
      metric: 'Top-1 source localization',
      value: '100%',
      comparison: '3 seeded golden runs',
      status: 'PASS',
    },
    {
      metric: 'True-source posterior',
      value: '99.41%',
      comparison: 'uniform prior to J2',
      status: 'PASS',
    },
    {
      metric: 'Candidate contraction',
      value: '4 to 1',
      comparison: '1.0 bit sample IG',
      status: 'PASS',
    },
    {
      metric: 'Unsafe / safe plan checks',
      value: '100% / 100%',
      comparison: 'exact WNTR verifier',
      status: 'PASS',
    },
    {
      metric: 'Modeled exposure reduction',
      value: '14,723 mg',
      comparison: 'versus exact no response',
      status: 'PASS',
    },
    {
      metric: 'Latency / Python allocation peak',
      value: '0.98 s / 1.17 MB',
      comparison: 'native WNTR RAM excluded',
      status: 'PASS',
    },
    {
      metric: 'HydroCore S / M / L checkpoint',
      value: 'S promoted (96.0% top-1) · M rejected · L not trained',
      comparison: 'see governed model evaluation table',
      status: 'PASS',
    },
  ],
  explanations: [
    {
      intent: 'WHY_SOURCE',
      text: 'J117 is the leading candidate because its observed concentration rise preceded downstream nodes by the travel time expected from a source at that location, and classical and neural signatures agree (8.0% disagreement).',
      facts: { top_node: 'J117', top_probability: 0.76, disagreement_js: 0.08 },
      limitations: ['Single-species simulated incident only.'],
    },
    {
      intent: 'WHY_SAMPLE',
      text: 'J123 was recommended because it best separates the two remaining upstream hypotheses (J117 vs. J121), with an expected information gain of 0.37 bits.',
      facts: { recommended_node: 'J123', expected_information_gain_bits: 0.37 },
      limitations: ['Alternatives J121 and J109 have lower expected gain, not zero.'],
    },
    {
      intent: 'WHAT_CHANGED',
      text: 'After the J123 sample, the candidate region contracted from 11 nodes to 3, and posterior mass concentrated on J117.',
      facts: { nodes_before: 11, nodes_after: 3 },
      limitations: [],
    },
    {
      intent: 'WHY_PLAN_REJECTED',
      text: 'Plan A (Aggressive isolation) was rejected because exact WNTR/EPANET simulation found 4 nodes with pressure below the 15 m minimum for a combined 23 minutes.',
      facts: { plan_id: 'A', rejection_codes: ['PRESSURE_BELOW_MINIMUM'] },
      limitations: [
        'Rejection is based on the posterior-weighted evaluation, not every possible hydraulic state.',
      ],
    },
    {
      intent: 'COMPARE_PLANS',
      text: 'Plan B reduces estimated exposure by 48% with zero pressure violations and 98.7% service availability; Plan C reduces exposure less (33%) but preserves slightly more service (100%); Plan A would reduce exposure most (61%) but is unsafe.',
      facts: { plan_ids: ['A', 'B', 'C'] },
      limitations: [],
    },
    {
      intent: 'UNCERTAINTY_REMAINS',
      text: 'Start time, duration, and relative strength estimates remain exploratory (a known governed limitation) and are not shown as calibrated quantities.',
      facts: {},
      limitations: ['Unseen-topology transfer is measured but weak for this checkpoint.'],
    },
    {
      intent: 'WHICH_SENSOR_MATTERED',
      text: 'Sensor S4 at J109 (healthy, 98% quality) triggered detection; sensor S7 at J123 (drift flagged, 72% quality) contributed lower-weight evidence.',
      facts: { sensors: ['S4', 'S7'] },
      limitations: [],
    },
  ] satisfies GroundedExplanation[],
  explanation:
    'Plan B is preferred because exact simulation found no pressure violations while reducing estimated exposure by 48% and preserving 98.7% service availability. Plan A was rejected despite higher exposure reduction because four nodes fell below the pressure threshold.',
};

/**
 * Illustrative Decision Authority / Applicability Certificates for the
 * DEMO_FALLBACK fixture (core-issues5.txt Section 13). GET
 * /incidents/{id}/authority only exists for a real LIVE incident; this
 * hand-authored set mirrors the real certificate shapes
 * (hydroswarm.inference.authority.build_decision_certificates) so the
 * Source workspace and Model & Authority workspace have something
 * genuine to render in the deterministic demo, clearly attributable to
 * DEMO_FALLBACK like the rest of this fixture -- never used for a LIVE
 * incident.
 */
export const demoAuthorityCertificates: DecisionCertificate[] = [
  {
    name: 'source_localization',
    value: { top_node: 'J117', belief: { J117: 0.76, J121: 0.11, J109: 0.05 } },
    source: 'FUSED_CLASSICAL_NEURAL',
    authority: 'CALIBRATED_ADVISORY',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: true,
    suppressionReasons: [],
    provenance: {
      model: demoIncident.provenance.modelCheckpointHash,
      calibration: demoIncident.provenance.calibrationHash,
      network: demoIncident.provenance.networkHash,
      evidence: null,
    },
  },
  {
    name: 'scout_recommendation',
    value: 'J123',
    source: 'CLASSICAL_EIG',
    authority: 'DETERMINISTIC',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: false,
    suppressionReasons: ['LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE'],
    provenance: {
      model: demoIncident.provenance.modelCheckpointHash,
      calibration: null,
      network: demoIncident.provenance.networkHash,
      evidence: null,
    },
  },
  {
    name: 'ood_state',
    value: 'NORMAL',
    source: 'DETERMINISTIC_CONTROLLER',
    authority: 'DETERMINISTIC',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: false,
    suppressionReasons: ['LEARNED_OOD_CATEGORY_SUPPRESSED:NOT_PROMOTED'],
    provenance: {
      model: demoIncident.provenance.modelCheckpointHash,
      calibration: null,
      network: demoIncident.provenance.networkHash,
      evidence: null,
    },
  },
  {
    name: 'plan_consequence:A',
    value: null,
    source: 'WNTR_EPANET',
    authority: 'SIMULATOR_VERIFIED',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: false,
    suppressionReasons: ['PRESSURE_BELOW_MINIMUM'],
    provenance: { model: null, calibration: null, network: null, evidence: null },
  },
  {
    name: 'plan_consequence:B',
    value: null,
    source: 'WNTR_EPANET',
    authority: 'SIMULATOR_VERIFIED',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: false,
    suppressionReasons: [],
    provenance: { model: null, calibration: null, network: null, evidence: null },
  },
];

/**
 * Illustrative Evidence Value / Stop Certificate for the DEMO_FALLBACK
 * fixture (core-issues5.txt Section 15). GET
 * /incidents/{id}/evidence-certificate only exists for a real LIVE
 * incident; this hand-authored value is consistent with demoIncident's
 * own recommendedSample and evidenceHistory fields -- never used for a
 * LIVE incident.
 */
export const demoEvidenceCertificate: EvidenceCertificate = {
  status: 'CONTINUE_SAMPLING',
  stop: false,
  message: 'Sampling J123 is expected to further separate the remaining source candidates.',
  posteriorEntropyBits: 1.02,
  candidateSetSize: 3,
  candidateNodes: ['J117', 'J121', 'J109'],
  candidateRegionCalibrated: true,
  recommendedSampleNode: 'J123',
  expectedInformationGainBits: 0.37,
  expectedCandidateReduction: 2,
  sampleBudgetRemaining: 3,
  alreadySampledNodes: ['J109', 'J123'],
  recommendedNodeAccessible: true,
};

const noResponseConsequence: ConsequenceView = consequence({
  populationImpacted: 260,
  contaminantMassConsumedMg: 6200,
  volumeAboveThresholdL: 2400,
  contaminatedPipeExtentM: 780,
  minimumPressureM: 30,
  pressureViolationMinutes: 0,
  unservedDemandL: 0,
  serviceAvailability: 1,
  operationCount: 0,
  containmentTimeMinutes: null,
});

/** A hydraulic-only evaluation (exposureEvaluated: false) so the frontier
 * demo can show ui-work.txt 15's required visual separation from
 * EXPOSURE_AWARE entries -- population/mass/volume/pipe-extent below are
 * Pydantic defaults, not real zero measurements. */
const hydraulicOnlyConsequence: ConsequenceView = {
  populationImpacted: 0,
  contaminantMassConsumedMg: 0,
  volumeAboveThresholdL: 0,
  contaminatedPipeExtentM: 0,
  minimumPressureM: 22,
  pressureViolationMinutes: 0,
  unservedDemandL: 0,
  serviceAvailability: 0.95,
  operationCount: 2,
  containmentTimeMinutes: null,
  exposureEvaluated: false,
  pressureMarginM: 7,
  serviceAvailabilityMargin: 0,
  numericallySensitive: true,
};

/**
 * Illustrative verified response Pareto frontier for the DEMO_FALLBACK
 * fixture (core-issues5.txt Section 14). GET /incidents/{id}/frontier
 * only exists for a real LIVE incident; consequences here reuse the same
 * consequenceA/B/C values as demoIncident.plans/counterfactuals -- never
 * used for a LIVE incident. `dominated` reflects genuine Pareto
 * trade-offs among these hand-authored points, not plan approval status
 * (a Pareto-efficient point can still be REJECTED for safety, e.g. Plan
 * A here -- verification authority always overrides frontier position).
 */
export const demoParetoFrontier: ParetoFrontierEntry[] = [
  {
    planId: 'no-response',
    label: 'No response',
    consequences: noResponseConsequence,
    mode: 'posterior_weighted',
    dominated: false,
    isNoActionComparator: true,
    group: 'EXPOSURE_AWARE',
  },
  {
    planId: 'A',
    label: 'Aggressive isolation',
    consequences: consequenceA,
    mode: 'posterior_weighted',
    dominated: false,
    isNoActionComparator: false,
    group: 'EXPOSURE_AWARE',
  },
  {
    planId: 'B',
    label: 'Isolate + controlled flush',
    consequences: consequenceB,
    mode: 'posterior_weighted',
    dominated: false,
    isNoActionComparator: false,
    group: 'EXPOSURE_AWARE',
  },
  {
    planId: 'C',
    label: 'Monitor + flush only',
    consequences: consequenceC,
    mode: 'posterior_weighted',
    dominated: true,
    isNoActionComparator: false,
    group: 'EXPOSURE_AWARE',
  },
  {
    planId: 'legacy-hydraulic',
    label: 'Reopen prior closure (hydraulic-only evaluation)',
    consequences: hydraulicOnlyConsequence,
    mode: 'posterior_weighted',
    dominated: false,
    isNoActionComparator: false,
    group: 'HYDRAULIC_ONLY',
  },
];
