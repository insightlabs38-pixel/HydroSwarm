import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import App from '../src/App';
import type { IncidentView } from '../src/types';

// overnight-plan.txt Task 3.3 test requirement: render a fixture with
// different node and plan IDs, assert none of the old hard-coded
// identifiers (J117, J121, J123, J131, Plan A/B, P4) leak through, and
// confirm plan order can change without breaking the UI.
const alternateIncident: IncidentView = {
  id: 'ZZ-9999',
  networkId: 'AltNet',
  status: 'APPROVAL',
  mode: 'DEMO_FALLBACK',
  modeReason: 'test fixture',
  offline: true,
  runtimeMs: 100,
  modelVersion: 'test-model',
  generatedAt: '2026-08-03T08:00:00Z',
  runtimeAnalysisMode: 'FULL_HYBRID',
  provenance: {
    networkHash: 'n'.repeat(64),
    featureSchemaHash: 'f'.repeat(64),
    modelCheckpointHash: 'm'.repeat(64),
    calibrationVersion: 'hydroswarm-calibration-v1',
    calibrationHash: 'c'.repeat(64),
    simulator: 'wntr',
    simulatorVersion: '1.0',
  },
  ood: 'NORMAL',
  approvalPending: true,
  candidateCoverage: 0.9,
  calibrationValid: true,
  measuredCoverage: 0.9,
  disagreement: 0.1,
  nodes: [
    {
      id: 'X1',
      kind: 'reservoir',
      coordinates: [0, 0],
      probability: 0,
      concentration: 0,
      candidate: false,
    },
    {
      id: 'X2',
      kind: 'junction',
      coordinates: [0.01, 0.01],
      probability: 0.7,
      concentration: 0.5,
      candidate: true,
    },
    {
      id: 'X3',
      kind: 'junction',
      coordinates: [0.02, 0.02],
      probability: 0.3,
      concentration: 0.2,
      candidate: true,
    },
  ],
  links: [
    { id: 'LINK-A', source: 'X1', target: 'X2', flow: 10, concentration: 0.1 },
    { id: 'LINK-B', source: 'X2', target: 'X3', flow: 8, concentration: 0.2 },
  ],
  candidates: [
    { nodeId: 'X2', probability: 0.7 },
    { nodeId: 'X3', probability: 0.3 },
  ],
  recommendedSample: {
    nodeId: 'X3',
    informationGain: 0.5,
    delayMinutes: 5,
    cost: 1,
    rationale: 'test rationale',
  },
  evidenceHistory: [
    {
      roundIndex: 0,
      observationCount: 4,
      validConcentrationCount: 3,
      sensorNodes: ['X2'],
      evidenceHash: 'h'.repeat(64),
    },
  ],
  hydraulicSeries: null,
  plans: [
    // Deliberately reversed order vs. a "recommended-first" assumption.
    {
      id: 'ZETA',
      name: 'Zeta response',
      exposureReduction: 0.2,
      actions: [
        {
          actionType: 'MONITOR_NODE',
          targetId: 'X2',
          startMinute: 0,
          durationMinutes: 10,
          flowRateLps: null,
        },
      ],
      status: 'VALID',
      verification: null,
    },
    {
      id: 'ALPHA',
      name: 'Alpha response',
      exposureReduction: 0.6,
      actions: [
        {
          actionType: 'CLOSE_PIPE',
          targetId: 'LINK-A',
          startMinute: 0,
          durationMinutes: 30,
          flowRateLps: null,
        },
        {
          actionType: 'ISOLATE_ZONE',
          targetId: 'X2',
          startMinute: 0,
          durationMinutes: 30,
          flowRateLps: null,
        },
        {
          actionType: 'END_PLAN',
          targetId: null,
          startMinute: 30,
          durationMinutes: 0,
          flowRateLps: null,
        },
      ],
      status: 'REJECTED',
      verification: {
        decision: 'REJECTED',
        simulator: 'wntr-epanet',
        simulatorVersion: '1.2.0',
        stateHash: 'b'.repeat(64),
        consequences: null,
        worstCaseConsequences: null,
        evaluationProvenance: null,
        rejectionCodes: ['test rejection'],
        abstentionReason: null,
        verifiedAt: '2026-08-03T08:00:00Z',
        contextHash: 'ctx-alpha',
        verificationStatus: 'CURRENT',
      },
    },
    {
      id: 'OMEGA',
      name: 'Omega response',
      exposureReduction: 0.5,
      actions: [
        {
          actionType: 'CLOSE_PIPE',
          targetId: 'LINK-B',
          startMinute: 0,
          durationMinutes: 20,
          flowRateLps: null,
        },
        {
          actionType: 'END_PLAN',
          targetId: null,
          startMinute: 20,
          durationMinutes: 0,
          flowRateLps: null,
        },
      ],
      status: 'RECOMMENDED',
      verification: {
        decision: 'VERIFIED',
        simulator: 'wntr-epanet',
        simulatorVersion: '1.2.0',
        stateHash: 'c'.repeat(64),
        consequences: {
          populationImpacted: 0,
          contaminantMassConsumedMg: 0,
          volumeAboveThresholdL: 0,
          contaminatedPipeExtentM: 0,
          minimumPressureM: 20,
          pressureViolationMinutes: 0,
          unservedDemandL: 0,
          serviceAvailability: 0.99,
          operationCount: 2,
          containmentTimeMinutes: 20,
          exposureEvaluated: true,
          pressureMarginM: 5,
          serviceAvailabilityMargin: 0.04,
          numericallySensitive: false,
        },
        worstCaseConsequences: null,
        evaluationProvenance: null,
        rejectionCodes: [],
        abstentionReason: null,
        verifiedAt: '2026-08-03T08:00:05Z',
        contextHash: 'ctx-omega',
        verificationStatus: 'CURRENT',
      },
    },
  ],
  selectedPlanId: 'OMEGA',
  recommendedPlanId: 'OMEGA',
  counterfactuals: {},
  audit: [],
  benchmarks: [],
  explanation: 'test explanation text',
};

vi.mock('../src/api', () => ({
  fetchIncidentWithFallback: async () => alternateIncident,
}));

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

const OLD_HARDCODED_IDENTIFIERS = ['J117', 'J121', 'J123', 'J131', 'Plan A', 'Plan B', 'P4'];

test('no hard-coded incident identifiers leak through when the underlying data uses different IDs', async () => {
  renderApp();
  await screen.findByText('Verified response awaiting approval');

  for (const identifier of OLD_HARDCODED_IDENTIFIERS) {
    expect(screen.queryByText(new RegExp(identifier))).toBeNull();
  }
  // And the new identifiers/plan names from the alternate fixture actually
  // rendered, proving the page is genuinely data-driven.
  expect(screen.getAllByText(/Omega response/).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/X3/).length).toBeGreaterThan(0);
});
