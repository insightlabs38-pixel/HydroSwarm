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
    { id: 'X1', kind: 'reservoir', coordinates: [0, 0], probability: 0, concentration: 0, candidate: false },
    { id: 'X2', kind: 'junction', coordinates: [0.01, 0.01], probability: 0.7, concentration: 0.5, candidate: true },
    { id: 'X3', kind: 'junction', coordinates: [0.02, 0.02], probability: 0.3, concentration: 0.2, candidate: true },
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
  evidence: {
    before: [{ nodeId: 'X2', probability: 0.5 }],
    after: [{ nodeId: 'X2', probability: 0.7 }],
    uncertaintyReduction: 0.3,
    nodesRemoved: 1,
  },
  plans: [
    // Deliberately reversed order vs. a "recommended-first" assumption.
    { id: 'ZETA', name: 'Zeta response', exposureReduction: 0.2, pressureViolations: 0, serviceAvailability: 1, actions: 1, containmentMinutes: 10, status: 'VALID' },
    { id: 'ALPHA', name: 'Alpha response', exposureReduction: 0.6, pressureViolations: 2, serviceAvailability: 0.9, actions: 3, containmentMinutes: 30, status: 'REJECTED', rejectionReason: 'test rejection' },
    { id: 'OMEGA', name: 'Omega response', exposureReduction: 0.5, pressureViolations: 0, serviceAvailability: 0.99, actions: 2, containmentMinutes: 20, status: 'RECOMMENDED' },
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
