import { act, renderHook, waitFor } from '@testing-library/react';
import { approvePlan } from '../src/api/approval';
import {
  analyzeLiveExampleIncident,
  createLiveExampleIncident,
  fetchLiveExampleInputs,
  generateLiveExamplePlans,
  importLiveExampleNetwork,
  recommendLiveExampleSample,
  submitLiveExampleSample,
  verifyLiveExamplePlan,
} from '../src/api/liveExampleFlow';
import { fetchEvidenceCertificate } from '../src/api/sampling';
import { selectIncident } from '../src/incidentSelection';
import { useLiveExampleFlow } from '../src/liveExample/useLiveExampleFlow';
import type { EvidenceCertificate } from '../src/types';

vi.mock('../src/api/liveExampleFlow', () => ({
  fetchLiveExampleInputs: vi.fn(),
  importLiveExampleNetwork: vi.fn(),
  createLiveExampleIncident: vi.fn(),
  submitLiveExampleSample: vi.fn(),
  analyzeLiveExampleIncident: vi.fn(),
  recommendLiveExampleSample: vi.fn(),
  generateLiveExamplePlans: vi.fn(),
  verifyLiveExamplePlan: vi.fn(),
}));
vi.mock('../src/api/sampling', () => ({
  fetchEvidenceCertificate: vi.fn(),
}));
vi.mock('../src/api/approval', () => ({
  approvePlan: vi.fn(),
}));
vi.mock('../src/incidentSelection', () => ({
  selectIncident: vi.fn(),
}));

const FAKE_INPUTS = {
  executionMode: 'LIVE' as const,
  inputSource: 'FROZEN_REFERENCE_SCENARIO' as const,
  cacheStatus: 'MISS' as const,
  computedAt: '2026-08-10T00:00:00Z',
  inputSha256: 'input-hash',
  networkSha256: 'network-hash',
  scenarioSha256: 'scenario-hash',
  networkFilename: 'live_example_network.inp',
  networkInpText: '[TITLE]\n',
  trueSource: 'J6',
  candidateNodes: ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
  initialObservation: { sensorId: 'S-J1', nodeId: 'J1', concentrationMgL: 0, pressureM: 37.36 },
  candidateSignaturesMgL: { J1: 0, J6: 1217.93, J7: 1091.82, J8: 983.56 },
  sampleTimeSeconds: 3600,
  contaminationThresholdMgL: 0.001,
};

function certificate(overrides: Partial<EvidenceCertificate> = {}): EvidenceCertificate {
  return {
    status: 'CONTINUE_SAMPLING',
    stop: false,
    message: 'CONTINUE SAMPLING\nCandidate region: 7 node(s)\nSample budget remaining: 5',
    posteriorEntropyBits: 2.9,
    candidateSetSize: 7,
    candidateNodes: ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8'],
    candidateRegionCalibrated: true,
    recommendedSampleNode: 'J8',
    expectedInformationGainBits: 1.0,
    expectedCandidateReduction: 2,
    sampleBudgetRemaining: 5,
    alreadySampledNodes: ['J1'],
    recommendedNodeAccessible: true,
    ...overrides,
  };
}

function setUpHappyPath() {
  vi.mocked(fetchLiveExampleInputs).mockResolvedValue(FAKE_INPUTS);
  vi.mocked(importLiveExampleNetwork).mockResolvedValue({
    networkId: 'loop-grid-v1',
    name: 'live_example_network',
    version: 1,
    sha256: 'hash',
    nodeCount: 9,
    linkCount: 11,
    valid: true,
    validatedAt: '2026-08-10T00:00:00Z',
    nodes: [],
    links: [],
    validationErrors: [],
  });
  vi.mocked(createLiveExampleIncident).mockResolvedValue('incident-live-1');
  vi.mocked(analyzeLiveExampleIncident).mockResolvedValue(undefined);
  vi.mocked(fetchEvidenceCertificate).mockResolvedValue(certificate());
  vi.mocked(recommendLiveExampleSample).mockResolvedValue({
    nodeId: 'J8',
    expectedInformationGain: 1.0,
    alternatives: ['J6', 'J7'],
  });
  vi.mocked(submitLiveExampleSample).mockResolvedValue(undefined);
  vi.mocked(generateLiveExamplePlans).mockResolvedValue([
    { planId: 'plan-unsafe', name: 'Close sole reservoir feeder' },
    { planId: 'plan-safe', name: 'Flush downstream J8' },
  ]);
  vi.mocked(verifyLiveExamplePlan).mockImplementation(async (_incidentId, planId) => ({
    decision: planId === 'plan-safe' ? 'VERIFIED' : 'REJECTED',
  }));
  vi.mocked(approvePlan).mockResolvedValue({
    incidentId: 'incident-live-1',
    planId: 'plan-safe',
    approved: true,
    operatorId: 'judge-live-example',
    approvedAt: '2026-08-10T00:05:00Z',
  });
}

beforeEach(() => {
  vi.clearAllMocks();
});

test('CONTINUE_SAMPLING: drives the full flow: import -> create -> analyze -> pause -> collect -> plans -> verify -> pause -> approve -> complete', async () => {
  setUpHappyPath();
  const { result } = renderHook(() => useLiveExampleFlow(true));

  await waitFor(() => expect(result.current.stage).toBe('awaiting_sample_collection'));
  expect(importLiveExampleNetwork).toHaveBeenCalledWith(FAKE_INPUTS);
  expect(createLiveExampleIncident).toHaveBeenCalledWith('loop-grid-v1', FAKE_INPUTS);
  expect(fetchEvidenceCertificate).toHaveBeenCalledWith('incident-live-1');
  expect(result.current.incidentId).toBe('incident-live-1');
  expect(result.current.recommendedNode).toBe('J8');
  expect(result.current.expectedInformationGainBits).toBe(1.0);

  act(() => {
    result.current.collectSample();
  });

  await waitFor(() => expect(result.current.stage).toBe('awaiting_approval'));
  expect(submitLiveExampleSample).toHaveBeenCalledWith('incident-live-1', 'J8', 983.56);
  expect(generateLiveExamplePlans).toHaveBeenCalledWith('incident-live-1');
  expect(verifyLiveExamplePlan).toHaveBeenCalledWith('incident-live-1', 'plan-unsafe');
  expect(verifyLiveExamplePlan).toHaveBeenCalledWith('incident-live-1', 'plan-safe');
  expect(result.current.verifiedPlan).toEqual({ planId: 'plan-safe', name: 'Flush downstream J8' });

  act(() => {
    result.current.approve();
  });

  await waitFor(() => expect(result.current.stage).toBe('complete'));
  expect(approvePlan).toHaveBeenCalledWith('incident-live-1', 'plan-safe', 'judge-live-example');
  expect(selectIncident).toHaveBeenCalledWith('incident-live-1');
});

test('EVIDENCE_SUFFICIENT: skips the sample-collection pause and goes straight to real plan generation/verification', async () => {
  setUpHappyPath();
  vi.mocked(fetchEvidenceCertificate).mockResolvedValue(
    certificate({
      status: 'EVIDENCE_SUFFICIENT',
      stop: true,
      message: 'EVIDENCE SUFFICIENT\nPlanning gate satisfied',
      recommendedSampleNode: null,
    }),
  );
  const { result } = renderHook(() => useLiveExampleFlow(true));

  await waitFor(() => expect(result.current.stage).toBe('awaiting_approval'));
  expect(recommendLiveExampleSample).not.toHaveBeenCalled();
  expect(submitLiveExampleSample).not.toHaveBeenCalled();
  expect(generateLiveExamplePlans).toHaveBeenCalledWith('incident-live-1');
  expect(result.current.verifiedPlan).toEqual({ planId: 'plan-safe', name: 'Flush downstream J8' });
});

test.each([
  ['STOP_NO_USEFUL_CANDIDATE', 'STOP: NO USEFUL SAMPLE CANDIDATE REMAINS\nPlanning suppressed'],
  ['STOP_BUDGET_EXHAUSTED', 'STOP: SAMPLE BUDGET EXHAUSTED\nPlanning suppressed'],
  ['STOP_ABSTAIN', 'STOP: ABSTAINING\nPlanning suppressed'],
] as const)(
  '%s: a governed stop is a real terminal state, not an error, and never fabricates a sample or plan',
  async (status, message) => {
    setUpHappyPath();
    vi.mocked(fetchEvidenceCertificate).mockResolvedValue(
      certificate({ status, stop: true, message, recommendedSampleNode: null }),
    );
    const { result } = renderHook(() => useLiveExampleFlow(true));

    await waitFor(() => expect(result.current.stage).toBe('governed_stop'));
    expect(result.current.evidenceCertificate?.status).toBe(status);
    expect(result.current.evidenceCertificate?.message).toBe(message);
    expect(result.current.errorMessage).toBeNull();
    expect(recommendLiveExampleSample).not.toHaveBeenCalled();
    expect(submitLiveExampleSample).not.toHaveBeenCalled();
    expect(generateLiveExamplePlans).not.toHaveBeenCalled();
  },
);

test('does nothing when disabled -- never starts a real flow the app did not ask for', () => {
  setUpHappyPath();
  const { result } = renderHook(() => useLiveExampleFlow(false));

  expect(result.current.stage).toBe('idle');
  expect(fetchLiveExampleInputs).not.toHaveBeenCalled();
});

test('a failed step reaches the error stage with a real error message, not an infinite pending state', async () => {
  vi.mocked(fetchLiveExampleInputs).mockRejectedValue(new Error('network unreachable'));
  const { result } = renderHook(() => useLiveExampleFlow(true));

  await waitFor(() => expect(result.current.stage).toBe('error'));
  expect(result.current.errorMessage).toBe('network unreachable');
});

test('a genuine evidence-certificate fetch failure is a real error, not a swallowed governed stop', async () => {
  setUpHappyPath();
  vi.mocked(fetchEvidenceCertificate).mockRejectedValue(new Error('HydroSwarm API 500: internal'));
  const { result } = renderHook(() => useLiveExampleFlow(true));

  await waitFor(() => expect(result.current.stage).toBe('error'));
  expect(result.current.errorMessage).toBe('HydroSwarm API 500: internal');
});

test('no plan VERIFIED is a real stop, not a forced approval', async () => {
  setUpHappyPath();
  vi.mocked(verifyLiveExamplePlan).mockResolvedValue({ decision: 'REJECTED' });
  const { result } = renderHook(() => useLiveExampleFlow(true));

  await waitFor(() => expect(result.current.stage).toBe('awaiting_sample_collection'));
  act(() => {
    result.current.collectSample();
  });

  await waitFor(() => expect(result.current.stage).toBe('error'));
  expect(result.current.errorMessage).toMatch(/no generated plan was VERIFIED/);
  expect(approvePlan).not.toHaveBeenCalled();
});

test('restart() clears state and re-runs the flow from scratch', async () => {
  setUpHappyPath();
  const { result } = renderHook(() => useLiveExampleFlow(true));
  await waitFor(() => expect(result.current.stage).toBe('awaiting_sample_collection'));

  act(() => {
    result.current.restart();
  });
  // restart() synchronously clears the prior run's incident id and
  // immediately kicks off a fresh run (stage may already have moved past
  // 'idle' by the time this synchronous assertion runs -- that
  // transience itself is expected, not asserted here).
  expect(result.current.incidentId).toBeNull();

  await waitFor(() => expect(result.current.stage).toBe('awaiting_sample_collection'));
  expect(createLiveExampleIncident).toHaveBeenCalledTimes(2);
});
