import { beforeEach, expect, test, vi } from 'vitest';

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    json: async () => body,
  } as Response;
}

beforeEach(() => {
  vi.resetModules();
});

test('maps a real /evidence-certificate response into camelCase, preserving null as a real value', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'CONTINUE_SAMPLING',
        stop: false,
        message: 'Sample J2 to reduce entropy.',
        posterior_entropy_bits: 1.1,
        candidate_set_size: 2,
        candidate_nodes: ['J1', 'J2'],
        candidate_region_calibrated: true,
        recommended_sample_node: 'J2',
        expected_information_gain_bits: 0.4,
        expected_candidate_reduction: 1,
        sample_budget_remaining: 4,
        already_sampled_nodes: [],
        recommended_node_accessible: true,
      }),
    ),
  );
  const { fetchEvidenceCertificate } = await import('../src/api/sampling');
  const certificate = await fetchEvidenceCertificate('incident-1');
  expect(certificate).toMatchObject({
    status: 'CONTINUE_SAMPLING',
    stop: false,
    posteriorEntropyBits: 1.1,
    candidateSetSize: 2,
    candidateNodes: ['J1', 'J2'],
    candidateRegionCalibrated: true,
    recommendedSampleNode: 'J2',
    expectedInformationGainBits: 0.4,
    expectedCandidateReduction: 1,
    sampleBudgetRemaining: 4,
    alreadySampledNodes: [],
    recommendedNodeAccessible: true,
  });
});

test('a real STOP certificate with no recommended node maps recommendedSampleNode to null, never a fabricated id', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'STOP_BUDGET_EXHAUSTED',
        stop: true,
        message: 'Sample budget exhausted.',
        posterior_entropy_bits: 0.2,
        candidate_set_size: 1,
        candidate_nodes: ['J1'],
        candidate_region_calibrated: true,
        recommended_sample_node: null,
        expected_information_gain_bits: null,
        expected_candidate_reduction: null,
        sample_budget_remaining: 0,
        already_sampled_nodes: ['J2', 'J3'],
        recommended_node_accessible: null,
      }),
    ),
  );
  const { fetchEvidenceCertificate } = await import('../src/api/sampling');
  const certificate = await fetchEvidenceCertificate('incident-1');
  expect(certificate.stop).toBe(true);
  expect(certificate.recommendedSampleNode).toBeNull();
  expect(certificate.expectedInformationGainBits).toBeNull();
  expect(certificate.sampleBudgetRemaining).toBe(0);
});

test('rejects (does not silently swallow) an HTTP error from the evidence-certificate endpoint', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({}, false, 409)));
  const { fetchEvidenceCertificate } = await import('../src/api/sampling');
  await expect(fetchEvidenceCertificate('incident-1')).rejects.toThrow(/409/);
});
