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

test('maps a real /authority response into camelCase DecisionCertificates', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse([
        {
          name: 'source_localization',
          value: { top_node: 'J1' },
          source: 'FUSED_CLASSICAL_NEURAL',
          authority: 'CALIBRATED_ADVISORY',
          applicability: 'VALIDATED',
          enabled: true,
          calibrated: true,
          suppression_reasons: [],
          provenance: { model: 'm', calibration: 'c', network: 'n', evidence: null },
        },
      ]),
    ),
  );
  const { fetchAuthorityCertificates } = await import('../src/api/authority');
  const certificates = await fetchAuthorityCertificates('incident-1');
  expect(certificates).toHaveLength(1);
  expect(certificates[0]).toMatchObject({
    name: 'source_localization',
    authority: 'CALIBRATED_ADVISORY',
    applicability: 'VALIDATED',
    enabled: true,
    calibrated: true,
    suppressionReasons: [],
    provenance: { model: 'm', calibration: 'c', network: 'n', evidence: null },
  });
});

test('a suppressed certificate keeps its real suppression reasons, never silently dropped', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse([
        {
          name: 'scout_recommendation',
          value: null,
          source: 'CLASSICAL_EIG',
          authority: 'DETERMINISTIC',
          applicability: 'INSUFFICIENT_EVIDENCE',
          enabled: false,
          calibrated: false,
          suppression_reasons: ['LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE'],
          provenance: { model: null, calibration: null, network: null, evidence: null },
        },
      ]),
    ),
  );
  const { fetchAuthorityCertificates } = await import('../src/api/authority');
  const certificates = await fetchAuthorityCertificates('incident-1');
  expect(certificates[0].suppressionReasons).toEqual([
    'LEARNED_SCOUT_SUPPRESSED:FAILED_PROMOTION_GATE',
  ]);
  expect(certificates[0].enabled).toBe(false);
});

test('rejects (does not silently swallow) an HTTP error from the authority endpoint', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({}, false, 409)));
  const { fetchAuthorityCertificates } = await import('../src/api/authority');
  await expect(fetchAuthorityCertificates('incident-1')).rejects.toThrow(/409/);
});
