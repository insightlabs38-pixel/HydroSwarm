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

function apiEntry(overrides: Record<string, unknown> = {}) {
  return {
    plan_id: 'plan-a',
    label: 'Plan A',
    consequences: {
      population_impacted: 10,
      contaminant_mass_consumed_mg: 5,
      volume_above_threshold_l: 2,
      contaminated_pipe_extent_m: 100,
      minimum_pressure_m: 20,
      pressure_violation_minutes: 0,
      unserved_demand_l: 0,
      service_availability: 0.99,
      operation_count: 1,
      containment_time_minutes: 12,
      exposure_evaluated: true,
      pressure_margin_m: 5,
      service_availability_margin: 0.04,
      numerically_sensitive: false,
    },
    mode: 'posterior_weighted',
    dominated: false,
    is_no_action_comparator: false,
    group: 'EXPOSURE_AWARE',
    ...overrides,
  };
}

test('maps a real /frontier response into camelCase, preserving the EXPOSURE_AWARE/HYDRAULIC_ONLY group', async () => {
  const fetchMock = vi.fn().mockResolvedValue(jsonResponse([apiEntry()]));
  vi.stubGlobal('fetch', fetchMock);
  const { fetchParetoFrontier } = await import('../src/api/planning');
  const entries = await fetchParetoFrontier('incident-1');
  expect(entries).toHaveLength(1);
  expect(entries[0]).toMatchObject({
    planId: 'plan-a',
    label: 'Plan A',
    mode: 'posterior_weighted',
    dominated: false,
    isNoActionComparator: false,
    group: 'EXPOSURE_AWARE',
  });
  expect(entries[0].consequences.exposureEvaluated).toBe(true);
});

test('requests the worst_case frontier mode via the query parameter, never silently defaulting', async () => {
  const fetchMock = vi.fn().mockResolvedValue(jsonResponse([apiEntry({ mode: 'worst_case' })]));
  vi.stubGlobal('fetch', fetchMock);
  const { fetchParetoFrontier } = await import('../src/api/planning');
  await fetchParetoFrontier('incident-1', 'worst_case');
  const requestedUrl = String(fetchMock.mock.calls[0][0]);
  expect(requestedUrl).toContain('mode=worst_case');
});

test('a HYDRAULIC_ONLY entry keeps group distinct, never merged with EXPOSURE_AWARE', async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    jsonResponse([
      apiEntry({
        plan_id: 'legacy',
        group: 'HYDRAULIC_ONLY',
        consequences: {
          ...apiEntry().consequences,
          exposure_evaluated: false,
          population_impacted: 0,
        },
      }),
    ]),
  );
  vi.stubGlobal('fetch', fetchMock);
  const { fetchParetoFrontier } = await import('../src/api/planning');
  const entries = await fetchParetoFrontier('incident-1');
  expect(entries[0].group).toBe('HYDRAULIC_ONLY');
  expect(entries[0].consequences.exposureEvaluated).toBe(false);
});

test('rejects (does not silently swallow) an HTTP error from the frontier endpoint', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({}, false, 409)));
  const { fetchParetoFrontier } = await import('../src/api/planning');
  await expect(fetchParetoFrontier('incident-1')).rejects.toThrow(/409/);
});
