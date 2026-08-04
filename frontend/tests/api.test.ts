import { beforeEach, describe, expect, test, vi } from 'vitest';

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
  window.history.pushState(null, '', '/');
});

describe('fetchIncident (no VITE_INCIDENT_ID configured, matching this test env)', () => {
  test('throws IncidentUnavailableError when the API is reachable but no incident is configured', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })),
    );
    const { fetchIncident, IncidentUnavailableError } = await import('../src/api');
    await expect(fetchIncident()).rejects.toBeInstanceOf(IncidentUnavailableError);
  });
});

describe('fetchIncidentWithFallback', () => {
  test('falls back to DEMO_FALLBACK when the API is entirely unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
    expect(incident.modeReason).toBeTruthy();
  });

  test('returns ERROR mode (not DEMO_FALLBACK) when the API is reachable but the incident cannot be resolved', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })),
    );
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('ERROR');
    expect(incident.modeReason).toMatch(/No active incident configured/);
  });

  test('never labels any fallback result as LIVE', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).not.toBe('LIVE');
  });
});

describe('fetchIncident with an incident configured (VITE_INCIDENT_ID stubbed)', () => {
  test('throws LiveViewIncompleteError rather than silently merging demo fixture content', async () => {
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        if (String(url).includes('/events')) return Promise.resolve(jsonResponse([]));
        return Promise.resolve(
          jsonResponse({
            incident_id: 'test-incident-id',
            network_id: 'net-x',
            status: 'SAMPLING',
            ood_level: 'NORMAL',
            approval_pending: false,
            disagreement_js: null,
            candidates: null,
          }),
        );
      }),
    );
    const { fetchIncident, LiveViewIncompleteError } = await import('../src/api');
    await expect(fetchIncident()).rejects.toBeInstanceOf(LiveViewIncompleteError);
  });

  test('fetchIncidentWithFallback explains exactly which fields are missing, not a generic message', async () => {
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        if (String(url).includes('/events')) return Promise.resolve(jsonResponse([]));
        return Promise.resolve(
          jsonResponse({
            incident_id: 'test-incident-id',
            network_id: 'net-x',
            status: 'SAMPLING',
            ood_level: 'NORMAL',
            approval_pending: false,
            disagreement_js: null,
            candidates: null,
          }),
        );
      }),
    );
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
    expect(incident.modeReason).toMatch(/nodes/);
    expect(incident.modeReason).toMatch(/plans/);
  });
});

describe('failure injection (overnight-plan.txt Task 3.8)', () => {
  test.each([
    'missing_checkpoint',
    'corrupt_checkpoint_hash',
    'incompatible_feature_schema',
    'corrupt_calibration',
    'unknown_topology',
    'wntr_timeout',
    'incomplete_simulator_output',
    'severe_sensor_dropout',
    'no_valid_plan',
  ] as const)('%s renders ERROR mode with an explicit, distinct reason -- never LIVE', async (category) => {
    window.history.pushState(null, '', `/?failure=${category}`);
    // Even an otherwise-healthy API must not override an injected failure.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('ERROR');
    expect(incident.mode).not.toBe('LIVE');
    expect(incident.modeReason).toContain(category);
  });

  test('an unrecognized failure category is ignored, not silently accepted', async () => {
    window.history.pushState(null, '', '/?failure=not_a_real_category');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
  });

  test('every failure category has a distinct reason string', async () => {
    const { FAILURE_INJECTION_CATEGORIES } = await import('../src/api');
    window.history.pushState(null, '', '/');
    const reasons = new Set<string>();
    for (const category of FAILURE_INJECTION_CATEGORIES) {
      window.history.pushState(null, '', `/?failure=${category}`);
      vi.resetModules();
      vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
      const { fetchIncidentWithFallback } = await import('../src/api');
      const incident = await fetchIncidentWithFallback();
      reasons.add(incident.modeReason ?? '');
    }
    expect(reasons.size).toBe(FAILURE_INJECTION_CATEGORIES.length);
  });
});
