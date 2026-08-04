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
