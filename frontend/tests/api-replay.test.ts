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

test('maps a real /replay response into camelCase, including the exact-run-budget fields not yet on IncidentView', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse({
        state: {
          incident_id: 'incident-1',
          network_id: 'net-x',
          status: 'CLOSED',
          sample_count: 2,
          approval_pending: false,
          ood_level: 'NORMAL',
          exact_simulations_used: 3,
          plans_exactly_verified: 2,
          exact_simulation_cache_hits: 1,
          remaining_epanet_budget: 0,
        },
        events: [
          {
            sequence: 1,
            timestamp: '2026-08-03T08:14:02',
            event_type: 'INCIDENT_CREATED',
            actor: 'OPERATOR',
            payload: {},
          },
        ],
        chain_valid: true,
      }),
    ),
  );
  const { replayIncident } = await import('../src/api/replay');
  const result = await replayIncident('incident-1');

  expect(result.chainValid).toBe(true);
  expect(result.events).toEqual([
    {
      sequence: 1,
      timestamp: '08:14:02',
      type: 'INCIDENT_CREATED',
      actor: 'OPERATOR',
      detail: '{}',
    },
  ]);
  expect(result.state).toEqual({
    incidentId: 'incident-1',
    networkId: 'net-x',
    status: 'CLOSED',
    sampleCount: 2,
    approvalPending: false,
    oodLevel: 'NORMAL',
    exactSimulationsUsed: 3,
    plansExactlyVerified: 2,
    exactSimulationCacheHits: 1,
    remainingEpanetBudget: 0,
  });
});

test('a real chain_valid: false is preserved, never silently upgraded to valid', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse({
        state: {
          incident_id: 'incident-1',
          network_id: 'net-x',
          status: 'CLOSED',
          sample_count: 0,
          approval_pending: false,
          ood_level: 'NORMAL',
          exact_simulations_used: 0,
          plans_exactly_verified: 0,
          exact_simulation_cache_hits: 0,
          remaining_epanet_budget: 3,
        },
        events: [],
        chain_valid: false,
      }),
    ),
  );
  const { replayIncident } = await import('../src/api/replay');
  const result = await replayIncident('incident-1');
  expect(result.chainValid).toBe(false);
});

test('rejects (does not silently swallow) an HTTP error from the replay endpoint', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({}, false, 404)));
  const { replayIncident } = await import('../src/api/replay');
  await expect(replayIncident('incident-1')).rejects.toThrow(/404/);
});
