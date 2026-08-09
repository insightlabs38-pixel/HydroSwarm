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

function apiNetwork(overrides: Record<string, unknown> = {}) {
  return {
    network_id: 'net-1',
    name: 'Net3',
    version: 1,
    sha256: 'a'.repeat(64),
    node_count: 9,
    link_count: 7,
    valid: true,
    validated_at: '2026-08-03T08:00:00Z',
    metadata: {
      nodes: [{ node_id: 'J1', node_type: 'junction', elevation_m: 10, coordinates: [1, 2] }],
      links: [{ link_id: 'P1', link_type: 'pipe', start_node: 'J1', end_node: 'J2' }],
    },
    validation_errors: [],
    ...overrides,
  };
}

test('maps a real /networks list into camelCase, including topology metadata', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([apiNetwork()])));
  const { fetchNetworks } = await import('../src/api/networks');
  const networks = await fetchNetworks();
  expect(networks).toHaveLength(1);
  expect(networks[0]).toMatchObject({
    networkId: 'net-1',
    name: 'Net3',
    nodeCount: 9,
    linkCount: 7,
    valid: true,
  });
  expect(networks[0].nodes).toEqual([
    { nodeId: 'J1', nodeType: 'junction', elevationM: 10, coordinates: [1, 2] },
  ]);
});

test('a network whose metadata has no nodes/links maps to real empty arrays, never fabricated topology', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([apiNetwork({ metadata: {} })])));
  const { fetchNetworks } = await import('../src/api/networks');
  const networks = await fetchNetworks();
  expect(networks[0].nodes).toEqual([]);
  expect(networks[0].links).toEqual([]);
});

test('an invalid network keeps its real validation_errors, never silently hidden', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        jsonResponse([apiNetwork({ valid: false, validation_errors: ['disconnected component'] })]),
      ),
  );
  const { fetchNetworks } = await import('../src/api/networks');
  const networks = await fetchNetworks();
  expect(networks[0].valid).toBe(false);
  expect(networks[0].validationErrors).toEqual(['disconnected component']);
});

test('sends a real multipart/form-data import request, not JSON, and surfaces a real 422 import error', async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(jsonResponse({ detail: 'unsupported EPANET section' }, false, 422));
  vi.stubGlobal('fetch', fetchMock);
  const { importNetwork } = await import('../src/api/networks');
  const file = new File(['[JUNCTIONS]\n'], 'net.inp', { type: 'text/plain' });
  await expect(importNetwork(file)).rejects.toThrow(/unsupported EPANET section/);

  const [url, init] = fetchMock.mock.calls[0];
  expect(String(url)).toContain('/networks/import');
  expect(init.body).toBeInstanceOf(FormData);
  expect(init.headers?.['Content-Type']).toBeUndefined();
});
