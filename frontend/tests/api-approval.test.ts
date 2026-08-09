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

test('POSTs the exact ui-work.txt 9.7 body shape and maps the real receipt into camelCase', async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    jsonResponse({
      incident_id: 'incident-1',
      plan_id: 'plan-1',
      approved: true,
      operator_id: 'operator-42',
      approved_at: '2026-08-03T09:00:00Z',
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  const { approvePlan } = await import('../src/api/approval');
  const receipt = await approvePlan('incident-1', 'plan-1', 'operator-42');

  expect(receipt).toEqual({
    incidentId: 'incident-1',
    planId: 'plan-1',
    approved: true,
    operatorId: 'operator-42',
    approvedAt: '2026-08-03T09:00:00Z',
  });
  const [url, init] = fetchMock.mock.calls[0];
  expect(String(url)).toContain('/incidents/incident-1/plans/plan-1/approve');
  expect(init.method).toBe('POST');
  expect(JSON.parse(init.body)).toEqual({ approved: true, operator_id: 'operator-42' });
});

test('surfaces the real backend detail message on a stale-verification 409, not just the HTTP status', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      jsonResponse(
        {
          detail:
            'verification is stale: incident evidence has changed since this plan was verified; re-verify before approval',
        },
        false,
        409,
      ),
    ),
  );
  const { approvePlan } = await import('../src/api/approval');
  await expect(approvePlan('incident-1', 'plan-1', 'operator-42')).rejects.toThrow(
    /verification is stale/,
  );
});

test('a 409 rejection carries the real HTTP status on the thrown ApiError, not just a message string', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ detail: 'only a VERIFIED plan can be approved' }, false, 409),
      ),
  );
  const { approvePlan } = await import('../src/api/approval');
  const { ApiError } = await import('../src/api/client');
  try {
    await approvePlan('incident-1', 'plan-1', 'operator-42');
    throw new Error('expected approvePlan to reject');
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    expect((error as InstanceType<typeof ApiError>).status).toBe(409);
  }
});
