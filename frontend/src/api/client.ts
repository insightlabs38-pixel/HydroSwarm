export const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

/** Carries the real HTTP status so callers can distinguish e.g. a 409
 * fail-closed rejection (ui-work.txt 9.7: stale-context approval must
 * fail closed) from any other error, without parsing .message. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** FastAPI's HTTPException serializes as {"detail": "..."}; surfacing
 * the real detail (not just the HTTP status text) matters here because
 * several of this backend's governed error messages -- e.g. the /approve
 * endpoint's stale-verification 409 -- are themselves the exact operator
 * copy ui-work.txt specifies, not just a debugging aid. */
async function detailFromResponse(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string' && body.detail.length > 0) return body.detail;
  } catch {
    // Response body wasn't JSON (or had no usable detail) -- fall back
    // to the generic HTTP status text below.
  }
  return response.statusText;
}

/** Shared fetch helper: every API module in this directory goes through
 * here so the base URL and error handling stay in exactly one place. */
export async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `HydroSwarm API ${response.status}: ${await detailFromResponse(response)}`,
    );
  }
  return (await response.json()) as T;
}

/** Same contract as request(), for a POST with no request body at all
 * (e.g. triggering analysis, requesting a sample recommendation, exact
 * plan verification -- endpoints whose handler takes only path
 * parameters). Distinct from requestJson() below, which sends a JSON
 * body; sending an unexpected body to a body-less endpoint is needless
 * even if FastAPI would silently ignore it. */
export async function requestPost<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `HydroSwarm API ${response.status}: ${await detailFromResponse(response)}`,
    );
  }
  return (await response.json()) as T;
}

/** Same contract as request(), for a POST with a JSON body (e.g. exact
 * verification, approval). */
export async function requestJson<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `HydroSwarm API ${response.status}: ${await detailFromResponse(response)}`,
    );
  }
  return (await response.json()) as T;
}
