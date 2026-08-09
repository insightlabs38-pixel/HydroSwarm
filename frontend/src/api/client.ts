export const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

/** Shared fetch helper: every API module in this directory goes through
 * here so the base URL and error handling stay in exactly one place. */
export async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) throw new Error(`HydroSwarm API ${response.status}: ${response.statusText}`);
  return (await response.json()) as T;
}
