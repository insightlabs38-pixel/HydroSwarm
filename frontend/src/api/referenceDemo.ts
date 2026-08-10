import type { ApiReferenceArtifact } from '../reference/types';
import { request } from './client';

/** Fetches the SUB-4 governed REFERENCE INCIDENT artifact from
 * GET /api/reference-demo. Throws (via `request`'s ApiError) on a 404 --
 * the artifact was never generated for this deployment -- so callers can
 * fail closed rather than render an empty reference experience. */
export async function fetchReferenceArtifact(signal?: AbortSignal): Promise<ApiReferenceArtifact> {
  return request<ApiReferenceArtifact>('/reference-demo', signal);
}
