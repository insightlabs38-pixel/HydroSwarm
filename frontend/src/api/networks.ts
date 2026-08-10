import type { NetworkRecord } from '../types';
import { API_BASE, ApiError, request, requestJson, requestPost } from './client';

interface ApiNetworkTopologyNode {
  node_id: string;
  node_type: string;
  elevation_m: number;
  coordinates: [number, number];
}

interface ApiNetworkTopologyLink {
  link_id: string;
  link_type: string;
  start_node: string;
  end_node: string;
}

/** Mirrors hydroswarm.api.state.NetworkRecord (Pydantic) field-for-field.
 * `metadata` is a loosely-typed dict server-side; nodes/links are only
 * ever populated by a real .inp import (see
 * hydroswarm/networks/importer.py), so both are optional here. */
interface ApiNetworkRecord {
  network_id: string;
  name: string;
  version: number;
  sha256: string;
  node_count: number;
  link_count: number;
  valid: boolean;
  validated_at: string;
  metadata: {
    nodes?: ApiNetworkTopologyNode[];
    links?: ApiNetworkTopologyLink[];
  };
  validation_errors: string[];
}

function recordFromApi(raw: ApiNetworkRecord): NetworkRecord {
  return {
    networkId: raw.network_id,
    name: raw.name,
    version: raw.version,
    sha256: raw.sha256,
    nodeCount: raw.node_count,
    linkCount: raw.link_count,
    valid: raw.valid,
    validatedAt: raw.validated_at,
    nodes: (raw.metadata.nodes ?? []).map((node) => ({
      nodeId: node.node_id,
      nodeType: node.node_type,
      elevationM: node.elevation_m,
      coordinates: node.coordinates,
    })),
    links: (raw.metadata.links ?? []).map((link) => ({
      linkId: link.link_id,
      linkType: link.link_type,
      startNode: link.start_node,
      endNode: link.end_node,
    })),
    validationErrors: [...raw.validation_errors],
  };
}

/** Fetch every locally imported network (ui-work.txt 17). */
export async function fetchNetworks(signal?: AbortSignal): Promise<NetworkRecord[]> {
  const raw = await request<ApiNetworkRecord[]>('/networks', signal);
  return raw.map(recordFromApi);
}

/**
 * Import a local .inp EPANET network file (ui-work.txt 17: "local file;
 * no cloud upload"). Uses multipart/form-data directly rather than
 * client.ts's requestJson(), since a file upload must never carry a
 * Content-Type: application/json header -- the browser sets the correct
 * multipart boundary itself when FormData is passed as the body.
 */
export async function importNetwork(file: File, signal?: AbortSignal): Promise<NetworkRecord> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE}/networks/import`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: formData,
    signal,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string' && body.detail.length > 0) detail = body.detail;
    } catch {
      // Non-JSON error body -- fall back to statusText above.
    }
    throw new ApiError(response.status, `HydroSwarm API ${response.status}: ${detail}`);
  }
  return recordFromApi((await response.json()) as ApiNetworkRecord);
}

interface ApiIncidentStateSummary {
  incident_id: string;
}

/**
 * SUB-12.1 P1 #6: the compact, API-assisted incident-creation form on the
 * "Import Your Own Network" advanced path -- deliberately minimal (one
 * initial observation, not a multi-step wizard), through the same real
 * POST /api/incidents endpoint the LIVE example uses. Never fabricates a
 * network node or observation: the caller supplies a real node id the
 * imported network actually has and a real reading for it. Also triggers
 * the incident's first real analysis before returning -- otherwise the
 * mission-control shell this hands off to has no way to get a freshly
 * created, still-DETECTED incident out of `/view`'s 409 ("requires a
 * completed hybrid analysis"): there is no separate "run analysis" control
 * anywhere else in the UI, so this compact API-assisted flow has to be the
 * one to trigger it.
 */
export async function createIncidentForNetwork(
  networkId: string,
  observation: { nodeId: string; concentrationMgL: number; pressureM: number | null },
  signal?: AbortSignal,
): Promise<string> {
  const now = new Date().toISOString();
  const state = await requestJson<ApiIncidentStateSummary>(
    '/incidents',
    {
      network_id: networkId,
      detected_at: now,
      observations: [
        {
          sensor_id: `S-${observation.nodeId}`,
          node_id: observation.nodeId,
          observed_at: now,
          received_at: now,
          concentration_mg_l: observation.concentrationMgL,
          pressure_m: observation.pressureM,
        },
      ],
    },
    signal,
  );
  await requestPost<ApiIncidentStateSummary>(`/incidents/${state.incident_id}/analyze`, signal);
  return state.incident_id;
}
