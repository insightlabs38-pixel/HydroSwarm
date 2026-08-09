import type { EvidenceCertificate, EvidenceCertificateStatus } from '../types';
import { request } from './client';

/** Mirrors hydroswarm.domain.schemas.EvidenceCertificate (Pydantic)
 * field-for-field, as returned by GET /incidents/{id}/evidence-certificate
 * (ui-work.txt 9.3). Kept snake_case here; certificateFromApi() below
 * does the one-time camelCase mapping. */
interface ApiEvidenceCertificate {
  status: EvidenceCertificateStatus;
  stop: boolean;
  message: string;
  posterior_entropy_bits: number;
  candidate_set_size: number;
  candidate_nodes: string[];
  candidate_region_calibrated: boolean;
  recommended_sample_node: string | null;
  expected_information_gain_bits: number | null;
  expected_candidate_reduction: number | null;
  sample_budget_remaining: number;
  already_sampled_nodes: string[];
  recommended_node_accessible: boolean | null;
}

function certificateFromApi(raw: ApiEvidenceCertificate): EvidenceCertificate {
  return {
    status: raw.status,
    stop: raw.stop,
    message: raw.message,
    posteriorEntropyBits: raw.posterior_entropy_bits,
    candidateSetSize: raw.candidate_set_size,
    candidateNodes: [...raw.candidate_nodes],
    candidateRegionCalibrated: raw.candidate_region_calibrated,
    recommendedSampleNode: raw.recommended_sample_node,
    expectedInformationGainBits: raw.expected_information_gain_bits,
    expectedCandidateReduction: raw.expected_candidate_reduction,
    sampleBudgetRemaining: raw.sample_budget_remaining,
    alreadySampledNodes: [...raw.already_sampled_nodes],
    recommendedNodeAccessible: raw.recommended_node_accessible,
  };
}

/**
 * Fetch the governed Evidence Value / Stop Certificate for this incident
 * (ui-work.txt 9.3). Only meaningful for a LIVE incident with a real
 * backend -- callers in other data modes should not call this.
 */
export async function fetchEvidenceCertificate(
  incidentId: string,
  signal?: AbortSignal,
): Promise<EvidenceCertificate> {
  const raw = await request<ApiEvidenceCertificate>(
    `/incidents/${incidentId}/evidence-certificate`,
    signal,
  );
  return certificateFromApi(raw);
}
