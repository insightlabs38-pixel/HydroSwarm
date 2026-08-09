import type { AuthorityLevel, ApplicabilityStatus, DecisionCertificate } from '../types';
import { request } from './client';

/** Mirrors hydroswarm.domain.schemas.DecisionCertificate (Pydantic)
 * field-for-field, as returned by GET /incidents/{id}/authority
 * (ui-work.txt 9.2, core-issues5.txt Section 13). Kept snake_case here;
 * certificateFromApi() below does the one-time camelCase mapping. */
interface ApiDecisionCertificate {
  name: string;
  value: unknown;
  source: string;
  authority: AuthorityLevel;
  applicability: ApplicabilityStatus;
  enabled: boolean;
  calibrated: boolean;
  suppression_reasons: string[];
  provenance: {
    model: string | null;
    calibration: string | null;
    network: string | null;
    evidence: string | null;
  };
}

function certificateFromApi(raw: ApiDecisionCertificate): DecisionCertificate {
  return {
    name: raw.name,
    value: raw.value,
    source: raw.source,
    authority: raw.authority,
    applicability: raw.applicability,
    enabled: raw.enabled,
    calibrated: raw.calibrated,
    suppressionReasons: [...raw.suppression_reasons],
    provenance: {
      model: raw.provenance.model,
      calibration: raw.provenance.calibration,
      network: raw.provenance.network,
      evidence: raw.provenance.evidence,
    },
  };
}

/**
 * Fetch the governed Decision Authority / Applicability Certificates for
 * this incident (ui-work.txt 9.2: "Do not infer authority from scattered
 * values"). Only meaningful for a LIVE incident with a real backend --
 * callers in other data modes should not call this.
 */
export async function fetchAuthorityCertificates(
  incidentId: string,
  signal?: AbortSignal,
): Promise<DecisionCertificate[]> {
  const raw = await request<ApiDecisionCertificate[]>(`/incidents/${incidentId}/authority`, signal);
  return raw.map(certificateFromApi);
}
