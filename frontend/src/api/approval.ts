import type { ApprovalReceipt } from '../types';
import { requestJson } from './client';

/** Mirrors hydroswarm.api.state.ApprovalReceipt (Pydantic)
 * field-for-field, as returned by
 * POST /incidents/{id}/plans/{plan_id}/approve (ui-work.txt 9.7). */
interface ApiApprovalReceipt {
  incident_id: string;
  plan_id: string;
  approved: true;
  operator_id: string;
  approved_at: string;
}

/**
 * Record a human operator's decision to approve a VERIFIED, CURRENT
 * plan. The backend fails closed with 409 if the plan is not VERIFIED,
 * or if verification has gone STALE since it was last checked
 * (ui-work.txt 9.7) -- callers must not retry automatically or infer
 * success; surface the real ApiError.message (see client.ts) and
 * re-fetch authoritative state.
 */
export async function approvePlan(
  incidentId: string,
  planId: string,
  operatorId: string,
  signal?: AbortSignal,
): Promise<ApprovalReceipt> {
  const raw = await requestJson<ApiApprovalReceipt>(
    `/incidents/${incidentId}/plans/${planId}/approve`,
    { approved: true, operator_id: operatorId },
    signal,
  );
  return {
    incidentId: raw.incident_id,
    planId: raw.plan_id,
    approved: raw.approved,
    operatorId: raw.operator_id,
    approvedAt: raw.approved_at,
  };
}
