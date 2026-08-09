import type { ApplicabilityStatus } from '../../types';
import { StatusBadge } from '../StatusBadge';

/** core-issues5.txt Section 13: why a result's authority is what it is,
 * distinct from the authority level itself. */
const APPLICABILITY_TONE: Record<ApplicabilityStatus, 'good' | 'warn' | 'danger' | 'info'> = {
  VALIDATED: 'good',
  UNVALIDATED: 'warn',
  OOD: 'danger',
  CALIBRATION_UNAVAILABLE: 'warn',
  STALE: 'warn',
  SIMULATOR_SENSITIVE: 'warn',
  INSUFFICIENT_EVIDENCE: 'warn',
  DISABLED_BY_GOVERNANCE: 'danger',
  FAILED_PROMOTION_GATE: 'danger',
};

export function ApplicabilityBadge({ applicability }: { applicability: ApplicabilityStatus }) {
  return (
    <StatusBadge tone={APPLICABILITY_TONE[applicability]} label={`Applicability: ${applicability}`}>
      {applicability.replaceAll('_', ' ')}
    </StatusBadge>
  );
}
