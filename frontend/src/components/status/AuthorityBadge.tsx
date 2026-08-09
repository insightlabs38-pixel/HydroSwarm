import type { AuthorityLevel } from '../../types';
import { StatusBadge } from '../StatusBadge';

/** ui-work.txt 7: "Authority labels -- Always use text/icon plus color."
 * Ordered from least to most operationally binding
 * (UNAVAILABLE -> SUPPRESSED -> ADVISORY -> CALIBRATED ADVISORY ->
 * DETERMINISTIC -> SIMULATOR VERIFIED -> HUMAN APPROVED); this badge
 * never encodes that ordering by color alone. */
const AUTHORITY_TONE: Record<AuthorityLevel, 'good' | 'warn' | 'danger' | 'info'> = {
  UNAVAILABLE: 'danger',
  SUPPRESSED: 'warn',
  ADVISORY: 'info',
  CALIBRATED_ADVISORY: 'info',
  DETERMINISTIC: 'good',
  SIMULATOR_VERIFIED: 'good',
  HUMAN_APPROVED: 'good',
};

const AUTHORITY_LABEL: Record<AuthorityLevel, string> = {
  UNAVAILABLE: 'UNAVAILABLE',
  SUPPRESSED: 'SUPPRESSED',
  ADVISORY: 'ADVISORY',
  CALIBRATED_ADVISORY: 'CALIBRATED ADVISORY',
  DETERMINISTIC: 'DETERMINISTIC',
  SIMULATOR_VERIFIED: 'SIMULATOR VERIFIED',
  HUMAN_APPROVED: 'HUMAN APPROVED',
};

export function AuthorityBadge({ authority }: { authority: AuthorityLevel }) {
  return (
    <StatusBadge
      tone={AUTHORITY_TONE[authority]}
      label={`Authority: ${AUTHORITY_LABEL[authority]}`}
    >
      {AUTHORITY_LABEL[authority]}
    </StatusBadge>
  );
}
