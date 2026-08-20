import type { IncidentView } from './types';
import { isCalibrationApplicable } from './calibrationDisplay';

/**
 * Centralized decision-path semantics (ui-work.txt §10).
 *
 * Derived from existing IncidentView fields only -- no new API.
 * Distinguishes calibration-not-applicable from calibration-invalid
 * (a novel topology that never exercised the calibration artifact
 * is NOT a corrupt calibration artifact).
 */
export type DecisionGateState =
  | 'READY'
  | 'DEGRADED'
  | 'OUTSIDE_VALIDATED_RANGE'
  | 'CALIBRATION_INVALID'
  | 'CALIBRATION_NOT_APPLICABLE'
  | 'ERROR';

export interface DecisionGateInfo {
  state: DecisionGateState;
  /** Compact path label for the header. */
  pathLabel: string;
  /** Accessible explanation for the status. */
  accessibleDetail: string;
  /** Tone for the status badge. */
  tone: 'good' | 'warn' | 'danger' | 'info';
}

export function deriveDecisionGate(incident: IncidentView): DecisionGateInfo {
  if (incident.mode === 'ERROR') {
    return {
      state: 'ERROR',
      pathLabel: 'PATH BLOCKED',
      accessibleDetail: 'Incident decision path is blocked.',
      tone: 'danger',
    };
  }

  if (incident.ood === 'OUTSIDE_VALIDATED_RANGE') {
    return {
      state: 'OUTSIDE_VALIDATED_RANGE',
      pathLabel: 'PATH DEGRADED',
      accessibleDetail: 'Incident decision path is degraded: outside validated range.',
      tone: 'warn',
    };
  }

  if (isCalibrationApplicable(incident) && !incident.calibrationValid) {
    return {
      state: 'CALIBRATION_INVALID',
      pathLabel: 'PATH DEGRADED',
      accessibleDetail: 'Incident decision path is degraded: calibration is invalid for this network.',
      tone: 'warn',
    };
  }

  if (!isCalibrationApplicable(incident)) {
    // Calibration not applicable (e.g. REFERENCE mode) -- not a
    // degradation, just a different operating mode.
    return {
      state: 'CALIBRATION_NOT_APPLICABLE',
      pathLabel: 'PATH READY',
      accessibleDetail: 'Incident decision path is ready. Calibration is not applicable in this mode.',
      tone: 'good',
    };
  }

  if (incident.ood !== 'NORMAL') {
    return {
      state: 'DEGRADED',
      pathLabel: 'PATH DEGRADED',
      accessibleDetail: 'Incident decision path is degraded.',
      tone: 'warn',
    };
  }

  return {
    state: 'READY',
    pathLabel: 'PATH READY',
    accessibleDetail: 'Incident decision path is ready.',
    tone: 'good',
  };
}
