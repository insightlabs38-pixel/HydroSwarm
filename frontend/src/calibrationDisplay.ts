import type { IncidentView } from './types';

/** submission.txt SUB-12.1 P0 #2B: REFERENCE mode replays the deterministic
 * golden workflow, not a production conformal-calibrated neural inference
 * run -- `calibrationValid`/`candidateCoverage`/`measuredCoverage` must
 * never be rendered as if they came from a real calibration exercise for
 * it. These four helpers are the single place every render site (Overview,
 * DecisionInspector, ModeBanner, MissionHeader) asks "is this applicable,
 * and if not, what should it say instead" -- so the distinction can't
 * silently drift back apart across those call sites. */

export function isCalibrationApplicable(incident: IncidentView): boolean {
  return incident.calibrationApplicable !== false;
}

export function calibrationStatusText(incident: IncidentView): string {
  if (!isCalibrationApplicable(incident)) return 'Not applicable to reference replay';
  return incident.calibrationValid ? 'valid' : 'invalid';
}

export function candidateCoverageLabel(incident: IncidentView): string {
  return isCalibrationApplicable(incident) ? 'Conformal target' : 'Region target';
}

export function candidateCoverageValueText(incident: IncidentView): string {
  const percent = Math.round(incident.candidateCoverage * 100);
  return isCalibrationApplicable(incident) ? `${percent}%` : `${percent}% reference criterion`;
}

export function measuredCoverageValueText(incident: IncidentView): string {
  if (!isCalibrationApplicable(incident)) return 'Not applicable';
  return typeof incident.measuredCoverage === 'number'
    ? `${Math.round(incident.measuredCoverage * 100)}%`
    : 'not measured';
}
