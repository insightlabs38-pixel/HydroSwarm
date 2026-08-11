import type { IncidentView } from '../types';
import { isCalibrationApplicable } from '../calibrationDisplay';
import type { ReferenceController } from '../reference/useReferenceIncident';

function replayPauseActionLabel(reference: ReferenceController): string {
  switch (reference.pauseAction) {
    case 'COLLECT_REFERENCE_SAMPLE':
      return 'Replay sample collection';
    case 'APPROVE_REFERENCE_PLAN':
      return 'Replay operator approval';
    default:
      return reference.pauseActionLabel ?? 'Continue replay';
  }
}

/**
 * Persistent narrow banner shown only when relevant (ui-work.txt 6): never
 * shown in ordinary LIVE/NORMAL state. REFERENCE/REPLAY/DEMO_FALLBACK/ERROR
 * take priority (they describe where the data came from); CALIBRATION
 * INVALID / OUTSIDE VALIDATED RANGE can additionally surface for an
 * otherwise-LIVE incident whose planning is suppressed.
 */
export function ModeBanner({
  incident,
  onRetry,
  reference,
  onExploreReplay,
}: {
  incident: IncidentView;
  onRetry?: () => void;
  /** Only meaningful when incident.mode === 'REFERENCE' -- the progressive
   * milestone controller driving this view (submission.txt SS5/SS6). */
  reference?: ReferenceController;
  onExploreReplay?: () => void;
}) {
  if (incident.mode === 'REFERENCE' && reference) {
    return (
      <div className="mode-banner mode-banner-info mode-banner-reference" role="status">
        <div className="mode-banner-reference-content">
          <strong>REFERENCE INCIDENT · VERIFIED REPLAY</strong>
          <span className="mode-banner-milestone">
            {reference.milestoneLabel} · {reference.milestoneIndex + 1} / {reference.milestoneCount}
          </span>
          <span className="mode-banner-progress" aria-hidden="true">
            <span
              style={{
                width: `${((reference.milestoneIndex + 1) / reference.milestoneCount) * 100}%`,
              }}
            />
          </span>
          <span className="mode-banner-reference-explanation">
            {reference.isPaused && reference.pauseReason
              ? reference.pauseReason
              : incident.modeReason}
          </span>
        </div>
        <div className="mode-banner-controls">
          {reference.isPaused ? (
            <button type="button" onClick={reference.performPauseAction}>
              {replayPauseActionLabel(reference)}
            </button>
          ) : (
            <button type="button" onClick={reference.togglePlay} aria-pressed={reference.isPlaying}>
              {reference.isPlaying ? 'Pause' : 'Play'}
            </button>
          )}
          <button
            type="button"
            onClick={reference.previous}
            disabled={reference.milestoneIndex === 0}
          >
            Back
          </button>
          <button
            type="button"
            onClick={reference.next}
            disabled={reference.isAtEnd || reference.isPaused}
          >
            Next
          </button>
          <button type="button" onClick={reference.reset}>
            Restart
          </button>
          {reference.isAtEnd && onExploreReplay && (
            <button type="button" onClick={onExploreReplay}>
              Explore full replay →
            </button>
          )}
        </div>
      </div>
    );
  }
  if (incident.mode === 'DEMO_FALLBACK') {
    return (
      <div className="mode-banner mode-banner-warn" role="status">
        <strong>ILLUSTRATIVE DEMO / DEMO_FALLBACK</strong>
        <span>{incident.modeReason}</span>
      </div>
    );
  }
  if (incident.mode === 'REPLAY') {
    return (
      <div className="mode-banner mode-banner-info" role="status">
        <strong>REPLAY</strong>
        <span>
          {incident.modeReason ?? 'Showing a selected stored trajectory, not live telemetry.'}
        </span>
      </div>
    );
  }
  if (incident.mode === 'ERROR') {
    return (
      <div className="mode-banner mode-banner-danger" role="alert">
        <strong>INCIDENT UNAVAILABLE</strong>
        <span>{incident.modeReason ?? 'The configured incident could not be loaded.'}</span>
        {onRetry && (
          <button type="button" onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    );
  }
  if (isCalibrationApplicable(incident) && !incident.calibrationValid) {
    return (
      <div className="mode-banner mode-banner-warn" role="status">
        <strong>CALIBRATION INVALID</strong>
        <span>
          The calibration artifact backing this incident&apos;s candidate set did not validate.
          Candidate-set sizing is not trustworthy.
        </span>
      </div>
    );
  }
  if (incident.ood === 'OUTSIDE_VALIDATED_RANGE') {
    return (
      <div className="mode-banner mode-banner-danger" role="alert">
        <strong>OUTSIDE VALIDATED RANGE</strong>
        <span>
          This incident is outside the model&apos;s validated operating range. Planning may be
          suppressed.
        </span>
      </div>
    );
  }
  return null;
}
