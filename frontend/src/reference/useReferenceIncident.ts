import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchReferenceArtifact } from '../api/referenceDemo';
import type { IncidentView } from '../types';
import { milestoneToIncidentView } from './mapMilestone';

/** How long an auto-advancing milestone stays on screen before stepping
 * forward, so the narrative is readable rather than flashing past
 * (submission.txt SS5's "understandable without narration" acceptance
 * criterion). Skipped entirely when reducedMotion is on -- SS5's
 * "reduced motion supported" acceptance criterion -- in which case
 * advancing is manual-only (the Next control), never a fake wait. */
const AUTO_ADVANCE_DELAY_MS = 3200;

export interface ReferenceController {
  incident: IncidentView | null;
  isLoading: boolean;
  isError: boolean;
  errorMessage: string | null;
  milestoneIndex: number;
  milestoneCount: number;
  milestoneLabel: string;
  narrative: string;
  isPaused: boolean;
  pauseReason: string | null;
  /** Machine-readable identifier for what the pause action does (e.g.
   * "COLLECT_REFERENCE_SAMPLE", "APPROVE_REFERENCE_PLAN"). Null when not
   * paused. Not every pause is an approval -- see pauseActionLabel for
   * what to actually put on the button. */
  pauseAction: string | null;
  /** Human-readable button label for the current pause, straight from the
   * artifact (e.g. "Collect reference sample", "Approve plan"). Null when
   * not paused. */
  pauseActionLabel: string | null;
  isPlaying: boolean;
  isAtEnd: boolean;
  next: () => void;
  previous: () => void;
  /** Advances past the current pause, whatever it is -- collecting a
   * sample, approving a plan, or any future pause type. The milestone
   * artifact (not this function) decides what each pause means; this
   * never assumes "paused" implies "awaiting approval". */
  performPauseAction: () => void;
  togglePlay: () => void;
  reset: () => void;
}

export function useReferenceIncident(reducedMotion: boolean, enabled = true): ReferenceController {
  const query = useQuery({
    queryKey: ['reference-demo-artifact'],
    queryFn: ({ signal }) => fetchReferenceArtifact(signal),
    staleTime: Number.POSITIVE_INFINITY,
    enabled,
  });
  const artifact = query.data ?? null;

  const [milestoneIndex, setMilestoneIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const milestone = artifact?.milestones[milestoneIndex] ?? null;
  const milestoneCount = artifact?.milestones.length ?? 0;
  const isAtEnd = milestoneCount > 0 && milestoneIndex >= milestoneCount - 1;
  // A milestone with auto_advance === false is a deliberate narrative
  // pause -- distinct from simply being paused via the play/pause
  // control. The reference narrative currently has two: collecting the
  // next evidence sample (sample_recommended) and approving the verified
  // plan (human_approval_boundary) -- see milestone.pause_action for
  // which one this is; never assume a pause means approval specifically.
  const isPaused = milestone !== null && !milestone.auto_advance;

  function next() {
    setMilestoneIndex((index) =>
      artifact ? Math.min(index + 1, artifact.milestones.length - 1) : index,
    );
  }
  function previous() {
    setMilestoneIndex((index) => Math.max(index - 1, 0));
  }
  function performPauseAction() {
    if (isPaused) next();
  }
  function togglePlay() {
    setIsPlaying((playing) => !playing);
  }
  function reset() {
    setMilestoneIndex(0);
    setIsPlaying(true);
  }

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (!artifact || !milestone) return;
    if (!isPlaying || reducedMotion || !milestone.auto_advance || isAtEnd) return;

    timerRef.current = setTimeout(() => {
      setMilestoneIndex((index) => Math.min(index + 1, artifact.milestones.length - 1));
    }, AUTO_ADVANCE_DELAY_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [artifact, milestone, isPlaying, reducedMotion, isAtEnd]);

  const incident = useMemo(
    () => (artifact && milestone ? milestoneToIncidentView(artifact, milestone) : null),
    [artifact, milestone],
  );

  return {
    incident,
    isLoading: query.isLoading,
    isError: query.isError,
    errorMessage: query.error instanceof Error ? query.error.message : null,
    milestoneIndex,
    milestoneCount,
    milestoneLabel: milestone?.label ?? '',
    narrative: milestone?.narrative ?? '',
    isPaused,
    pauseReason: milestone?.pause_reason ?? null,
    pauseAction: milestone?.pause_action ?? null,
    pauseActionLabel: milestone?.pause_action_label ?? null,
    isPlaying,
    isAtEnd,
    next,
    previous,
    performPauseAction,
    togglePlay,
    reset,
  };
}
