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
  isPlaying: boolean;
  isAtEnd: boolean;
  next: () => void;
  previous: () => void;
  approve: () => void;
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
  // pause (currently only the human-approval boundary) -- distinct from
  // simply being paused via the play/pause control.
  const isPaused = milestone !== null && !milestone.auto_advance;

  function next() {
    setMilestoneIndex((index) =>
      artifact ? Math.min(index + 1, artifact.milestones.length - 1) : index,
    );
  }
  function previous() {
    setMilestoneIndex((index) => Math.max(index - 1, 0));
  }
  function approve() {
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
    isPlaying,
    isAtEnd,
    next,
    previous,
    approve,
    togglePlay,
    reset,
  };
}
