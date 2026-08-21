import type { IncidentView } from './types';

/** ui-work.txt / hotfix: single place that decides the source-ranking
 * eyebrow label so REFERENCE, CLASSICAL_SAFE and the normal hybrid/fused
 * path can't drift apart across render sites (SourceWorkspace, Overview).
 * CLASSICAL_SAFE must never be labeled as fused -- learned localization is
 * explicitly unavailable for that incident. */
export function sourceRankingLabel(incident: IncidentView): string {
  if (incident.mode === 'REFERENCE') return 'DETERMINISTIC REFERENCE LOCALIZATION';
  if (incident.runtimeAnalysisMode === 'CLASSICAL_SAFE') {
    return 'DETERMINISTIC CLASSICAL LOCALIZATION';
  }
  return 'FUSED SOURCE BELIEF';
}

/** Whether the "classical + Sentinel" composition sentence is truthful for
 * this incident -- only when learned Sentinel evidence actually
 * contributes to the final ranking. */
export function showsFusedComposition(incident: IncidentView): boolean {
  return incident.mode !== 'REFERENCE' && incident.runtimeAnalysisMode !== 'CLASSICAL_SAFE';
}
