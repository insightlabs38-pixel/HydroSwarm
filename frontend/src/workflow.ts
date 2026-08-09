import type { IncidentView } from './types';

export type StageStatus =
  'complete' | 'current' | 'waiting' | 'blocked' | 'caution' | 'unavailable';

/**
 * Single authoritative mapping from the backend controller's real
 * workflow stage (`incident.status`) to the Source/Sampling/Response/
 * Approval rail statuses and the operator's next step.
 *
 * Before this helper existed, Overview.tsx and DecisionInspector.tsx each
 * independently derived "next step" from retained artifacts
 * (`recommendedSample`, `plans`, `selectedPlanId`) rather than the
 * authoritative controller stage, and WorkflowRail derived rail-item
 * status the same way. Those artifacts can legitimately outlive the
 * stage that produced them -- e.g. `recommendedSample` is a real
 * historical field describing the last sample recommended during
 * SAMPLING, and stays populated after the controller has moved on to
 * PLANNING/APPROVAL/CLOSED, since nothing overwrites it with null on
 * stage transition. Deriving "next step" from its mere presence, instead
 * of from `incident.status`, could contradict the authoritative stage:
 * e.g. status=APPROVAL with approvalPending=true and a still-present
 * recommendedSample incorrectly saying "Collect recommended sample"
 * instead of "Approve verified plan" (the actual current requirement).
 *
 * `incident.status` is the single source of truth for "what stage is the
 * incident in right now"; this function is the single place that maps it
 * to UI stage/next-step text so WorkflowRail, Overview, and
 * DecisionInspector cannot drift out of agreement with each other again.
 */
export interface WorkflowProgression {
  source: StageStatus;
  sampling: StageStatus;
  response: StageStatus;
  approval: StageStatus;
  /** Concise, human-readable description of the operator's next step,
   * derived from `incident.status` first and only refined by real
   * per-stage detail (e.g. the recommended sample's node id) when that
   * detail is consistent with the current stage. */
  nextStep: string;
}

export function deriveWorkflowProgression(incident: IncidentView): WorkflowProgression {
  // Analysis itself failed to produce a usable incident -- every
  // downstream stage is honestly unavailable, not merely "not started".
  if (incident.mode === 'ERROR') {
    return {
      source: 'unavailable',
      sampling: 'unavailable',
      response: 'unavailable',
      approval: 'unavailable',
      nextStep: 'Resolve the incident-load error before continuing.',
    };
  }

  // Source localization always precedes SAMPLING/PLANNING/APPROVAL/CLOSED
  // in the product pipeline, so by the time incident.status carries any
  // of those four values, Source has already produced a result. Caution
  // (invalid calibration or non-NORMAL OOD) is an ongoing property of
  // that result, independent of which later stage the incident is now
  // in, so it applies across every stage rather than being stage-gated.
  const sourceCaution = !incident.calibrationValid || incident.ood !== 'NORMAL';
  const source: StageStatus = sourceCaution ? 'caution' : 'complete';

  const hasValidPlan = incident.plans.some(
    (plan) => plan.status === 'VALID' || plan.status === 'RECOMMENDED',
  );
  // Every generated plan was rejected by exact verification: a real
  // blocking condition (correct behavior is abstention, not forcing a
  // plan through), independent of controller stage.
  const noValidPlan = incident.plans.length > 0 && !hasValidPlan;

  let sampling: StageStatus;
  let response: StageStatus;
  let approval: StageStatus;
  let nextStep: string;

  switch (incident.status) {
    case 'SAMPLING':
      sampling = 'current';
      response = noValidPlan ? 'blocked' : 'waiting';
      approval = 'waiting';
      nextStep = incident.recommendedSample
        ? `Collect recommended sample at ${incident.recommendedSample.nodeId}`
        : 'Continue evidence collection';
      break;
    case 'PLANNING':
      sampling = 'complete';
      response = noValidPlan ? 'blocked' : 'current';
      approval = 'waiting';
      nextStep = noValidPlan
        ? 'No valid response plan yet -- every candidate was rejected by exact verification'
        : 'Review response plan';
      break;
    case 'APPROVAL':
      sampling = 'complete';
      response = 'complete';
      approval = incident.selectedPlanId
        ? 'complete'
        : incident.approvalPending
          ? 'current'
          : 'waiting';
      nextStep = incident.selectedPlanId
        ? 'Plan approved -- monitor for closure'
        : 'Approve verified plan';
      break;
    case 'CLOSED':
      sampling = 'complete';
      response = 'complete';
      approval = 'complete';
      nextStep = 'Incident closed -- review the replay/audit trail';
      break;
  }

  return { source, sampling, response, approval, nextStep };
}
