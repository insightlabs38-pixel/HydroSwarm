import { describe, expect, test } from 'vitest';
import { deriveWorkflowProgression } from '../src/workflow';
import { demoIncident } from '../src/demoFixture';
import type { IncidentView } from '../src/types';

function withOverrides(overrides: Partial<IncidentView>): IncidentView {
  return { ...demoIncident, ...overrides };
}

describe('deriveWorkflowProgression', () => {
  test('SAMPLING: sampling is current, response/approval wait, next step names the recommended sample', () => {
    const incident = withOverrides({
      status: 'SAMPLING',
      plans: [],
      selectedPlanId: null,
      recommendedPlanId: null,
      approvalPending: false,
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.source).toBe('complete');
    expect(progression.sampling).toBe('current');
    expect(progression.response).toBe('waiting');
    expect(progression.approval).toBe('waiting');
    expect(progression.nextStep).toBe(
      `Collect recommended sample at ${demoIncident.recommendedSample!.nodeId}`,
    );
  });

  test('SAMPLING: no recommended sample falls back to a generic evidence-collection next step', () => {
    const incident = withOverrides({
      status: 'SAMPLING',
      plans: [],
      selectedPlanId: null,
      recommendedPlanId: null,
      approvalPending: false,
      recommendedSample: null,
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.nextStep).toBe('Continue evidence collection');
  });

  test('PLANNING: sampling is complete, response is current, next step asks to review the plan', () => {
    const incident = withOverrides({
      status: 'PLANNING',
      selectedPlanId: null,
      approvalPending: false,
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.source).toBe('complete');
    expect(progression.sampling).toBe('complete');
    expect(progression.response).toBe('current');
    expect(progression.approval).toBe('waiting');
    expect(progression.nextStep).toBe('Review response plan');
  });

  test('PLANNING: every plan rejected blocks response instead of showing it as current', () => {
    const rejectedOnly = demoIncident.plans.map((plan) => ({
      ...plan,
      status: 'REJECTED' as const,
    }));
    const incident = withOverrides({
      status: 'PLANNING',
      plans: rejectedOnly,
      selectedPlanId: null,
      approvalPending: false,
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.response).toBe('blocked');
    expect(progression.nextStep).toMatch(/no valid response plan/i);
  });

  // This is the exact contradiction reported: incident.status = APPROVAL,
  // approvalPending = true, and an earlier recommendedSample is still
  // present (a real, retained artifact from the SAMPLING stage that
  // nothing clears on stage transition). The next step must follow the
  // authoritative controller stage (APPROVAL), never the stale artifact.
  test('APPROVAL: a stale recommendedSample never overrides the authoritative approval next step', () => {
    const incident = withOverrides({
      status: 'APPROVAL',
      approvalPending: true,
      selectedPlanId: null,
    });
    expect(incident.recommendedSample).not.toBeNull();
    const progression = deriveWorkflowProgression(incident);
    expect(progression.source).toBe('complete');
    expect(progression.sampling).toBe('complete');
    expect(progression.response).toBe('complete');
    expect(progression.approval).toBe('current');
    expect(progression.nextStep).toBe('Approve verified plan');
  });

  test('APPROVAL: an already-approved plan is reflected as complete, not pending', () => {
    const incident = withOverrides({
      status: 'APPROVAL',
      approvalPending: false,
      selectedPlanId: 'B',
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.approval).toBe('complete');
    expect(progression.nextStep).toBe('Plan approved -- monitor for closure');
  });

  test('CLOSED: every stage is complete and the next step points to replay/audit', () => {
    const incident = withOverrides({
      status: 'CLOSED',
      approvalPending: false,
      selectedPlanId: 'B',
    });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.source).toBe('complete');
    expect(progression.sampling).toBe('complete');
    expect(progression.response).toBe('complete');
    expect(progression.approval).toBe('complete');
    expect(progression.nextStep).toMatch(/incident closed/i);
  });

  test('ERROR mode: every downstream stage is honestly unavailable, not merely unstarted', () => {
    const incident = withOverrides({ mode: 'ERROR', status: 'SAMPLING' });
    const progression = deriveWorkflowProgression(incident);
    expect(progression.source).toBe('unavailable');
    expect(progression.sampling).toBe('unavailable');
    expect(progression.response).toBe('unavailable');
    expect(progression.approval).toBe('unavailable');
    expect(progression.nextStep).toMatch(/resolve the incident-load error/i);
  });

  test('caution overrides source completeness when calibration is invalid or OOD is not NORMAL, independent of stage', () => {
    const invalidCalibration = deriveWorkflowProgression(
      withOverrides({ status: 'PLANNING', calibrationValid: false }),
    );
    expect(invalidCalibration.source).toBe('caution');

    const outOfDistribution = deriveWorkflowProgression(
      withOverrides({ status: 'CLOSED', ood: 'CAUTION' }),
    );
    expect(outOfDistribution.source).toBe('caution');
  });
});
