import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { ApprovalWorkspace } from '../src/workspaces/ApprovalWorkspace';
import { ApiError } from '../src/api/client';
import { demoIncident } from '../src/demoFixture';
import type { IncidentView, Plan, PlanVerificationView } from '../src/types';

vi.mock('../src/api/approval', () => ({
  approvePlan: vi.fn(),
}));

function renderWorkspace(incident: IncidentView) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ApprovalWorkspace incident={incident} />
    </QueryClientProvider>,
  );
}

function verification(overrides: Partial<PlanVerificationView>): PlanVerificationView {
  return {
    decision: 'VERIFIED',
    simulator: 'wntr-epanet',
    simulatorVersion: '1.2.0',
    stateHash: 'a'.repeat(64),
    consequences: null,
    worstCaseConsequences: null,
    evaluationProvenance: null,
    rejectionCodes: [],
    abstentionReason: null,
    verifiedAt: '2026-08-03T08:00:00Z',
    contextHash: 'ctx',
    verificationStatus: 'CURRENT',
    ...overrides,
  };
}

function planWith(overrides: Partial<Plan>): Plan {
  return {
    id: 'plan-x',
    name: 'Test plan',
    exposureReduction: null,
    actions: [
      {
        actionType: 'MONITOR_NODE',
        targetId: 'J1',
        startMinute: 0,
        durationMinutes: 10,
        flowRateLps: null,
      },
    ],
    status: 'VALID',
    verification: null,
    ...overrides,
  };
}

function liveIncidentWith(plan: Plan): IncidentView {
  return {
    ...demoIncident,
    mode: 'LIVE',
    id: 'live-incident-1',
    plans: [plan],
    selectedPlanId: null,
    recommendedPlanId: plan.id,
    approvalPending: true,
  };
}

test('an unverified plan cannot be approved -- no confirmation form is shown', () => {
  const incident = liveIncidentWith(planWith({ verification: null }));
  renderWorkspace(incident);
  expect(screen.getByText('This plan cannot be approved.')).toBeVisible();
  expect(screen.queryByLabelText('Operator ID')).toBeNull();
});

test('a REJECTED plan cannot be approved -- no confirmation form is shown', () => {
  const incident = liveIncidentWith(
    planWith({
      status: 'REJECTED',
      verification: verification({
        decision: 'REJECTED',
        rejectionCodes: ['PRESSURE_BELOW_MINIMUM'],
      }),
    }),
  );
  renderWorkspace(incident);
  expect(screen.getByText('This plan cannot be approved.')).toBeVisible();
  expect(screen.getByText(/Verification decision is REJECTED/)).toBeVisible();
  expect(screen.queryByLabelText('Operator ID')).toBeNull();
});

test('a STALE verification cannot be approved -- no confirmation form is shown', () => {
  const incident = liveIncidentWith(
    planWith({ verification: verification({ verificationStatus: 'STALE' }) }),
  );
  renderWorkspace(incident);
  expect(screen.getByText('Verification is stale.')).toBeVisible();
  expect(
    screen.getByText(
      /Verification is stale because incident evidence or verification context changed/,
    ),
  ).toBeVisible();
  expect(screen.queryByLabelText('Operator ID')).toBeNull();
});

test('a current VERIFIED plan can enter confirmation, but Approve stays disabled until both operator ID and the review checkbox are filled', async () => {
  const user = userEvent.setup();
  const incident = liveIncidentWith(planWith({ verification: verification({}) }));
  renderWorkspace(incident);

  const approveButton = screen.getByRole('button', { name: 'Approve verified plan' });
  expect(approveButton).toBeDisabled();

  await user.type(screen.getByLabelText('Operator ID'), 'operator-42');
  expect(approveButton).toBeDisabled();

  await user.click(screen.getByLabelText(/I reviewed the verified actions/));
  expect(approveButton).toBeEnabled();

  // Clearing the operator ID re-disables it -- confirmation alone is not enough.
  await user.clear(screen.getByLabelText('Operator ID'));
  expect(approveButton).toBeDisabled();
});

test('a real backend stale-verification 409 fails closed: no receipt, the real backend reason is shown, no silent retry', async () => {
  const user = userEvent.setup();
  const { approvePlan } = await import('../src/api/approval');
  vi.mocked(approvePlan).mockRejectedValue(
    new ApiError(
      409,
      'HydroSwarm API 409: verification is stale: incident evidence has changed since this plan was verified; re-verify before approval',
    ),
  );
  const incident = liveIncidentWith(planWith({ verification: verification({}) }));
  renderWorkspace(incident);

  await user.type(screen.getByLabelText('Operator ID'), 'operator-42');
  await user.click(screen.getByLabelText(/I reviewed the verified actions/));
  await user.click(screen.getByRole('button', { name: 'Approve verified plan' }));

  await waitFor(() => {
    expect(screen.getByRole('alert')).toHaveTextContent(/verification is stale/);
  });
  expect(screen.queryByText('Approval receipt')).toBeNull();
});

test('a successful approval shows the real receipt from the backend response', async () => {
  const user = userEvent.setup();
  const { approvePlan } = await import('../src/api/approval');
  vi.mocked(approvePlan).mockResolvedValue({
    incidentId: 'live-incident-1',
    planId: 'plan-x',
    approved: true,
    operatorId: 'operator-42',
    approvedAt: '2026-08-03T09:00:00Z',
  });
  const incident = liveIncidentWith(planWith({ verification: verification({}) }));
  renderWorkspace(incident);

  await user.type(screen.getByLabelText('Operator ID'), 'operator-42');
  await user.click(screen.getByLabelText(/I reviewed the verified actions/));
  await user.click(screen.getByRole('button', { name: 'Approve verified plan' }));

  expect(await screen.findByText('Approval receipt')).toBeVisible();
  expect(screen.getByText('operator-42')).toBeVisible();
});
