import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, test } from 'vitest';
import { PlanTable } from '../src/components/PlanTable';
import { PlanVerdictStrip } from '../src/components/plans/PlanVerdictStrip';
import { demoIncident } from '../src/demoFixture';
import type { LiveExampleController } from '../src/liveExample/useLiveExampleFlow';
import { LiveExampleProgress } from '../src/shell/LiveExampleProgress';
import { TechnicalDock } from '../src/shell/TechnicalDock';
import { useConsoleStore } from '../src/store';
import type { Plan } from '../src/types';

const fullPlanId = 'b870e1f2-9da1-4b84-a581-3b0fd219c2aa';
const compactPlanId = 'b870e1f2…c2aa';

const plan: Plan = {
  ...demoIncident.plans[0],
  id: fullPlanId,
  name: 'Compact display test',
};

function controller(incidentId: string | null): LiveExampleController {
  return {
    stage: 'creating_incident',
    errorMessage: null,
    incidentId,
    recommendedNode: null,
    expectedInformationGainBits: null,
    plans: [],
    verifiedPlan: null,
    collectSample: () => {},
    approve: () => {},
    restart: () => {},
  };
}

beforeEach(() => {
  useConsoleStore.setState({
    selectedPlanId: null,
    dockCollapsed: false,
    dockTab: 'timeline',
    dockHeight: 190,
  });
});

test('plan identities are compact in operational UI while full identity remains inspectable and selected', () => {
  render(<PlanTable plans={[plan]} />);

  const planButton = screen.getByRole('button', {
    name: `${compactPlanId} · Compact display test`,
  });
  expect(planButton).toHaveAttribute('title', fullPlanId);
  fireEvent.click(planButton);
  expect(useConsoleStore.getState().selectedPlanId).toBe(fullPlanId);
});

test('LIVE progress only reveals a real locally-created incident ID after creation', () => {
  const { rerender } = render(
    <LiveExampleProgress controller={controller(null)} onExploreFallback={() => {}} />,
  );
  expect(screen.queryByText(/created locally/)).toBeNull();

  const fullIncidentId = '3d7cbd66-90ee-4c8e-bf1d-6d4a81751a22';
  rerender(
    <LiveExampleProgress controller={controller(fullIncidentId)} onExploreFallback={() => {}} />,
  );
  expect(screen.getByText('3d7cbd66…1a22')).toHaveAttribute('title', fullIncidentId);
  expect(screen.getByText(/created locally/)).toBeVisible();
});

test('Replay dock timeline explains its primary workspace instead of silently switching to Audit', () => {
  const { rerender } = render(<TechnicalDock incident={demoIncident} workspace="replay" />);

  expect(screen.getByRole('tab', { name: 'Timeline (Replay workspace)' })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  expect(screen.getByText('Timeline is displayed in the Replay workspace.')).toBeVisible();

  fireEvent.click(screen.getByRole('tab', { name: 'Audit' }));
  expect(screen.getByRole('tab', { name: 'Audit' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByText('INCIDENT DETECTED')).toBeVisible();

  useConsoleStore.getState().setDockTab('timeline');
  rerender(<TechnicalDock incident={demoIncident} workspace="response" />);
  expect(screen.getByRole('tab', { name: 'Timeline' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.queryByText('Timeline is displayed in the Replay workspace.')).toBeNull();
});

test('plan verdict strip adds compact service availability only when the simulator supplied it', () => {
  const onSelect = (id: string) => useConsoleStore.getState().selectPlan(id);
  render(<PlanVerdictStrip plans={[plan]} selectedPlanId={null} onSelect={onSelect} />);

  expect(screen.getByText((_, element) => element?.textContent === 'service 91.8%')).toBeVisible();
  const verdict = screen.getByRole('button', { name: /Compact display test/ });
  fireEvent.click(verdict);
  expect(useConsoleStore.getState().selectedPlanId).toBe(fullPlanId);
});
