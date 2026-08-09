import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { OperationalMap } from '../src/components/OperationalMap';
import { demoIncident } from '../src/demoFixture';
import { useConsoleStore } from '../src/store';
import type { IncidentView } from '../src/types';

beforeEach(() => {
  useConsoleStore.setState({
    mapLayers: {
      assets: true,
      flow: true,
      concentration: true,
      candidates: true,
      sensors: true,
      sample: true,
      actions: true,
    },
    mapLayerControlVisible: true,
    selectedNodeId: null,
    selectedLinkId: null,
    selectedPlanId: null,
  });
});

test('layer checkboxes toggle the shared store, not just local state', async () => {
  const user = userEvent.setup();
  render(<OperationalMap incident={demoIncident} />);
  const candidatesCheckbox = screen.getByRole('checkbox', { name: /Candidates/ });
  expect(candidatesCheckbox).toBeChecked();
  await user.click(candidatesCheckbox);
  expect(candidatesCheckbox).not.toBeChecked();
  expect(useConsoleStore.getState().mapLayers.candidates).toBe(false);
});

test('Flow/Concentration checkboxes are disabled and unchecked when no link carries real data', () => {
  // demoIncident has real link flow/concentration values; a LIVE-shaped
  // incident (api.ts's viewFromApi) always maps them to null today.
  const liveShapedIncident: IncidentView = {
    ...demoIncident,
    links: demoIncident.links.map((link) => ({ ...link, flow: null, concentration: null })),
  };
  render(<OperationalMap incident={liveShapedIncident} />);
  const flowCheckbox = screen.getByRole('checkbox', { name: /Flow/ });
  const concentrationCheckbox = screen.getByRole('checkbox', { name: /Concentration/ });
  expect(flowCheckbox).toBeDisabled();
  expect(concentrationCheckbox).toBeDisabled();
  expect(screen.getAllByText('(data unavailable)')).toHaveLength(2);
});

test('the layer control panel hides entirely when toggled off from the toolbar', () => {
  useConsoleStore.setState({ mapLayerControlVisible: false });
  render(<OperationalMap incident={demoIncident} />);
  expect(screen.queryByRole('group', { name: 'Network map layers' })).toBeNull();
});

test('an incident with no node geometry renders an honest empty state, not a blank canvas', () => {
  render(<OperationalMap incident={{ ...demoIncident, nodes: [] }} />);
  expect(screen.getByText('No network geometry available for this incident.')).toBeVisible();
});
