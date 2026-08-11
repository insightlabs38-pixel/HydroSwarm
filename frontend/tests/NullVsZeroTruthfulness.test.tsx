import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../src/App';
import { demoIncident } from '../src/demoFixture';
import type { IncidentView } from '../src/types';

// UI-11.1 §2: component-level proof that null (unavailable) and legitimate
// numeric 0 render as genuinely different, distinguishable UI states --
// the unit-level api-incident.test.ts coverage proves the *mapping* keeps
// them distinct; this proves the *rendering* does too (a component could
// still collapse them back together, e.g. `value ?? 0` in a render path).

let currentIncident: IncidentView = demoIncident;

vi.mock('../src/api/incident', () => ({
  fetchIncidentWithFallback: async () => currentIncident,
  hasConfiguredLiveIncident: () => true,
  requestedExperience: () => null,
}));

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

test('a null disagreement renders "not measured", never a fabricated 0.0%', async () => {
  currentIncident = { ...demoIncident, disagreement: null };
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  expect(screen.getAllByText(/not measured/i).length).toBeGreaterThan(0);
  expect(screen.queryByText('0.0%')).toBeNull();
});

test('a genuine 0 disagreement renders "0.0%", not "not measured"', async () => {
  currentIncident = { ...demoIncident, disagreement: 0 };
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  expect(screen.getAllByText(/0\.0%/).length).toBeGreaterThan(0);
});

test('a null sensor pressure/concentration renders as "not measured" text in the hydraulic chart, not "null"', async () => {
  const user = userEvent.setup();
  currentIncident = {
    ...demoIncident,
    hydraulicSeries: null, // force the live-snapshot (sensors) branch, not the series branch
    nodes: demoIncident.nodes.map((node) =>
      node.sensor
        ? { ...node, sensor: { ...node.sensor, pressure: null, concentration: null } }
        : node,
    ),
  };
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: 'Expand technical dock' }));
  await user.click(screen.getByRole('tab', { name: 'Hydraulics' }));
  const dockPanel = document.getElementById('dock-panel-hydraulics');
  expect(dockPanel).not.toBeNull();
  expect(dockPanel!.textContent).toMatch(/not measured/i);
  expect(dockPanel!.textContent).not.toMatch(/pressure null/i);
  expect(dockPanel!.textContent).not.toMatch(/concentration null/i);
});
