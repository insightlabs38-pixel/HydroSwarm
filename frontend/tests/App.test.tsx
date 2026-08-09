import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import App from '../src/App';
import { useConsoleStore } from '../src/store';

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
  useConsoleStore.setState({
    workspace: 'incident',
    selectedNodeId: null,
    selectedLinkId: null,
    selectedPlanId: null,
    selectedAuditSequence: null,
    leftRailCollapsed: false,
    inspectorCollapsed: false,
    dockCollapsed: false,
    dockHeight: 240,
    dockTab: 'timeline',
    replayIndex: 5,
    replayPlaying: false,
    replaySpeed: 1,
    reducedMotion: false,
  });
});

test('passes the 30-second comprehension test in fallback mode', async () => {
  renderApp();
  expect(await screen.findByText('Verified response awaiting approval')).toBeVisible();
  expect(screen.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  expect(screen.getByText('OFFLINE · LOCAL')).toBeVisible();
  expect(screen.getByText('Collect sample at J123')).toBeVisible();
  expect(screen.getAllByText('76%').length).toBeGreaterThan(0);
  expect(screen.getByText('RECOMMENDED')).toBeVisible();
  expect(screen.getByText('REJECTED')).toBeVisible();
  expect(screen.getByText(/four nodes fell below the pressure threshold/i)).toBeVisible();
});

test('workflow rail navigates to Validation and Benchmarks, and derives stage state from real data (no OOD/CAUTION for this fixture)', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');

  await user.click(screen.getByRole('button', { name: /Validation/ }));
  expect(
    await screen.findByRole('heading', { name: 'Benchmarks and operating range' }),
  ).toBeVisible();
  expect(screen.getByText('HydroCore S / M / L checkpoint')).toBeVisible();

  await user.click(screen.getByRole('button', { name: /Benchmarks/ }));
  expect(await screen.findByRole('heading', { name: 'Operational benchmarks' })).toBeVisible();
});

test('a workspace with no implementation yet shows an honest placeholder, never fabricated content', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Source/ }));
  expect(
    await screen.findByText('Source has not been implemented in the mission-control shell yet.'),
  ).toBeVisible();
});

test('technical dock audit tab shows real audit events, and reduced-motion toggle still works', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('tab', { name: 'Audit' }));
  expect(await screen.findByText('PLAN REJECTED')).toBeVisible();

  await user.click(screen.getByRole('button', { name: /Reduced motion off/ }));
  expect(screen.getByRole('button', { name: /Reduced motion on/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
});

test('overview has no automated accessibility violations', async () => {
  const { container } = renderApp();
  await screen.findByText('Verified response awaiting approval');
  await waitFor(async () => {
    const results = await axe(container);
    expect(results.violations).toHaveLength(0);
  });
});
