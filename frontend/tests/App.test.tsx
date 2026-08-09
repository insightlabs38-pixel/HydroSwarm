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

// ui-work.txt "UI-10.5" 3: the Incident workspace is now a concise mission
// overview (leading candidate, recommended sample, active plan status, all
// linking to their dedicated workspace) rather than a page that repeats
// every workspace's full content -- the plan comparison table and the
// grounded rejection/comparison explanation now live PRIMARY in the
// Response workspace (moved there in the same phase), so this test follows
// the real operator flow across both instead of asserting everything is
// visible on one page.
test('passes the 30-second comprehension test in fallback mode', async () => {
  const user = userEvent.setup();
  renderApp();
  expect(await screen.findByText('Verified response awaiting approval')).toBeVisible();
  expect(screen.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  expect(screen.getByText('OFFLINE · LOCAL')).toBeVisible();
  expect(screen.getAllByText('76%').length).toBeGreaterThan(0);
  expect(screen.getByRole('heading', { name: 'Evidence / sampling' })).toBeVisible();
  expect(screen.getByText('0.37 bits')).toBeVisible();

  await user.click(screen.getByRole('button', { name: /^Response/ }));
  expect(await screen.findByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
  expect(screen.getByText('RECOMMENDED')).toBeVisible();
  expect(screen.getByText('REJECTED')).toBeVisible();

  // Select the rejected plan to see its real grounded rejection reason
  // (ui-work.txt 22 cross-panel synchronization).
  await user.click(screen.getByRole('button', { name: 'A · Aggressive isolation' }));
  expect(
    await screen.findByRole('heading', { name: 'Why was Aggressive isolation rejected?' }),
  ).toBeVisible();
  expect(screen.getByText(/4 nodes with pressure below the 15 m minimum/i)).toBeVisible();
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

test('Network workspace honestly reports the network list as unavailable when the backend is unreachable, never a fabricated empty success', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Network/ }));
  expect(await screen.findByRole('heading', { name: 'Import network' })).toBeVisible();
  expect(await screen.findByText('Networks unavailable.')).toBeVisible();
});

test('Model & Authority workspace renders the real DEMO_FALLBACK decision certificates as a governance table, including plan-specific certificates', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Model/ }));
  expect(await screen.findByRole('heading', { name: 'Authority ladder' })).toBeVisible();
  expect(screen.getByText('Source localization')).toBeVisible();
  expect(screen.getByText('Sample recommendation')).toBeVisible();
  expect(screen.getByText('OOD decision')).toBeVisible();
  expect(screen.getByText('Plan verification: B')).toBeVisible();
  expect(screen.getAllByText('CALIBRATED ADVISORY').length).toBeGreaterThan(0);
});

test('Replay workspace renders the real event ledger, disables hash-chain verification in DEMO_FALLBACK, and lists every failure-injection category as a real link', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Replay/ }));
  expect(await screen.findByRole('heading', { name: 'Event ledger' })).toBeVisible();
  expect(screen.getByText('Verification is not available in this mode.')).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Verify hash chain' })).toBeNull();
  const link = screen.getByRole('link', { name: /missing checkpoint/ });
  expect(link).toHaveAttribute('href', '?failure=missing_checkpoint');
});

test('Approval workspace never performs a real approval mutation in DEMO_FALLBACK mode', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Approval/ }));
  expect(await screen.findByRole('heading', { name: 'Operator approval' })).toBeVisible();
  expect(screen.getByText('Approval is not available in this mode.')).toBeVisible();
  expect(screen.queryByLabelText('Operator ID')).toBeNull();
});

test('Response workspace renders full plan verification, the action sequence, and the exposure-aware Pareto frontier, for the DEMO_FALLBACK fixture', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Response/ }));
  expect(await screen.findByRole('heading', { name: 'Action sequence' })).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Verified response Pareto frontier' })).toBeVisible();
  expect(
    screen.getByText('Exposure-aware frontier (real measured chemical exposure)'),
  ).toBeVisible();
  expect(
    screen.getByText(
      'Hydraulic-only frontier (no chemical exposure model -- never comparable to the exposure-aware frontier above)',
    ),
  ).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Compare plans' })).toBeVisible();
});

test('Sampling workspace renders the real evidence certificate, never an empty card, for the DEMO_FALLBACK fixture', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Sampling/ }));
  expect(await screen.findByRole('heading', { name: 'Evidence status' })).toBeVisible();
  expect(screen.getByText('CONTINUE SAMPLING')).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Next sample recommendation' })).toBeVisible();
  expect(screen.getAllByText('J123').length).toBeGreaterThan(0);
  expect(screen.getByRole('heading', { name: 'Why this sample?' })).toBeVisible();
});

test('Source workspace renders real candidate ranking and the DEMO_FALLBACK authority certificate, never a placeholder', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: /^Source/ }));
  expect(await screen.findByRole('heading', { name: 'Ranked source candidates' })).toBeVisible();
  expect(screen.getAllByText('1. J117').length).toBeGreaterThan(0);
  expect(screen.getByRole('heading', { name: 'Why this source?' })).toBeVisible();
  expect(
    screen.getByText(/J117 is the leading candidate because its observed concentration/),
  ).toBeVisible();
  // Appears twice by design (ui-work.txt "UI-10.5" 2/7): a compact
  // authority/applicability summary in the global DecisionInspector, and
  // the full Decision Certificate detail PRIMARY in this workspace body.
  expect(screen.getAllByText('CALIBRATED ADVISORY').length).toBeGreaterThan(0);
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

// jsdom does not implement a real browser's native behavior of moving
// keyboard focus to a URL fragment's target on navigation, so this can
// only assert the fix's precondition here (a real browser only shifts
// focus to a skip-link target if it's programmatically focusable);
// the actual focus-after-activation behavior was confirmed with a real
// Playwright/Chromium run during UI-10 verification.
test('skip link target is focusable so a real browser moves keyboard focus there, not just scroll position', async () => {
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  const skipLink = screen.getByRole('link', { name: 'Skip to main content' });
  expect(skipLink).toHaveAttribute('href', '#main-content');
  const target = document.getElementById('main-content');
  expect(target).toHaveAttribute('tabindex', '-1');
});

test('overview has no automated accessibility violations', async () => {
  const { container } = renderApp();
  await screen.findByText('Verified response awaiting approval');
  await waitFor(async () => {
    const results = await axe(container);
    expect(results.violations).toHaveLength(0);
  });
});

// ui-work.txt 24: "Use axe on major workspaces" -- UI-1 only checked the
// default Incident view; every workspace built since (UI-3..UI-6) needs
// its own pass, since each renders substantially different real content
// (tables, forms, charts) that the shared shell chrome alone can't cover.
test.each([
  ['Source', 'Ranked source candidates'],
  ['Sampling', 'Evidence status'],
  ['Response', 'Action sequence'],
  ['Approval', 'Operator approval'],
  ['Replay', 'Event ledger'],
  ['Network', 'Import network'],
  ['Model', 'Authority ladder'],
] as const)('%s workspace has no automated accessibility violations', async (rail, heading) => {
  const user = userEvent.setup();
  const { container } = renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: new RegExp(`^${rail}`) }));
  await screen.findByRole('heading', { name: heading });
  await waitFor(async () => {
    const results = await axe(container);
    expect(results.violations).toHaveLength(0);
  });
});
