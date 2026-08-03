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
    page: 'overview',
    timeIndex: 5,
    playing: false,
    speed: 1,
    reducedMotion: false,
    selectedPlan: 'B',
  });
});

test('passes the 30-second comprehension test in fallback mode', async () => {
  renderApp();
  expect(await screen.findByText('Verified response awaiting approval')).toBeVisible();
  expect(screen.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  expect(screen.getByText('OFFLINE · LOCAL')).toBeVisible();
  expect(screen.getByText('OOD NORMAL')).toBeVisible();
  expect(screen.getByText('HUMAN APPROVAL PENDING')).toBeVisible();
  expect(screen.getByText('Collect sample at J123')).toBeVisible();
  expect(screen.getAllByText('76%')).toHaveLength(2);
  expect(screen.getByText('RECOMMENDED')).toBeVisible();
  expect(screen.getByText('REJECTED')).toBeVisible();
  expect(screen.getByText(/four nodes fell below the pressure threshold/i)).toBeVisible();
});

test('keyboard-operable navigation exposes audit and validation state', async () => {
  const user = userEvent.setup();
  renderApp();
  await screen.findByText('Verified response awaiting approval');
  await user.click(screen.getByRole('button', { name: 'Audit' }));
  expect(
    await screen.findByRole('heading', { name: 'Incident audit and replay' }),
  ).toBeVisible();
  expect(screen.getByText('PLAN REJECTED')).toBeVisible();
  await user.click(screen.getByRole('button', { name: 'Validation' }));
  expect(
    await screen.findByRole('heading', { name: 'Benchmarks and operating range' }),
  ).toBeVisible();
  expect(screen.getByText('Unseen network Top-3')).toBeVisible();
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
