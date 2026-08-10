import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FirstLaunchGateway } from '../src/shell/FirstLaunchGateway';

test('renders the SS5 target copy and all four actions', () => {
  render(
    <FirstLaunchGateway
      onRunReference={() => {}}
      onRunLive={() => {}}
      onImportNetwork={() => {}}
      onExploreFallback={() => {}}
    />,
  );

  expect(screen.getByText('HydroSwarm is ready')).toBeVisible();
  expect(
    screen.getByText(
      /Offline mission-control decision support for drinking-water contamination incidents\./,
    ),
  ).toBeVisible();
  expect(screen.getByRole('button', { name: /Run Reference Incident/ })).toBeVisible();
  expect(screen.getByText('Recommended')).toBeVisible();
  expect(screen.getByRole('button', { name: /Run Live Example/ })).toBeVisible();
  expect(screen.getByText('Real computation, reference inputs')).toBeVisible();
  expect(screen.getByRole('button', { name: /Import Your Own Network/ })).toBeVisible();
  expect(screen.getByText('Advanced')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Explore illustrative fallback' })).toBeVisible();
});

test('each action invokes its own handler, not a shared one', async () => {
  const user = userEvent.setup();
  const onRunReference = vi.fn();
  const onRunLive = vi.fn();
  const onImportNetwork = vi.fn();
  const onExploreFallback = vi.fn();
  render(
    <FirstLaunchGateway
      onRunReference={onRunReference}
      onRunLive={onRunLive}
      onImportNetwork={onImportNetwork}
      onExploreFallback={onExploreFallback}
    />,
  );

  await user.click(screen.getByRole('button', { name: /Run Reference Incident/ }));
  expect(onRunReference).toHaveBeenCalledTimes(1);
  expect(onRunLive).not.toHaveBeenCalled();

  await user.click(screen.getByRole('button', { name: /Run Live Example/ }));
  expect(onRunLive).toHaveBeenCalledTimes(1);

  await user.click(screen.getByRole('button', { name: /Import Your Own Network/ }));
  expect(onImportNetwork).toHaveBeenCalledTimes(1);

  await user.click(screen.getByRole('button', { name: 'Explore illustrative fallback' }));
  expect(onExploreFallback).toHaveBeenCalledTimes(1);
});
