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

  expect(screen.getByRole('heading', { name: /Local incident decision support/ })).toBeVisible();
  expect(
    screen.getByText(
      /Localize suspected sources, collect evidence selectively, verify response options with WNTR\/EPANET, and keep the final decision with a human operator\./,
    ),
  ).toBeVisible();
  expect(screen.getByRole('button', { name: /Run Reference Incident/ })).toBeVisible();
  expect(screen.getByText(/Recommended.*deterministic checksummed workflow replay/)).toBeVisible();
  expect(screen.getByRole('button', { name: /Run Live Example/ })).toBeVisible();
  expect(
    screen.getByText(/Current HydroCore-v5 runtime.*real computation on reference inputs/),
  ).toBeVisible();
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
