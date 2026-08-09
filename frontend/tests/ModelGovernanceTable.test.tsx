import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { ModelGovernanceTable } from '../src/components/ModelGovernanceTable';

const SAMPLE_DOCUMENT = {
  schemaVersion: 'hydroswarm-model-governance-v1',
  models: [
    {
      id: 'classical',
      name: 'Classical',
      parameters: null,
      top1: 0.915,
      ece: null,
      latencyMs: null,
      decision: 'fallback',
    },
    {
      id: 'hydrocore-s-hybrid',
      name: 'HydroCore-S hybrid',
      parameters: 4040645,
      top1: 0.96,
      ece: 0.0269,
      latencyMs: 8.94,
      decision: 'promoted',
    },
    {
      id: 'hydrocore-l',
      name: 'HydroCore-L',
      parameters: 24420000,
      top1: null,
      ece: null,
      latencyMs: null,
      decision: 'ineligible',
    },
  ],
};

test('renders governed model rows with correct formatting once fetched', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({ ok: true, json: async () => SAMPLE_DOCUMENT }),
  );
  render(<ModelGovernanceTable />);

  await waitFor(() => expect(screen.getByText('HydroCore-S hybrid')).toBeInTheDocument());
  expect(screen.getByText('4.04M')).toBeInTheDocument();
  expect(screen.getByText('96.0%')).toBeInTheDocument();
  expect(screen.getByText('0.0269')).toBeInTheDocument();
  expect(screen.getByText('8.94 ms')).toBeInTheDocument();
  expect(screen.getByText('PROMOTED')).toBeInTheDocument();

  // HydroCore-L is explicitly not evaluated -- must say so, not show a fake number.
  expect(screen.getAllByText('not evaluated').length).toBeGreaterThan(0);
  expect(screen.getAllByText('profile only').length).toBeGreaterThan(0);
});

test('shows a clear error state when the governance artifact cannot be loaded', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')));
  render(<ModelGovernanceTable />);

  await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  expect(screen.getByRole('alert')).toHaveTextContent('network down');
});
