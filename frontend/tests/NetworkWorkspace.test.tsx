import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { NetworkWorkspace } from '../src/workspaces/NetworkWorkspace';
import { fetchNetworks, createIncidentForNetwork } from '../src/api/networks';
import { selectIncident } from '../src/incidentSelection';
import type { NetworkRecord } from '../src/types';

vi.mock('../src/api/networks', () => ({
  fetchNetworks: vi.fn(),
  importNetwork: vi.fn(),
  createIncidentForNetwork: vi.fn(),
}));
vi.mock('../src/incidentSelection', () => ({
  selectIncident: vi.fn(),
}));
// jsdom has no real layout engine, which crashes cytoscape's breadthfirst
// layout (it needs a real bounding box) -- this test is about the
// incident-creation form, not the topology preview, so stub it out the
// same way a headless test environment always has to for canvas/graph
// libraries.
vi.mock('cytoscape', () => ({
  default: vi.fn(() => ({ destroy: vi.fn() })),
}));

function network(overrides: Partial<NetworkRecord> = {}): NetworkRecord {
  return {
    networkId: 'net-1',
    name: 'loop-grid',
    version: 1,
    sha256: 'a'.repeat(64),
    nodeCount: 2,
    linkCount: 1,
    valid: true,
    validatedAt: '2026-08-10T00:00:00Z',
    nodes: [
      { nodeId: 'J1', nodeType: 'junction', elevationM: 305, coordinates: [900, 500] },
      { nodeId: 'R3', nodeType: 'reservoir', elevationM: 428, coordinates: [0, 500] },
    ],
    links: [{ linkId: 'P_R3_J1', linkType: 'pipe', startNode: 'R3', endNode: 'J1' }],
    validationErrors: [],
    ...overrides,
  };
}

function renderWorkspace() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <NetworkWorkspace />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

test('a validated network offers a compact create-incident form defaulting to a junction node', async () => {
  vi.mocked(fetchNetworks).mockResolvedValue([network()]);
  const user = userEvent.setup();
  renderWorkspace();

  await user.click(await screen.findByRole('button', { name: 'loop-grid' }));

  const heading = await screen.findByRole('heading', { name: 'Create incident' });
  const form = heading.closest('section');
  expect(form).not.toBeNull();
  const select = within(form as HTMLElement).getByLabelText('Sensor node');
  expect(select).toHaveValue('J1');
  // The real production signature-localization policy only recognizes
  // junctions as sensor candidates -- the reservoir R3 must never be
  // offered here, or a real submission against it would fail.
  expect(within(form as HTMLElement).queryByText('R3')).not.toBeInTheDocument();
});

test('submitting the compact form creates a real incident and selects it', async () => {
  vi.mocked(fetchNetworks).mockResolvedValue([network()]);
  vi.mocked(createIncidentForNetwork).mockResolvedValue('incident-new-1');
  const user = userEvent.setup();
  renderWorkspace();

  await user.click(await screen.findByRole('button', { name: 'loop-grid' }));
  await screen.findByRole('heading', { name: 'Create incident' });
  await user.click(screen.getByRole('button', { name: 'Create incident' }));

  await waitFor(() => expect(createIncidentForNetwork).toHaveBeenCalled());
  expect(createIncidentForNetwork).toHaveBeenCalledWith('net-1', {
    nodeId: 'J1',
    concentrationMgL: 0,
    pressureM: null,
  });
  await waitFor(() => expect(selectIncident).toHaveBeenCalledWith('incident-new-1'));
});

test('an invalid network never offers incident creation', async () => {
  vi.mocked(fetchNetworks).mockResolvedValue([
    network({ valid: false, validationErrors: ['bad'] }),
  ]);
  const user = userEvent.setup();
  renderWorkspace();

  await user.click(await screen.findByRole('button', { name: 'loop-grid' }));
  await waitFor(() => expect(screen.getAllByText('INVALID')).toHaveLength(2));
  expect(screen.queryByRole('heading', { name: 'Create incident' })).not.toBeInTheDocument();
});
