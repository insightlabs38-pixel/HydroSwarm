import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { useReferenceIncident } from '../src/reference/useReferenceIncident';
import type { ApiReferenceArtifact, ApiReferenceMilestone } from '../src/reference/types';

function milestone(
  id: string,
  autoAdvance: boolean,
  pauseReason: string | null = null,
): ApiReferenceMilestone {
  return {
    index: 0,
    milestone_id: id,
    label: id,
    controller_state: 'STATE',
    event_sequence_start: 0,
    event_sequence_end: 0,
    auto_advance: autoAdvance,
    pause_reason: pauseReason,
    highlight: 'x',
    narrative: `narrative for ${id}`,
    incident_view: {
      incident_id: 'incident-1',
      network_id: 'golden-network-v1',
      ood_level: 'NORMAL',
      controller_state: 'STATE',
      candidates: null,
      candidate_region: null,
      evidence_sufficient: null,
      recommended_sample: null,
      sample_observation: null,
      plans: null,
      selected_plan_id: null,
      approved_plan_id: null,
      approval_pending: id === 'boundary',
      final_event_hash: id === 'completed' ? 'final-hash' : null,
    },
  };
}

const fakeArtifact: ApiReferenceArtifact = {
  schema_version: 'hydroswarm-reference-incident-v1',
  reference_id: 'reference-incident-v1',
  title: 'test',
  description: 'test',
  generator: 'test',
  generated_at: '2026-08-10T00:00:00+00:00',
  source_commit: 'deadbeef',
  network_sha256: 'network-hash',
  golden_result_hash: 'golden-hash',
  final_event_hash: 'final-hash',
  event_count: 3,
  network_topology: null,
  milestones: [
    milestone('alert', true),
    milestone('boundary', false, 'awaiting approval'),
    milestone('completed', true),
  ],
  artifact_sha256: 'artifact-hash',
};

vi.mock('../src/api/referenceDemo', () => ({
  fetchReferenceArtifact: async () => fakeArtifact,
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
});

test('auto-advances through an auto_advance milestone after the fixed delay', async () => {
  const { result } = renderHook(() => useReferenceIncident(false), { wrapper });
  await waitFor(() => expect(result.current.incident).not.toBeNull());
  expect(result.current.milestoneIndex).toBe(0);

  await act(async () => {
    await vi.advanceTimersByTimeAsync(3300);
  });

  expect(result.current.milestoneIndex).toBe(1);
});

test('pauses at a non-auto_advance milestone and does not advance on its own', async () => {
  const { result } = renderHook(() => useReferenceIncident(false), { wrapper });
  await waitFor(() => expect(result.current.incident).not.toBeNull());

  await act(async () => {
    await vi.advanceTimersByTimeAsync(3300);
  });
  expect(result.current.milestoneIndex).toBe(1);
  expect(result.current.isPaused).toBe(true);
  expect(result.current.pauseReason).toBe('awaiting approval');

  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000);
  });
  expect(result.current.milestoneIndex).toBe(1);
});

test('approve() advances past a paused milestone', async () => {
  const { result } = renderHook(() => useReferenceIncident(false), { wrapper });
  await waitFor(() => expect(result.current.incident).not.toBeNull());
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3300);
  });
  expect(result.current.isPaused).toBe(true);

  act(() => {
    result.current.approve();
  });

  expect(result.current.milestoneIndex).toBe(2);
  expect(result.current.isAtEnd).toBe(true);
});

test('reduced motion disables the auto-advance timer -- no fake waiting, manual Next only', async () => {
  const { result } = renderHook(() => useReferenceIncident(true), { wrapper });
  await waitFor(() => expect(result.current.incident).not.toBeNull());

  await act(async () => {
    await vi.advanceTimersByTimeAsync(20_000);
  });
  expect(result.current.milestoneIndex).toBe(0);

  act(() => {
    result.current.next();
  });
  expect(result.current.milestoneIndex).toBe(1);
});

test('togglePlay pauses auto-advance without discarding state, reset returns to the first milestone', async () => {
  const { result } = renderHook(() => useReferenceIncident(false), { wrapper });
  await waitFor(() => expect(result.current.incident).not.toBeNull());

  act(() => {
    result.current.togglePlay();
  });
  expect(result.current.isPlaying).toBe(false);

  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000);
  });
  expect(result.current.milestoneIndex).toBe(0);

  act(() => {
    result.current.next();
  });
  expect(result.current.milestoneIndex).toBe(1);

  act(() => {
    result.current.reset();
  });
  expect(result.current.milestoneIndex).toBe(0);
  expect(result.current.isPlaying).toBe(true);
});
