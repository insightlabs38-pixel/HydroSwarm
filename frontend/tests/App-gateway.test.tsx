import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import * as referenceDemoApi from '../src/api/referenceDemo';
import type { ApiReferenceArtifact, ApiReferenceMilestone } from '../src/reference/types';

// SUB-5/SUB-6: this file specifically exercises the first-launch gateway,
// which only appears when no LIVE incident is configured. api/incident.ts
// reads VITE_INCIDENT_ID into a module-scope constant once at import time
// (tests/api-incident.test.ts hit this same issue) -- vi.resetModules() +
// a dynamic import *after* stubbing the env, every test, is required for
// the stub to actually take effect here, since tests/setup.ts's global
// stub (a configured incident, for the rest of the suite) already ran
// before this file's own top-level imports would otherwise freeze it.
beforeEach(() => {
  vi.resetModules();
  window.history.pushState(null, '', '/');
});

function alertMilestone(): ApiReferenceMilestone {
  return {
    index: 0,
    milestone_id: 'alert',
    label: 'Incident detected',
    controller_state: 'NETWORK_VALIDATED',
    event_sequence_start: 0,
    event_sequence_end: 2,
    auto_advance: true,
    pause_reason: null,
    pause_action: null,
    pause_action_label: null,
    highlight: 'incident_opened',
    narrative: 'A contamination alert opens the incident.',
    incident_view: {
      incident_id: 'incident-1',
      network_id: 'golden-network-v1',
      ood_level: 'NORMAL',
      controller_state: 'NETWORK_VALIDATED',
      candidates: null,
      candidate_region: null,
      evidence_sufficient: null,
      recommended_sample: null,
      sample_observation: null,
      plans: null,
      selected_plan_id: null,
      approved_plan_id: null,
      approval_pending: false,
      final_event_hash: null,
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
  network_topology: { nodes: [], links: [] },
  milestones: [alertMilestone()],
  artifact_sha256: 'artifact-hash',
};

vi.mock('../src/api/referenceDemo', () => ({
  fetchReferenceArtifact: async () => fakeArtifact,
}));

async function renderFreshApp() {
  const { default: App } = await import('../src/App');
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

test('a fresh install with no configured incident and no routing param shows the first-launch gateway', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
  await renderFreshApp();

  expect(
    await screen.findByRole('heading', { name: /Local incident decision support/ }),
  ).toBeVisible();
  expect(screen.queryByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).not.toBeInTheDocument();
});

test('?experience=reference bypasses the gateway entirely', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  window.history.pushState(null, '', '/?experience=reference');
  await renderFreshApp();

  expect(
    await screen.findByText('REFERENCE INCIDENT · VERIFIED REPLAY', { exact: false }),
  ).toBeVisible();
  expect(
    screen.queryByRole('heading', { name: /Local incident decision support/ }),
  ).not.toBeInTheDocument();
});

test('clicking Run Reference Incident on the gateway shows the REFERENCE banner', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
  const user = userEvent.setup();
  await renderFreshApp();

  await user.click(await screen.findByRole('button', { name: /Run Reference Incident/ }));

  expect(
    await screen.findByText('REFERENCE INCIDENT · VERIFIED REPLAY', { exact: false }),
  ).toBeVisible();
  expect(new URLSearchParams(window.location.search).get('experience')).toBe('reference');
});

test('clicking Explore illustrative fallback on the gateway shows DEMO_FALLBACK', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
  const user = userEvent.setup();
  await renderFreshApp();

  await user.click(await screen.findByRole('button', { name: 'Explore illustrative fallback' }));

  expect(await screen.findByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
});

test('a failed reference-demo fetch shows an error, not an infinite loading spinner', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  window.history.pushState(null, '', '/?experience=reference');
  vi.spyOn(referenceDemoApi, 'fetchReferenceArtifact').mockRejectedValueOnce(
    new Error('reference-demo artifact not found'),
  );
  await renderFreshApp();

  expect(await screen.findByText('Reference incident unavailable.')).toBeVisible();
  expect(screen.getByText('reference-demo artifact not found')).toBeVisible();
  expect(screen.queryByText('Loading local incident state…')).not.toBeInTheDocument();
});

test('?failure= bypasses the gateway and still shows the ERROR state', async () => {
  vi.stubEnv('VITE_INCIDENT_ID', '');
  window.history.pushState(null, '', '/?failure=missing_checkpoint');
  await renderFreshApp();

  expect(await screen.findByText('INCIDENT UNAVAILABLE')).toBeVisible();
  expect(
    screen.queryByRole('heading', { name: /Local incident decision support/ }),
  ).not.toBeInTheDocument();
});
