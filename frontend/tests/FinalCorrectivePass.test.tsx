/**
 * Focused regression tests for the Final Frontend Corrective + Documentation
 * Freeze pass.
 *
 * A. Source attribution
 * B. Benchmarks
 * C. Validation
 * D. Reference Authority
 * E. First launch
 * F. Stale assets
 */
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { FirstLaunchGateway } from '../src/shell/FirstLaunchGateway';
import { SourceWorkspace } from '../src/workspaces/SourceWorkspace';
import { AuthorityWorkspace } from '../src/workspaces/AuthorityWorkspace';
import { Overview } from '../src/pages/Overview';
import { BenchmarkPage } from '../src/pages/BenchmarkPage';
import { ValidationPage } from '../src/pages/ValidationPage';
import { demoIncident } from '../src/demoFixture';
import type { IncidentView } from '../src/types';

function renderWithQuery(node: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// A. Source attribution
// ---------------------------------------------------------------------------
describe('A. Source attribution', () => {
  test('LIVE/DEMO SourceWorkspace labels the final ranked candidates as fused, not raw Sentinel', async () => {
    renderWithQuery(<SourceWorkspace incident={demoIncident} />);
    expect(await screen.findByText('CALIBRATED FUSION')).toBeVisible();
    expect(screen.queryByText('HYDROCORE-v5 SENTINEL')).toBeNull();
    expect(
      screen.getByText('Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.'),
    ).toBeVisible();
  });

  test('REFERENCE SourceWorkspace still says deterministic reference localization', async () => {
    const referenceIncident: IncidentView = { ...demoIncident, mode: 'REFERENCE' };
    renderWithQuery(<SourceWorkspace incident={referenceIncident} />);
    expect(await screen.findByText('DETERMINISTIC REFERENCE LOCALIZATION')).toBeVisible();
    expect(screen.queryByText('CALIBRATED FUSION')).toBeNull();
    // No fusion-composition sentence in REFERENCE mode -- it does not fuse anything live.
    expect(
      screen.queryByText(
        'Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.',
      ),
    ).toBeNull();
  });

  test('Overview Source card labels the final ranking as fused belief, not raw Sentinel', async () => {
    render(<Overview incident={demoIncident} />);
    expect(await screen.findByText('FUSED SOURCE BELIEF')).toBeVisible();
    expect(screen.queryByText('SENTINEL')).toBeNull();
  });

  test('Overview Source card for REFERENCE incidents says deterministic reference localization', async () => {
    const referenceIncident: IncidentView = { ...demoIncident, mode: 'REFERENCE' };
    render(<Overview incident={referenceIncident} />);
    expect(await screen.findByText('DETERMINISTIC REFERENCE LOCALIZATION')).toBeVisible();
    expect(screen.queryByText('FUSED SOURCE BELIEF')).toBeNull();
  });

  test('learned HydroCore-v5 Sentinel remains visible as an advisory subsystem on the authority path', async () => {
    renderWithQuery(<AuthorityWorkspace incident={demoIncident} />);
    expect(await screen.findByText('HydroCore-v5 Sentinel')).toBeVisible();
    expect(screen.getByText('ADVISORY')).toBeVisible();
    expect(
      screen.getByText(
        /Source fusion combines deterministic hydraulic\/signature evidence with HydroCore-v5 Sentinel evidence/,
      ),
    ).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// B. Benchmarks
// ---------------------------------------------------------------------------
describe('B. Benchmarks', () => {
  const sampleDoc = {
    schemaVersion: 'hydroswarm-system-regression-evidence-v1',
    generatedFrom: [{ path: 'reports/results/summary.md', sha256: 'abc123' }],
    note: 'test doc',
    metrics: [
      {
        metric: 'Top-1 source localization',
        value: '100%',
        comparison: '3 seeded golden runs',
        status: 'PASS',
      },
    ],
    limitations: ['This is a regression benchmark, not field evidence.'],
  };

  test('renders provenance-backed static regression evidence instead of an empty measured-results table', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => sampleDoc }));
    render(<BenchmarkPage />);
    expect(await screen.findByText('Top-1 source localization')).toBeVisible();
    expect(screen.getByText('reports/results/summary.md')).toBeVisible();
    expect(screen.getByText('This is a regression benchmark, not field evidence.')).toBeVisible();
  });

  test('shows a truthful empty/error state, not a blank table pretending measured evidence exists, when the artifact cannot be loaded', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    render(<BenchmarkPage />);
    expect(await screen.findByText('Regression evidence unavailable.')).toBeVisible();
    expect(screen.queryByRole('table')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// C. Validation
// ---------------------------------------------------------------------------
describe('C. Validation', () => {
  test('does not characterize novel-topology transfer as unsupported "weak"', () => {
    render(<ValidationPage />);
    expect(screen.queryByText(/measured but weak/)).toBeNull();
    expect(
      screen.getByText(/Unseen-topology transfer retains measurable localization signal/),
    ).toBeVisible();
  });

  test('still marks topology predictive metrics DESCRIPTIVE / NON-GATING', () => {
    render(<ValidationPage />);
    expect(screen.getByText(/DESCRIPTIVE \/ NON-GATING/)).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// D. Reference Authority
// ---------------------------------------------------------------------------
describe('D. Reference Authority', () => {
  test('REFERENCE explains why live DecisionCertificates are not applicable, without fabricating one', async () => {
    const referenceIncident: IncidentView = { ...demoIncident, mode: 'REFERENCE' };
    renderWithQuery(<AuthorityWorkspace incident={referenceIncident} />);
    expect(
      await screen.findByText(
        /The deterministic Reference replay does not claim live HydroCore-v5 DecisionCertificates/,
      ),
    ).toBeVisible();
    expect(screen.queryByRole('table')).toBeNull();
    expect(screen.queryByText('No decision certificates available for this incident.')).toBeNull();
  });

  test('LIVE with no certificates still shows the generic empty state, not the REFERENCE-specific one', async () => {
    const liveIncident: IncidentView = { ...demoIncident, mode: 'LIVE' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    renderWithQuery(<AuthorityWorkspace incident={liveIncident} />);
    expect(
      await screen.findByText('No decision certificates available for this incident.'),
    ).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// E. First launch
// ---------------------------------------------------------------------------
describe('E. First launch', () => {
  test('renders the final compact workflow-story sentence', () => {
    render(
      <FirstLaunchGateway
        onRunReference={() => {}}
        onRunLive={() => {}}
        onImportNetwork={() => {}}
        onExploreFallback={() => {}}
      />,
    );
    expect(
      screen.getByText(
        /Localize suspected sources, collect evidence selectively, verify response options with WNTR\/EPANET, and keep the final decision with a human operator\./,
      ),
    ).toBeVisible();
  });

  test('adds no new panel -- still exactly three primary actions plus the fallback link', () => {
    render(
      <FirstLaunchGateway
        onRunReference={() => {}}
        onRunLive={() => {}}
        onImportNetwork={() => {}}
        onExploreFallback={() => {}}
      />,
    );
    expect(screen.getAllByRole('button')).toHaveLength(4);
  });
});

// ---------------------------------------------------------------------------
// F. Stale assets
// ---------------------------------------------------------------------------
describe('F. Stale assets', () => {
  test('the published frontend no longer ships the stale learning-v1 model-governance.json asset', () => {
    const matches = import.meta.glob('../public/model-governance.json');
    expect(Object.keys(matches)).toHaveLength(0);
  });

  test('current frontend source does not reintroduce the historical HydroCore-S/M/L 96% governance table', () => {
    const matches = import.meta.glob('../src/components/ModelGovernanceTable.tsx');
    expect(Object.keys(matches)).toHaveLength(0);
  });
});
