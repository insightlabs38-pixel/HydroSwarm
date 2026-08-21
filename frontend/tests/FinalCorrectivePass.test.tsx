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
import regressionEvidenceArtifact from '../public/system-regression-evidence.json';
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
    expect(await screen.findByText('FUSED SOURCE BELIEF')).toBeVisible();
    expect(screen.queryByText('HYDROCORE-v5 SENTINEL')).toBeNull();
    expect(
      screen.getByText('Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.'),
    ).toBeVisible();
  });

  test('REFERENCE SourceWorkspace still says deterministic reference localization', async () => {
    const referenceIncident: IncidentView = { ...demoIncident, mode: 'REFERENCE' };
    renderWithQuery(<SourceWorkspace incident={referenceIncident} />);
    expect(await screen.findByText('DETERMINISTIC REFERENCE LOCALIZATION')).toBeVisible();
    expect(screen.queryByText('FUSED SOURCE BELIEF')).toBeNull();
    // No fusion-composition sentence in REFERENCE mode -- it does not fuse anything live.
    expect(
      screen.queryByText(
        'Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.',
      ),
    ).toBeNull();
  });

  test('CLASSICAL_SAFE SourceWorkspace says deterministic classical localization, not fused', async () => {
    const classicalSafeIncident: IncidentView = {
      ...demoIncident,
      mode: 'LIVE',
      runtimeAnalysisMode: 'CLASSICAL_SAFE',
    };
    renderWithQuery(<SourceWorkspace incident={classicalSafeIncident} />);
    expect(await screen.findByText('DETERMINISTIC CLASSICAL LOCALIZATION')).toBeVisible();
    expect(screen.queryByText('FUSED SOURCE BELIEF')).toBeNull();
    expect(screen.queryByText('CALIBRATED FUSION')).toBeNull();
    // No fabricated learned contribution -- Sentinel evidence did not run.
    expect(
      screen.queryByText(
        'Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.',
      ),
    ).toBeNull();
    expect(
      screen.getByText(
        /Learned localization was unavailable for this incident; deterministic classical/,
      ),
    ).toBeVisible();
  });

  test('normal hybrid/fused SourceWorkspace path shows the classical + Sentinel composition sentence', async () => {
    const hybridIncident: IncidentView = {
      ...demoIncident,
      mode: 'LIVE',
      runtimeAnalysisMode: 'FULL_HYBRID',
    };
    renderWithQuery(<SourceWorkspace incident={hybridIncident} />);
    expect(await screen.findByText('FUSED SOURCE BELIEF')).toBeVisible();
    expect(
      screen.getByText('Classical hydraulic/signature evidence + HydroCore-v5 Sentinel evidence.'),
    ).toBeVisible();
    expect(
      screen.queryByText(
        /Learned localization was unavailable for this incident; deterministic classical/,
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

  test('Overview Source card for CLASSICAL_SAFE incidents says deterministic classical localization, not fused', async () => {
    const classicalSafeIncident: IncidentView = {
      ...demoIncident,
      mode: 'LIVE',
      runtimeAnalysisMode: 'CLASSICAL_SAFE',
    };
    render(<Overview incident={classicalSafeIncident} />);
    expect(await screen.findByText('DETERMINISTIC CLASSICAL LOCALIZATION')).toBeVisible();
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
// B2. Benchmark artifact truthfulness (real committed artifact)
// ---------------------------------------------------------------------------
describe('B2. Benchmark artifact truthfulness', () => {
  const realDoc = regressionEvidenceArtifact;

  test('does not include historical/pre-final model-runtime rows', () => {
    const metricNames = realDoc.metrics.map((metric: { metric: string }) => metric.metric);
    expect(metricNames).not.toContain('Small-variant model runtime (reference graph, 128 nodes)');
    expect(metricNames).not.toContain('Small-variant model runtime (stress graph, 1000 nodes)');
    const values = realDoc.metrics.map((metric: { value: string }) => metric.value);
    expect(values).not.toContain('31.0 ms median / 32.3 ms p95');
    expect(values).not.toContain('104.3 ms median / 135.0 ms p95');
  });

  test('provenance no longer lists reports/results/performance.json', () => {
    const paths = realDoc.generatedFrom.map((entry: { path: string }) => entry.path);
    expect(paths).not.toContain('reports/results/performance.json');
  });

  test('WNTR gate labels use narrower reference/gate wording, not broad safe/unsafe language', () => {
    const metricNames = realDoc.metrics.map((metric: { metric: string }) => metric.metric);
    expect(metricNames).not.toContain('Unsafe plan rejection (WNTR)');
    expect(metricNames).not.toContain('Safe plan acceptance (WNTR)');
    expect(metricNames).toContain('Reference plan rejection gate (WNTR)');
    expect(metricNames).toContain('Reference plan acceptance gate (WNTR)');
  });

  test('deterministic regression evidence remains present', () => {
    const metricNames = realDoc.metrics.map((metric: { metric: string }) => metric.metric);
    expect(metricNames).toContain('Top-1 source localization (frozen golden fixture)');
    expect(metricNames).toContain('Hash-chain replay validity');
  });

  test('BenchmarkPage rendering the real artifact shows no stale model-runtime rows and shows the new gate labels', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => realDoc }));
    render(<BenchmarkPage />);
    expect(await screen.findByText('Reference plan rejection gate (WNTR)')).toBeVisible();
    expect(screen.getByText('Reference plan acceptance gate (WNTR)')).toBeVisible();
    expect(
      screen.queryByText('Small-variant model runtime (reference graph, 128 nodes)'),
    ).toBeNull();
    expect(screen.queryByText('Small-variant model runtime (stress graph, 1000 nodes)')).toBeNull();
    expect(screen.queryByText('31.0 ms median / 32.3 ms p95')).toBeNull();
    expect(screen.queryByText('104.3 ms median / 135.0 ms p95')).toBeNull();
    expect(screen.queryByText('reports/results/performance.json')).toBeNull();
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
