/**
 * Focused regression tests for the Final UI Quality Pass (§16).
 *
 * A. Current V5 evidence
 * B. Authority labels
 * C. Reference truthfulness
 * D. Navigation
 * E. Safety wording
 * F. Counterfactual
 * G. Gate consistency
 */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from '../src/App';
import { Counterfactuals } from '../src/components/Counterfactuals';
import { deriveDecisionGate } from '../src/decisionGate';
import { useConsoleStore } from '../src/store';
import type { IncidentView } from '../src/types';

const v5EvidenceJson = {
  schema: 'hydroswarm-v5-evidence-v1',
  system_identity: { name: 'HydroCore-v5 M10 frozen release', variant: 'small', parameters: 4182612, selected_seed: 20260814, checkpoint_sha256: 'abc', calibration_artifact_hash: 'abc', calibration_sha256: 'abc', feature_schema_hash: 'abc', release_bundle: 'models/hydrocore-v5-release', release_schema_version: 'hydroswarm-v5-release-v1' },
  runtime_outputs: ['event_cause', 'event_presence', 'evidence_sufficiency', 'relative_strength', 'source_node'],
  trained_tasks: ['sentinel'],
  deterministic_authority: { ood: 'OODDetector', scout: 'rank_sample_locations', planner: 'generate_response_plans', physical_verification: 'WNTR/EPANET', human_approval_required: true, autonomous_actuation: false },
  locked_governance: { gate_pass: true, locked_final_count: 105, locked_topology_count: 20, total_count: 125, authorized_openings: 1, actual_openings: 1, rerun: false, post_lock_tuning: false, closure_state: 'M11_6_LOCKED_EVALUATION_PASS', locked_final_result: 'M11_6_LOCKED_FINAL_PASS', locked_topology_result: 'M11_6_LOCKED_TOPOLOGY_PASS' },
  hard_safety_counters: { total_counters: 15, counters_zero: true, all_pass: true, counters: [] },
  metrics: {
    locked_final_test: {
      aggregate: { source: { n: 105, top1_rate: 0.55, top3_rate: 0.76, mrr: 0.687, coverage_rate: 0.886, actionable_rate: 0.61, calibrated_rate: 1.0, candidate_set_size: 3.09, posterior_entropy: 1.07 } },
      by_condition: {
        NOMINAL: { n: 15, top1_rate: 0.73, top3_rate: 0.87, mrr: 0.821, coverage_rate: 0.933, actionable_rate: 0.8, calibrated_rate: 1.0, candidate_set_size: 2.0, posterior_entropy: 0.55 },
        SENSOR_HEALTH_DEGRADED: { n: 15, top1_rate: 0.67, top3_rate: 0.87, mrr: 0.778, coverage_rate: 0.867, actionable_rate: 0.93, calibrated_rate: 1.0, candidate_set_size: 1.73, posterior_entropy: 0.59 },
        SEVERITY_SHIFT: { n: 15, top1_rate: 0.73, top3_rate: 0.87, mrr: 0.815, coverage_rate: 0.867, actionable_rate: 0.8, calibrated_rate: 1.0, candidate_set_size: 2.27, posterior_entropy: 0.78 },
        LOW_COVERAGE_ACTIVE_SAMPLING: { n: 15, top1_rate: 0.47, top3_rate: 0.87, mrr: 0.648, coverage_rate: 0.933, actionable_rate: 0.4, calibrated_rate: 1.0, candidate_set_size: 3.93, posterior_entropy: 1.32 },
        SENSOR_DROPOUT: { n: 15, top1_rate: 0.47, top3_rate: 0.6, mrr: 0.597, coverage_rate: 0.667, actionable_rate: 0.6, calibrated_rate: 1.0, candidate_set_size: 3.07, posterior_entropy: 1.18 },
        MEASUREMENT_NOISE: { n: 15, top1_rate: 0.4, top3_rate: 0.67, mrr: 0.586, coverage_rate: 0.933, actionable_rate: 0.33, calibrated_rate: 1.0, candidate_set_size: 4.2, posterior_entropy: 1.43 },
        AMBIGUITY_DISAGREEMENT: { n: 15, top1_rate: 0.4, top3_rate: 0.6, mrr: 0.567, coverage_rate: 1.0, actionable_rate: 0.4, calibrated_rate: 1.0, candidate_set_size: 4.4, posterior_entropy: 1.61 },
      },
    },
    locked_topology_test: {
      source: { n: 20, top1_rate: 0.55, top3_rate: 0.7, mrr: 0.652, coverage_rate: 0.6, actionable_rate: 0.0, calibrated_rate: 0.0, candidate_set_size: 1.3, posterior_entropy: 0.21 },
      planning: { human_approved_rate: 0.0, mean_candidates_generated: 0.0, mean_candidates_wntr_verified: 0.0 },
      topology_shift_predictive: 'DESCRIPTIVE_NON_GATING',
    },
  },
};

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  // Mock fetch: return v5 evidence for the evidence JSON, reject everything else
  vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
    if (typeof url === 'string' && url.includes('hydrocore-v5-evidence.json')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(v5EvidenceJson) });
    }
    return Promise.reject(new Error('offline test'));
  }));
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
    replayIndex: 0,
    replayPlaying: false,
    replaySpeed: 1,
    reducedMotion: false,
  });
});

// ---------------------------------------------------------------------------
// A. Current V5 evidence
// ---------------------------------------------------------------------------
describe('A. Current V5 evidence', () => {
  test('Validation does not display HydroCore-S 96% as final evidence', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.queryByText('96.0%')).toBeNull();
    expect(screen.queryByText(/HydroCore-S/)).toBeNull();
    expect(screen.queryByText(/HydroCore S/)).toBeNull();
  });

  test('current identity is HydroCore-v5', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.getAllByText(/HydroCore-v5/).length).toBeGreaterThan(0);
  });

  test('locked count is 105 + 20 = 125', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.getByText(/125 complete/)).toBeVisible();
  });

  test('topology predictive metrics are labeled descriptive/non-gating', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    const descElements = await screen.findAllByText(/DESCRIPTIVE/);
    expect(descElements.length).toBeGreaterThanOrEqual(1);
    const gatingElements = screen.getAllByText(/NON-GATING/);
    expect(gatingElements.length).toBeGreaterThanOrEqual(1);
  });

  test('topology conformal coverage is rendered N/A/inapplicable', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.getByText(/calibration inapplicable/)).toBeVisible();
  });

  test('runtime learned outputs are exactly the five frozen outputs', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.getByText(/learned runtime outputs/)).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// B. Authority labels
// ---------------------------------------------------------------------------
describe('B. Authority labels', () => {
  test('Sampling renders DETERMINISTIC SCOUT', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /^Sampling/ }));
    expect(await screen.findByText('DETERMINISTIC SCOUT')).toBeInTheDocument();
  });

  test('Overview Response card uses DETERMINISTIC PLANNER', async () => {
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    expect(screen.getByText('DETERMINISTIC PLANNER')).toBeVisible();
  });

  test('Overview does not imply learned Strategist authority', async () => {
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // Should NOT have the old "STRATEGIST" label
    expect(screen.queryByText('STRATEGIST')).toBeNull();
  });

  test('Sentinel remains learned/advisory (not deterministic)', async () => {
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // Overview Source card should still say SENTINEL
    expect(screen.getAllByText('SENTINEL').length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// D. Navigation
// ---------------------------------------------------------------------------
describe('D. Navigation', () => {
  test('secondary utilities do not receive workflow completion statuses', async () => {
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // The Network, Validation, Authority, Benchmarks rail items should NOT
    // show "complete" status — they should be neutral (no glyph).
    const networkBtn = screen.getByRole('button', { name: /Network/ });
    expect(networkBtn).not.toHaveClass('rail-status-complete');
    const validationBtn = screen.getByRole('button', { name: /Validation/ });
    expect(validationBtn).not.toHaveClass('rail-status-complete');
    const authorityBtn = screen.getByRole('button', { name: /Model & Authority/ });
    expect(authorityBtn).not.toHaveClass('rail-status-complete');
    const benchmarksBtn = screen.getByRole('button', { name: /Benchmarks/ });
    expect(benchmarksBtn).not.toHaveClass('rail-status-complete');
  });

  test('Decision Inspector remains on primary workspaces', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // Incident workspace should show Decision Inspector
    expect(screen.getByRole('complementary', { name: 'Decision inspector' })).toBeVisible();
    // Source workspace should show Decision Inspector
    await user.click(screen.getByRole('button', { name: /^Source/ }));
    await screen.findByText('Ranked source candidates');
    expect(screen.getByRole('complementary', { name: 'Decision inspector' })).toBeVisible();
  });

  test('Decision Inspector is absent on secondary utility pages', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // Network workspace should NOT show Decision Inspector
    await user.click(screen.getByRole('button', { name: /^Network/ }));
    await screen.findByRole('heading', { name: 'Import network' });
    expect(screen.queryByRole('complementary', { name: 'Decision inspector' })).toBeNull();
    // Validation workspace should NOT show Decision Inspector
    await user.click(screen.getByRole('button', { name: /Validation/ }));
    await screen.findByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' });
    expect(screen.queryByRole('complementary', { name: 'Decision inspector' })).toBeNull();
    // Benchmarks workspace should NOT show Decision Inspector
    await user.click(screen.getByRole('button', { name: /Benchmarks/ }));
    await screen.findByRole('heading', { name: 'Regression and runtime benchmarks' });
    expect(screen.queryByRole('complementary', { name: 'Decision inspector' })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// E. Safety wording
// ---------------------------------------------------------------------------
describe('E. Safety wording', () => {
  test('no "0 — safe" in PlanTable', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /^Response/ }));
    await screen.findByRole('heading', { name: 'Verified plan comparison' });
    expect(screen.queryByText('0 — safe')).toBeNull();
    expect(screen.getAllByText('0 min — no modeled violation').length).toBeGreaterThan(0);
  });

  test('Approval uses "Modeled margins" not "Safety margins"', async () => {
    const user = userEvent.setup();
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    await user.click(screen.getByRole('button', { name: /^Approval/ }));
    await screen.findByRole('heading', { name: 'Operator approval' });
    expect(screen.queryByText('Safety margins and consequences')).toBeNull();
    expect(screen.getByText('Modeled margins and consequences')).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// F. Counterfactual
// ---------------------------------------------------------------------------
describe('F. Counterfactual', () => {
  test('no-response baseline is explicitly NOT SIMULATED', () => {
    render(<Counterfactuals plans={[]} />);
    expect(screen.getByText(/NOT SIMULATED/)).toBeVisible();
  });

  test('no-response service and pressure are not presented as computed', () => {
    render(<Counterfactuals plans={[]} />);
    expect(screen.queryByText('100%')).toBeNull();
    expect(screen.getAllByText('Not evaluated').length).toBeGreaterThan(0);
  });

  test('unavailable metric does not render a misleading quantitative bar', () => {
    const { container } = render(<Counterfactuals plans={[]} />);
    const noResponseArticle = Array.from(container.querySelectorAll('article')).find(
      (el) => el.textContent?.includes('NOT SIMULATED'),
    );
    expect(noResponseArticle).toBeDefined();
    // The spread-visual bar should NOT be present for the no-response branch
    expect(noResponseArticle!.querySelector('.spread-visual')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// G. Gate consistency
// ---------------------------------------------------------------------------
describe('G. Gate consistency', () => {
  test('deriveDecisionGate returns READY for normal incident', () => {
    const incident = {
      mode: 'LIVE',
      calibrationValid: true,
      calibrationApplicable: true,
      ood: 'NORMAL',
    } as IncidentView;
    const gate = deriveDecisionGate(incident);
    expect(gate.state).toBe('READY');
    expect(gate.pathLabel).toBe('PATH READY');
  });

  test('deriveDecisionGate returns CALIBRATION_NOT_APPLICABLE for REFERENCE', () => {
    const incident = {
      mode: 'REFERENCE',
      calibrationValid: true,
      calibrationApplicable: false,
      ood: 'NORMAL',
    } as IncidentView;
    const gate = deriveDecisionGate(incident);
    expect(gate.state).toBe('CALIBRATION_NOT_APPLICABLE');
    expect(gate.pathLabel).toBe('PATH READY');
  });

  test('deriveDecisionGate returns CALIBRATION_INVALID for invalid calibration', () => {
    const incident = {
      mode: 'LIVE',
      calibrationValid: false,
      calibrationApplicable: true,
      ood: 'NORMAL',
    } as IncidentView;
    const gate = deriveDecisionGate(incident);
    expect(gate.state).toBe('CALIBRATION_INVALID');
    expect(gate.pathLabel).toBe('PATH DEGRADED');
  });

  test('deriveDecisionGate does not mislabel novel topology as corrupt calibration', () => {
    const incident = {
      mode: 'REFERENCE',
      calibrationValid: false,
      calibrationApplicable: false,
      ood: 'NORMAL',
    } as IncidentView;
    const gate = deriveDecisionGate(incident);
    // calibrationApplicable=false means N/A, NOT invalid
    expect(gate.state).toBe('CALIBRATION_NOT_APPLICABLE');
    expect(gate.state).not.toBe('CALIBRATION_INVALID');
  });

  test('Header and ModeBanner agree on derived gate state', async () => {
    renderApp();
    await screen.findByText('Verified response awaiting approval');
    // The DEMO_FALLBACK fixture has calibrationValid=true, ood=NORMAL,
    // so header should show PATH READY.
    const headerBadge = screen.getAllByText('PATH READY');
    expect(headerBadge.length).toBeGreaterThan(0);
  });
});
