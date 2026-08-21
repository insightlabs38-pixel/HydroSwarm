import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const REFERENCE_ARTIFACT_PATH = new URL(
  '../../../artifacts/reference-demo/reference-incident-v1.json',
  import.meta.url,
).pathname;

/**
 * Reference replay is an offline, checksummed artifact.  Route its one API
 * request here so these browser tests exercise the production mapper and UI
 * without requiring a separately-running API server.  This is deliberately
 * not used for LIVE: LIVE visual proof is captured against the real backend.
 */
async function mockReferenceArtifact(page: Page) {
  await page.route('**/api/reference-demo', (route) =>
    route.fulfill({ contentType: 'application/json', path: REFERENCE_ARTIFACT_PATH }),
  );
}

async function mockLiveExampleFlow(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    const json = (body: unknown) =>
      route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) });
    if (path.endsWith('/live-example-inputs'))
      return json({
        network_filename: 'reference.inp',
        network_inp_text: '[TITLE]',
        true_source: 'J8',
        candidate_nodes: ['J1', 'J8'],
        initial_observation: {
          sensor_id: 'S-J1',
          node_id: 'J1',
          concentration_mg_l: 0,
          pressure_m: 37,
        },
        candidate_signatures_mg_l: { J8: 1.2 },
        sample_time_seconds: 3600,
        contamination_threshold_mg_l: 0.001,
      });
    if (path.endsWith('/networks/import'))
      return json({
        network_id: 'network-live-1',
        name: 'reference',
        version: 1,
        sha256: 'network-hash',
        node_count: 2,
        link_count: 1,
        valid: true,
        validated_at: '2026-08-11T00:00:00Z',
        metadata: { nodes: [], links: [] },
        validation_errors: [],
      });
    if (path.endsWith('/incidents') && method === 'POST')
      return json({ incident_id: 'live-incident-12345678', status: 'SAMPLING' });
    if (path.endsWith('/samples/recommend'))
      return json({ node_id: 'J8', expected_information_gain: 1.2, alternatives: ['J1'] });
    if (path.endsWith('/plans/generate'))
      return json([
        { plan_id: 'unsafe', name: 'Close sole reservoir feeder' },
        { plan_id: 'safe', name: 'Flush downstream J8' },
      ]);
    if (path.endsWith('/verify'))
      return json({
        plan_id: path.includes('/safe/') ? 'safe' : 'unsafe',
        decision: path.includes('/safe/') ? 'VERIFIED' : 'REJECTED',
      });
    if (method === 'POST')
      return json({ incident_id: 'live-incident-12345678', status: 'SAMPLING' });
    return route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
  });
}

async function openReferenceAtMilestone(page: Page, milestone: number) {
  await mockReferenceArtifact(page);
  await page.goto('/?experience=reference');
  await expect(page.getByText('REFERENCE INCIDENT · CHECKSUMMED REPLAY')).toBeVisible();
  // Freeze auto-advance before moving to an exact authored milestone.
  await page.getByRole('button', { name: 'Pause' }).click();
  for (let index = 0; index < milestone; index += 1) {
    const next = page.getByRole('button', { name: 'Next', exact: true });
    if (await next.isEnabled()) {
      await next.click();
    } else {
      // Authored pause boundaries are deliberately non-bypassable. The
      // replay-specific action is the only control allowed to advance.
      await page
        .locator('.mode-banner-controls button')
        .filter({ hasText: /^Replay / })
        .click();
    }
  }
}

// overnight-plan.txt Task 3.7: visual and interaction regression gates.
// Screenshots are stored as real baseline artifacts (Playwright's
// toHaveScreenshot snapshot mechanism), not approved without evidence.

// Overview's map sits behind the same Suspense boundary as the other
// lazy-loaded map/chart components; waiting only for the (Suspense-
// external) header banner is not enough to know the page has finished
// loading. Wait for something inside the boundary before asserting on
// layout-sensitive state. "UI-10.5" 3 condensed Overview from an
// "everything dashboard" (which used to include the full plan table --
// moved to the Response workspace, its natural home) into a compact
// strip + dominant map + three small summary panels, so the map canvas
// is now the right thing to wait for here, not a plan-table row.
async function waitForOverviewLoaded(page: Page) {
  await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
  await expect(page.locator('.map-canvas[role="img"]')).toBeVisible();
}

async function expandTechnicalDock(page: Page) {
  const trigger = page.getByRole('button', { name: 'Expand technical dock' });
  if (await trigger.isVisible()) await trigger.click();
}

// One entry per required UI-11 baseline workspace at 1920x1080 (`rail`
// null for Incident, the default landing workspace -- every other stage
// is reached the same way the rest of this file navigates, via the real
// workflow rail, never a direct route/query hack).
const WORKSPACE_BASELINES: { name: string; rail: RegExp | null; heading: string | null }[] = [
  { name: 'incident', rail: null, heading: null },
  { name: 'source', rail: /^Source/, heading: 'Ranked source candidates' },
  { name: 'sampling', rail: /^Sampling/, heading: 'Evidence status' },
  { name: 'response', rail: /^Response/, heading: 'Verified plan comparison' },
  { name: 'approval', rail: /^Approval/, heading: 'Operator approval' },
  { name: 'replay', rail: /^Replay/, heading: 'Event ledger' },
];

async function gotoWorkspace(
  page: Page,
  width: number,
  height: number,
  target: { rail: RegExp | null; heading: string | null },
) {
  await page.setViewportSize({ width, height });
  await page.goto('/?experience=fallback');
  await waitForOverviewLoaded(page);
  if (target.rail) {
    await page.getByRole('button', { name: target.rail }).click();
    await expect(page.getByRole('heading', { name: target.heading! })).toBeVisible();
  }
  // Lets the map/chart Suspense boundary and any post-navigation ECharts
  // animation settle before the screenshot, on top of Playwright's own
  // "wait for fonts / stable screenshot" retries.
  await page.waitForTimeout(300);
}

test.describe('viewport regression', () => {
  for (const target of WORKSPACE_BASELINES) {
    test(`${target.name} @ 1920x1080`, async ({ page }) => {
      await gotoWorkspace(page, 1920, 1080, target);
      await expect(page).toHaveScreenshot(`${target.name}-1920x1080.png`, { fullPage: true });
    });
  }

  test('incident @ 1440x900', async ({ page }) => {
    await gotoWorkspace(page, 1440, 900, { rail: null, heading: null });
    await expect(page).toHaveScreenshot('incident-1440x900.png', { fullPage: true });
  });

  test('response @ 1440x900', async ({ page }) => {
    await gotoWorkspace(page, 1440, 900, {
      rail: /^Response/,
      heading: 'Verified plan comparison',
    });
    await expect(page).toHaveScreenshot('response-1440x900.png', { fullPage: true });
  });

  test('incident @ 1366x768 (compact desktop)', async ({ page }) => {
    await gotoWorkspace(page, 1366, 768, { rail: null, heading: null });
    await expect(page).toHaveScreenshot('incident-1366x768.png', { fullPage: true });
  });
});

// The primary judge flow is intentionally covered independently from the
// illustrative fallback baselines above.  Each snapshot is pinned to an
// authored milestone in the checksummed reference artifact, so a visual
// change cannot accidentally hide a safety boundary or make the progressive
// narrative look like a completed incident on first load.
test.describe('reference incident visual regression', () => {
  test('first-launch gateway @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /Local incident decision support/ }),
    ).toBeVisible();
    await expect(page).toHaveScreenshot('gateway-1920x1080.png', { fullPage: true });
  });

  test('sampling pause @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await openReferenceAtMilestone(page, 3);
    await expect(page.getByRole('button', { name: 'Replay sample collection' })).toBeVisible();
    await expect(page).toHaveScreenshot('reference-sampling-pause-1920x1080.png', {
      fullPage: true,
    });
  });

  test('posterior contraction @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await openReferenceAtMilestone(page, 5);
    await expect(page.getByText('Posterior contracts')).toBeVisible();
    await expect(page).toHaveScreenshot('reference-posterior-1920x1080.png', { fullPage: true });
  });

  test('verification @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await openReferenceAtMilestone(page, 8);
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
    await expect(page).toHaveScreenshot('reference-verification-1920x1080.png', { fullPage: true });
  });

  test('approval boundary @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await openReferenceAtMilestone(page, 9);
    await page.getByRole('button', { name: /^Approval/ }).click();
    await expect(page.getByRole('heading', { name: 'Operator approval' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Replay operator approval' })).toBeVisible();
    await expect(page).toHaveScreenshot('reference-approval-1920x1080.png', { fullPage: true });
  });

  test('authored reference pauses cannot be bypassed and replay actions advance one milestone', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openReferenceAtMilestone(page, 3);
    const sampleMilestone = await page.locator('.mode-banner-milestone').textContent();
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Replay sample collection' }).click();
    await expect(page.locator('.mode-banner-milestone')).not.toHaveText(sampleMilestone ?? '');

    await openReferenceAtMilestone(page, 9);
    const approvalMilestone = await page.locator('.mode-banner-milestone').textContent();
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Replay operator approval' }).click();
    await expect(page.locator('.mode-banner-milestone')).not.toHaveText(approvalMilestone ?? '');
  });

  test('LIVE flow starts in an explicitly live-computation state @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Keep the real API request pending only long enough to snapshot the
    // first truthful LIVE state.  The full LIVE workflow is exercised against
    // the production backend in Docker/native checks; this baseline prevents
    // a visual regression from relabeling this state as a replay or fallback.
    await page.route('**/api/**', () => new Promise(() => undefined));
    await page.goto('/?experience=live');
    await expect(page.getByText('LIVE COMPUTATION · REFERENCE INPUTS')).toBeVisible();
    await expect(page).toHaveScreenshot('live-computation-start-1920x1080.png', {
      fullPage: true,
    });
  });

  test('LIVE sample and approval pauses @ 1920x1080', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await mockLiveExampleFlow(page);
    await page.goto('/?experience=live');
    await expect(page.getByRole('button', { name: 'Collect reference sample' })).toBeVisible();
    await expect(page).toHaveScreenshot('live-sample-pause-1920x1080.png', { fullPage: true });
    await page.getByRole('button', { name: 'Collect reference sample' }).click();
    await expect(page.getByRole('button', { name: 'Approve plan' })).toBeVisible();
    await expect(page).toHaveScreenshot('live-approval-pause-1920x1080.png', { fullPage: true });
  });
});

test.describe('keyboard-only navigation', () => {
  // Every rail stage the mission-control shell ships (ui-work.txt §31
  // UI-0..UI-9), reachable purely via Tab-to-focus + Enter-to-activate,
  // each landing on real content -- not the pre-UI-1 flat page-tab nav
  // this test used to check (Audit/Topology, both unrouted since UI-1).
  const stages: { rail: RegExp; heading: string }[] = [
    { rail: /^Source/, heading: 'Ranked source candidates' },
    { rail: /^Sampling/, heading: 'Evidence status' },
    { rail: /^Response/, heading: 'Action sequence' },
    { rail: /^Approval/, heading: 'Operator approval' },
    { rail: /^Replay/, heading: 'Event ledger' },
    { rail: /^Network/, heading: 'Import network' },
    { rail: /^Validation/, heading: 'HydroCore-v5 final evaluation evidence' },
    { rail: /^Model/, heading: 'Decision authority path' },
    { rail: /^Benchmarks/, heading: 'Regression and runtime benchmarks' },
  ];

  test('every workflow-rail stage is reachable and focus is visible without a mouse', async ({
    page,
  }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();

    for (const { rail, heading } of stages) {
      const railButton = page.getByRole('button', { name: rail });
      await railButton.focus();
      await expect(railButton).toBeFocused();
      await page.keyboard.press('Enter');
      await expect(page.getByRole('heading', { name: heading })).toBeVisible();
    }
  });

  // ui-work.txt §24: the skip link must move real keyboard focus to
  // main content, not just scroll it into view -- a UI-10 fix (the
  // `<main>` previously had no `tabIndex`, so a real browser had
  // nothing to focus even though the link itself worked). jsdom cannot
  // exercise a real browser's native fragment-focus behavior, so this
  // is only verifiable end-to-end.
  test('skip link moves real keyboard focus to main content', async ({ page }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    const skipLink = page.getByRole('link', { name: 'Skip to main content' });
    await skipLink.focus();
    await expect(skipLink).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.locator('#main-content')).toBeFocused();
  });

  // UI-11 hardening find: index.html carried its own static skip-link
  // ("Skip to incident workspace") from the pre-mission-control dashboard,
  // never removed once App.tsx grew a real one ("Skip to main content")
  // in UI-1. Both existed simultaneously -- the static one, outside
  // React's #root, was first in DOM/tab order and had no working focus
  // target of its own. Locks down there is exactly one skip link now.
  test('exactly one skip link exists (no stale static duplicate)', async ({ page }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    await expect(page.locator('.skip-link')).toHaveCount(1);
  });
});

test.describe('reduced-motion mode', () => {
  test('toggle is reachable, persists, and is announced via aria-pressed', async ({ page }) => {
    await page.goto('/?experience=fallback');
    const toggle = page.getByRole('button', { name: /Reduced motion/ });
    await expect(toggle).toHaveAttribute('aria-pressed', 'false');
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText('Reduced motion on')).toBeVisible();
  });

  test('emulated prefers-reduced-motion still renders a usable page', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
  });
});

test.describe('data-mode banners', () => {
  test('DEMO_FALLBACK banner is visible with no backend reachable (default test condition)', async ({
    page,
  }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    // Both the header's mode badge and the decision inspector's own
    // status badge render the real DEMO_FALLBACK mode -- intentional
    // duplication (ui-work.txt §13: the inspector always echoes the
    // current mode), so assert the header instance specifically rather
    // than an ambiguous page-wide text match.
    await expect(page.getByLabel('Data mode DEMO_FALLBACK')).toBeVisible();
  });

  test('failure injection (Task 3.8) renders ERROR mode with a Retry action, never a false LIVE state', async ({
    page,
  }) => {
    await page.goto('/?failure=no_valid_plan');
    await expect(page.getByText('INCIDENT UNAVAILABLE')).toBeVisible();
    await expect(page.getByText(/no_valid_plan/)).toBeVisible();
    await expect(page.getByText(/abstention, not forcing a plan through/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible();
    await expect(page.getByText('LIVE', { exact: true })).toHaveCount(0);
  });
});

test.describe('selected-plan synchronization', () => {
  // The plan comparison table moved from Overview to the Response
  // workspace in "UI-10.5" 2 (its own dedicated home, alongside the
  // Pareto frontier and full action sequence it was previously
  // duplicated against).
  test('selecting a different plan in the table updates the highlighted row', async ({ page }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();

    const planButtons = page.locator('.plan-table .table-plan-button');
    const count = await planButtons.count();
    expect(count).toBeGreaterThan(1);

    const secondPlanButton = planButtons.nth(1);
    await secondPlanButton.click();
    await expect(secondPlanButton).toHaveAttribute('aria-pressed', 'true');
  });

  // ui-work.txt §22: selecting a plan must update the map overlay
  // (verified indirectly via the toolbar breadcrumb, which is driven by
  // the same selectedPlanId), the Decision Inspector, and the technical
  // dock together -- not just the table's own highlighted row. Plan C
  // (not the default-recommended plan B) makes the change observable.
  test('selecting plan C updates the toolbar breadcrumb, inspector, and verification dock together', async ({
    page,
  }) => {
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();

    await page
      .locator('.plan-table')
      .getByRole('button', { name: 'C · Monitor + flush only' })
      .click();

    await expect(page.locator('.breadcrumb')).toHaveText('plan C');
    const inspector = page.getByRole('complementary', { name: 'Decision inspector' });
    await expect(inspector.getByText('C · Monitor + flush only')).toBeVisible();

    await expandTechnicalDock(page);
    await page.getByRole('tab', { name: 'Verification' }).click();
    await expect(
      page.locator('#dock-panel-verification').getByText('C · Monitor + flush only'),
    ).toBeVisible();
  });
});

test.describe('long identifiers do not break layout', () => {
  test('overview renders without horizontal overflow when node/plan names are unusually long', async ({
    page,
  }) => {
    await page.goto('/?experience=fallback');
    await waitForOverviewLoaded(page);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(overflow).toBe(false);
  });
});

test.describe('responsive layout (ui-work.txt §25)', () => {
  // UI-10 found and fixed a real horizontal page-overflow bug at narrow
  // widths: the mission header's status badge row didn't wrap or shrink,
  // forcing the whole document wider than the viewport. Locks that fix
  // down at each documented breakpoint tier rather than only the
  // default desktop width the other tests use.
  for (const width of [1366, 1100, 1099, 900, 768, 767]) {
    test(`no horizontal page overflow at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/?experience=fallback');
      await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
      const overflow = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth);
    });
  }

  test('decision inspector becomes an overlay drawer at the 768-1099px tablet tier', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 900, height: 900 });
    await page.goto('/?experience=fallback');
    await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
    const inspector = page.locator('.decision-inspector');
    await expect(inspector).toBeVisible();
    await expect(inspector).toHaveCSS('position', 'absolute');
  });

  test('compact desktop rail remains labeled until its explicit collapse control is used', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await page.goto('/?experience=fallback');
    await waitForOverviewLoaded(page);
    const rail = page.locator('.workflow-rail');
    const toggle = page.getByRole('button', { name: /Collapse workflow/ });
    await expect(rail.getByText('Approval', { exact: true })).toBeVisible();
    await expect(toggle).toHaveAttribute('aria-pressed', 'false');
    await toggle.click();
    await expect(rail).toHaveClass(/collapsed/);
    await expect(page.getByRole('button', { name: /Expand workflow/ })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  test('tablet rail state, control semantics, and layout agree at 900px', async ({ page }) => {
    await page.setViewportSize({ width: 900, height: 900 });
    await page.goto('/?experience=fallback');
    await waitForOverviewLoaded(page);

    const rail = page.locator('.workflow-rail');
    const collapse = page.getByRole('button', { name: /Collapse workflow/ });
    await expect(rail.getByText('Approval', { exact: true })).toBeVisible();
    await expect(collapse).toHaveAttribute('aria-pressed', 'false');

    await collapse.click();
    await expect(rail).toHaveClass(/collapsed/);
    const expand = page.getByRole('button', { name: /Expand workflow/ });
    await expect(expand).toHaveAttribute('aria-pressed', 'true');

    await expand.click();
    await expect(rail).not.toHaveClass(/collapsed/);
    await expect(rail.getByText('Approval', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: /Collapse workflow/ })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
    const dimensions = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  });

  test('approval has no map controls and navigation controls stay inside map bounds', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/?experience=fallback');
    await waitForOverviewLoaded(page);
    const mapBox = await page.locator('.map-shell').first().boundingBox();
    const navBox = await page.locator('.maplibregl-ctrl-top-right').boundingBox();
    expect(mapBox).not.toBeNull();
    expect(navBox).not.toBeNull();
    expect(navBox!.x).toBeGreaterThanOrEqual(mapBox!.x);
    expect(navBox!.y).toBeGreaterThanOrEqual(mapBox!.y);
    expect(navBox!.x + navBox!.width).toBeLessThanOrEqual(mapBox!.x + mapBox!.width);
    expect(navBox!.y + navBox!.height).toBeLessThanOrEqual(mapBox!.y + mapBox!.height);

    await page.getByRole('button', { name: /^Approval/ }).click();
    await expect(page.getByRole('heading', { name: 'Operator approval' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Fit network' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Layers' })).toHaveCount(0);
  });
});

test.describe('final visual coverage matrix', () => {
  test('gateway @ 1440x900 and 1366x768', async ({ page }) => {
    for (const [width, height] of [
      [1440, 900],
      [1366, 768],
    ] as const) {
      await page.setViewportSize({ width, height });
      await page.goto('/');
      await expect(
        page.getByRole('heading', { name: /Local incident decision support/ }),
      ).toBeVisible();
      await expect(page).toHaveScreenshot(`gateway-${width}x${height}.png`, { fullPage: true });
    }
  });

  test('Approval @ 1440x900 and 1366x768', async ({ page }) => {
    for (const [width, height] of [
      [1440, 900],
      [1366, 768],
    ] as const) {
      await gotoWorkspace(page, width, height, { rail: /^Approval/, heading: 'Operator approval' });
      await expect(page).toHaveScreenshot(`approval-${width}x${height}.png`, { fullPage: true });
    }
  });

  test('Sampling map workspace @ 1366x768', async ({ page }) => {
    await gotoWorkspace(page, 1366, 768, { rail: /^Sampling/, heading: 'Evidence status' });
    await expect(page).toHaveScreenshot('sampling-1366x768.png', { fullPage: true });
  });

  test('reference decision states @ 1440x900', async ({ page }) => {
    for (const [milestone, name, rail, heading] of [
      [3, 'reference-sampling-pause-1440x900.png', null, null],
      [8, 'reference-verification-1440x900.png', /^Response/, 'Verified plan comparison'],
      [9, 'reference-approval-1440x900.png', /^Approval/, 'Operator approval'],
    ] as const) {
      await page.setViewportSize({ width: 1440, height: 900 });
      await openReferenceAtMilestone(page, milestone);
      if (rail) await page.getByRole('button', { name: rail }).click();
      if (heading) await expect(page.getByRole('heading', { name: heading })).toBeVisible();
      await expect(page).toHaveScreenshot(name, { fullPage: true });
    }
  });

  test('reference approval boundary @ 1366x768 remains legible and non-bypassable', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await openReferenceAtMilestone(page, 9);
    await page.getByRole('button', { name: /^Approval/ }).click();
    await expect(page.getByRole('heading', { name: 'Operator approval' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Replay operator approval' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    await expect(
      page.locator('.workflow-rail').getByText('Approval', { exact: true }),
    ).toBeVisible();
    const dimensions = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
    await expect(page).toHaveScreenshot('reference-approval-1366x768.png', { fullPage: true });
  });

  test('utility workspaces @ 1440x900', async ({ page }) => {
    for (const [rail, heading, name] of [
      [/^Network/, 'Import network', 'network-1440x900.png'],
      [/^Validation/, 'HydroCore-v5 final evaluation evidence', 'validation-1440x900.png'],
      [/^Model/, 'Decision authority path', 'authority-1440x900.png'],
      [/^Benchmarks/, 'Regression and runtime benchmarks', 'benchmarks-1440x900.png'],
    ] as const) {
      await gotoWorkspace(page, 1440, 900, { rail, heading });
      if (name === 'benchmarks-1440x900.png') {
        await expect(page.getByText('Measured frozen WNTR regression')).toBeVisible();
      }
      await expect(page).toHaveScreenshot(name, { fullPage: true });
    }
  });
});
