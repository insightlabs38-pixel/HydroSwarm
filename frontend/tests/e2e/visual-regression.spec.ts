import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

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
  await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  await expect(page.locator('.map-canvas[role="img"]')).toBeVisible();
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
  await page.goto('/');
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
    { rail: /^Validation/, heading: 'Benchmarks and operating range' },
    { rail: /^Model/, heading: 'Authority ladder' },
    { rail: /^Benchmarks/, heading: 'Operational benchmarks' },
  ];

  test('every workflow-rail stage is reachable and focus is visible without a mouse', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();

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
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
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
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
    await expect(page.locator('.skip-link')).toHaveCount(1);
  });
});

test.describe('reduced-motion mode', () => {
  test('toggle is reachable, persists, and is announced via aria-pressed', async ({ page }) => {
    await page.goto('/');
    const toggle = page.getByRole('button', { name: /Reduced motion/ });
    await expect(toggle).toHaveAttribute('aria-pressed', 'false');
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText('Reduced motion on')).toBeVisible();
  });

  test('emulated prefers-reduced-motion still renders a usable page', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  });
});

test.describe('data-mode banners', () => {
  test('DEMO_FALLBACK banner is visible with no backend reachable (default test condition)', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
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
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
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
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();

    await page.getByRole('button', { name: 'C · Monitor + flush only' }).click();

    await expect(page.locator('.breadcrumb')).toHaveText('plan C');
    const inspector = page.getByRole('complementary', { name: 'Decision inspector' });
    await expect(inspector.getByText('C · Monitor + flush only')).toBeVisible();

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
    await page.goto('/');
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
  for (const width of [1300, 900, 600]) {
    test(`no horizontal page overflow at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto('/');
      await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
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
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
    const inspector = page.locator('.decision-inspector');
    await expect(inspector).toBeVisible();
    await expect(inspector).toHaveCSS('position', 'absolute');
  });
});
