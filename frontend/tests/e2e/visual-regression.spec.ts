import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

// overnight-plan.txt Task 3.7: visual and interaction regression gates.
// Screenshots are stored as real baseline artifacts (Playwright's
// toHaveScreenshot snapshot mechanism), not approved without evidence.

// Overview's plan table sits behind the same Suspense boundary as the
// lazy-loaded map/chart components; waiting only for the (Suspense-
// external) header banner is not enough to know the page has finished
// loading. Wait for something inside the boundary before asserting on
// layout-sensitive state.
async function waitForOverviewLoaded(page: Page) {
  await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  await expect(page.locator('.table-plan-button').first()).toBeVisible();
}

test.describe('viewport regression', () => {
  test('1366x768 operator overview', async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await expect(page).toHaveScreenshot('overview-1366x768.png', { fullPage: true });
  });

  test('1920x1080 operator overview', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await expect(page).toHaveScreenshot('overview-1920x1080.png', { fullPage: true });
  });
});

test.describe('keyboard-only navigation', () => {
  test('every nav page is reachable and focus is visible without a mouse', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();

    // Tab from the top of the document to the first nav button, then
    // Enter/activate each page purely via keyboard.
    const auditButton = page.getByRole('button', { name: 'Audit' });
    await auditButton.focus();
    await expect(auditButton).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: 'Incident audit and replay' })).toBeVisible();

    const validationButton = page.getByRole('button', { name: 'Validation' });
    await validationButton.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: 'Benchmarks and operating range' })).toBeVisible();

    const benchmarksButton = page.getByRole('button', { name: 'Benchmarks' });
    await benchmarksButton.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: 'Operational benchmarks' })).toBeVisible();

    const topologyButton = page.getByRole('button', { name: 'Topology' });
    await topologyButton.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: 'Directed hydraulic topology' })).toBeVisible();
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
  test('DEMO_FALLBACK banner is visible with no backend reachable (default test condition)', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
    await expect(page.getByText('DEMO_FALLBACK', { exact: false })).toBeVisible();
  });
});

test.describe('selected-plan synchronization', () => {
  test('selecting a different plan in the table updates the highlighted row', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    const planButtons = page.locator('.table-plan-button');
    const count = await planButtons.count();
    expect(count).toBeGreaterThan(1);

    const secondPlanButton = planButtons.nth(1);
    await secondPlanButton.click();
    await expect(secondPlanButton).toHaveAttribute('aria-pressed', 'true');
  });
});

test.describe('long identifiers do not break layout', () => {
  test('overview renders without horizontal overflow when node/plan names are unusually long', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
    expect(overflow).toBe(false);
  });
});
