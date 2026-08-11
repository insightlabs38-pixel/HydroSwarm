import { expect, test } from '@playwright/test';
import type { Page, TestInfo } from '@playwright/test';

const REFERENCE_ARTIFACT_PATH = new URL(
  '../../../artifacts/reference-demo/reference-incident-v1.json',
  import.meta.url,
).pathname;

async function mockReferenceArtifact(page: Page) {
  await page.route('**/api/reference-demo', (route) =>
    route.fulfill({ contentType: 'application/json', path: REFERENCE_ARTIFACT_PATH }),
  );
}

async function openReferenceAtMilestone(page: Page, milestone: number) {
  await mockReferenceArtifact(page);
  await page.goto('/?experience=reference');
  await expect(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY')).toBeVisible();
  await page.getByRole('button', { name: 'Pause' }).click();
  for (let index = 0; index < milestone; index += 1) {
    const next = page.getByRole('button', { name: 'Next', exact: true });
    if (await next.isEnabled()) {
      await next.click();
    } else {
      await page
        .locator('.mode-banner-controls button')
        .filter({ hasText: /^Replay / })
        .click();
    }
  }
}

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
}

async function expectWorkspaceUsesShellWidth(page: Page) {
  const [workspace, shellMain] = await Promise.all([
    page.locator('.workspace-content').boundingBox(),
    page.locator('.mission-shell-main').boundingBox(),
  ]);
  expect(workspace).not.toBeNull();
  expect(shellMain).not.toBeNull();
  expect(workspace!.width).toBeGreaterThanOrEqual(shellMain!.width * 0.85);
}

async function capture(page: Page, testInfo: TestInfo, name: string) {
  await page.screenshot({ path: testInfo.outputPath(name), fullPage: true });
}

async function expectApprovalComposition(
  page: Page,
  testInfo: TestInfo,
  width: number,
  height: number,
) {
  await page.setViewportSize({ width, height });
  await openReferenceAtMilestone(page, 9);
  await page.getByRole('button', { name: /^Approval/ }).click();

  const summary = page.locator('.approval-plan-summary');
  const decision = page.locator('.approval-primary-panel');
  const evidence = page.locator('.approval-evidence-stack');
  const referenceCallout = page.locator('.reference-approval-boundary');
  await expect(summary).toBeVisible();
  await expect(decision).toBeVisible();
  await expect(evidence).toBeVisible();
  await expect(page.getByRole('region', { name: 'Decision gate' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
    'Exact verification',
  );
  await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
    'Verification context',
  );
  await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
    'Infrastructure actuation',
  );
  await expect(page.getByRole('complementary', { name: 'Decision inspector' })).toContainText(
    'Exact verification',
  );
  await expect(page.getByRole('complementary', { name: 'Decision inspector' })).toContainText(
    'Infrastructure actuation',
  );
  await expect(page.getByRole('button', { name: 'Clear selection' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Replay operator approval' })).toHaveClass(
    /mode-banner-pause-action/,
  );
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();

  const decisionBox = await decision.boundingBox();
  const referenceCalloutBox = await referenceCallout.boundingBox();
  expect(decisionBox).not.toBeNull();
  expect(referenceCalloutBox).not.toBeNull();
  expect(decisionBox!.y).toBeLessThan(height);
  expect(referenceCalloutBox!.height).toBeLessThanOrEqual(decisionBox!.height * 0.45);

  const evidenceBoxes = await evidence.locator(':scope > .panel').evaluateAll((panels) =>
    panels.map((panel) => {
      const { y, height } = panel.getBoundingClientRect();
      return { y, height };
    }),
  );
  for (let index = 0; index < evidenceBoxes.length - 1; index += 1) {
    const gap = evidenceBoxes[index + 1].y - (evidenceBoxes[index].y + evidenceBoxes[index].height);
    expect(gap).toBeLessThanOrEqual(64);
  }

  await expectNoHorizontalOverflow(page);
  await expectWorkspaceUsesShellWidth(page);
  await capture(page, testInfo, `approval-${width}x${height}.png`);
}

test('REFERENCE Approval @ 1920x1080 has a full-width decision composition', async ({
  page,
}, testInfo) => {
  await expectApprovalComposition(page, testInfo, 1920, 1080);
});

test('REFERENCE Approval @ 1440x900 remains above fold', async ({ page }, testInfo) => {
  await expectApprovalComposition(page, testInfo, 1440, 900);
});

test('REFERENCE Approval @ 1366x768 remains compact and complete', async ({ page }, testInfo) => {
  await expectApprovalComposition(page, testInfo, 1366, 768);
});

test('REFERENCE Replay @ 1920x1080 presents ledger, diagnostics, and hash verification', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  // Replay becomes available only after the authored approval pause has
  // been replayed; do not bypass that workflow boundary in the smoke test.
  await openReferenceAtMilestone(page, 10);
  await page.getByRole('button', { name: /^Replay/ }).click();
  await expect(page.getByRole('heading', { name: 'Event ledger' })).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Failure-injection demonstration' }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Verify hash chain' })).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await expectWorkspaceUsesShellWidth(page);
  await capture(page, testInfo, 'replay-1920x1080.png');
});

test('REFERENCE sampling pause @ 1920x1080 keeps the authored action primary', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openReferenceAtMilestone(page, 3);
  await expect(page.getByRole('button', { name: 'Replay sample collection' })).toHaveClass(
    /mode-banner-pause-action/,
  );
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
  await expectNoHorizontalOverflow(page);
  await capture(page, testInfo, 'sampling-pause-1920x1080.png');
});

test('REFERENCE response verification @ 1920x1080 remains visible in the widened shell', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openReferenceAtMilestone(page, 8);
  await page.getByRole('button', { name: /^Response/ }).click();
  await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
  await expect(page.getByText('VERIFIED').first()).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await expectWorkspaceUsesShellWidth(page);
  await capture(page, testInfo, 'response-verification-1920x1080.png');
});

test('fallback Incident @ 1920x1080 uses the available mission-control pane', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('/?experience=fallback');
  await expect(page.locator('.map-canvas[role="img"]')).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await expectWorkspaceUsesShellWidth(page);
  await capture(page, testInfo, 'incident-1920x1080.png');
});
