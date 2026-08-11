import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function expandTechnicalDock(page: Page) {
  const trigger = page.getByRole('button', { name: 'Expand technical dock' });
  if (await trigger.isVisible()) await trigger.click();
}

// ui-work.txt §32 "Normal demo flow" -- the product's own definition of a
// judge-mode-free walkthrough an operator (or a demo audience) should be
// able to follow without reading long paragraphs. This test locks that
// exact sequence down against the real mission-control shell in a real
// browser (DEMO_FALLBACK, since no backend is reachable from this
// environment) -- deliberately not a jsdom re-run of App.test.tsx's
// "30-second comprehension test", which only exercises the Overview
// workspace and can't see real layout, real async map/chart rendering,
// or real cross-workspace navigation the way this can.
test('operator can follow the full normal demo flow (ui-work.txt §32) using only the real UI', async ({
  page,
}) => {
  // 1. Open HydroSwarm; 2. see local/offline/readiness status.
  await page.goto('/?experience=fallback');
  await expect(page.getByText('OFFLINE · LOCAL')).toBeVisible();
  await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
  await expect(page.getByText('READY')).toBeVisible();

  // 3. Open incident; 4. map establishes network context.
  await expect(
    page.getByRole('heading', { name: 'Verified response awaiting approval' }),
  ).toBeVisible();
  await expect(page.locator('.map-canvas[role="img"]')).toBeVisible();

  // 5. Show source candidates (Overview's own compact summary panel --
  // "UI-10.5" 3.C condensed the old "Source candidates" panel to "Source").
  const sourcePanel = page.getByRole('region', { name: 'Source', exact: true });
  await expect(sourcePanel).toBeVisible();
  await expect(sourcePanel.getByText('J117')).toBeVisible();

  // 6. Show calibrated candidate-set status/uncertainty -- moved from a
  // Source-workspace-local sidebar into the global Decision Inspector in
  // "UI-10.5" 2, which is now the single primary right-side decision pane.
  await page.getByRole('button', { name: /^Source/ }).click();
  await expect(page.getByRole('heading', { name: 'Ranked source candidates' })).toBeVisible();
  const inspector = page.getByRole('complementary', { name: 'Decision inspector' });
  await expect(inspector).toBeVisible();
  await expect(inspector.getByText('Conformal target', { exact: true })).toBeVisible();
  await expect(inspector.getByText('Held-out measured coverage')).toBeVisible();

  // 7. Show why evidence is sufficient or where to sample (Sampling workspace).
  await page.getByRole('button', { name: /^Sampling/ }).click();
  await expect(page.getByRole('heading', { name: 'Evidence status' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Next sample recommendation' })).toBeVisible();

  // 8. Select response plan; 9. compare exact WNTR/EPANET consequences;
  // 10. show a rejected alternative and reason.
  await page.getByRole('button', { name: /^Response/ }).click();
  await expect(
    page.getByRole('heading', { name: 'Verified response Pareto frontier' }),
  ).toBeVisible();
  const planButtons = page.locator('.table-plan-button');
  await expect(planButtons.first()).toBeVisible();
  await planButtons.first().click();
  await expect(page.getByText('REJECTED').first()).toBeVisible();
  await expect(
    page.getByText(/4 nodes with pressure below the 15 m minimum/i).first(),
  ).toBeVisible();

  // 11. Map updates with actions -- the toolbar breadcrumb reflects the
  // selection made in the plan table above (ui-work.txt §22 cross-panel
  // synchronization).
  await expect(page.getByText(/^plan /)).toBeVisible();

  // 12. Show CURRENT verification/context (technical dock).
  await expandTechnicalDock(page);
  await page.getByRole('tab', { name: 'Verification' }).click();
  await expect(page.getByText('Verification status')).toBeVisible();

  // 13. Open dock and show provenance.
  await page.getByRole('tab', { name: 'Provenance' }).click();
  await expect(page.getByText('Network hash')).toBeVisible();
  await expect(page.getByText('Model checkpoint hash')).toBeVisible();

  // 14. Show human approval boundary (Approval workspace).
  await page.getByRole('button', { name: /^Approval/ }).click();
  await expect(page.getByRole('heading', { name: 'Operator approval' })).toBeVisible();
  await expect(page.locator('.approval-hierarchy')).toBeVisible();
  await expect(page.getByText('HUMAN APPROVED').first()).toBeVisible();

  // 15. Open Replay/Audit and show a deterministic trace.
  await page.getByRole('button', { name: /^Replay/ }).click();
  await expect(page.getByRole('heading', { name: 'Event ledger' })).toBeVisible();
  await expect(page.getByLabel('Event ledger').getByText('INCIDENT DETECTED')).toBeVisible();
});

// UI-11 required scenario: OOD / suppressed planning. Uses the
// `?demo=ood_suppressed` deterministic variant (api/incident.ts) so this
// is reproducible without a live backend -- the governed suppression UI
// itself (ModeBanner, MissionHeader OOD badge, WorkflowRail caution
// status) was already wired in earlier phases; this locks down that a
// real out-of-range incident actually reaches it end-to-end.
test('OOD / suppressed planning shows the governed caution state, not a fabricated plan', async ({
  page,
}) => {
  await page.goto('/?demo=ood_suppressed');
  // DEMO_FALLBACK's own banner takes priority over the OUTSIDE_VALIDATED_RANGE
  // mode banner (ModeBanner.tsx: "REPLAY/DEMO_FALLBACK/ERROR take priority"),
  // so the always-rendered header badges are the real signal here, not the
  // narrow banner strip.
  await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();
  await expect(page.getByText('OOD OUTSIDE_VALIDATED_RANGE')).toBeVisible();
  await expect(page.getByText('DEGRADED')).toBeVisible();

  await page.getByRole('button', { name: /^Response/ }).click();
  // UI-11.1 §3: while planning is suppressed, no panel may draw content
  // that only makes sense once plans exist -- neither the plan comparison
  // table (previously correct) nor the Pareto frontier panel below it
  // (previously a bug: it kept showing demoParetoFrontier, an illustrative
  // dataset authored for a populated scenario, even though this incident's
  // real plans are empty). Both must show the same governed suppressed
  // state instead.
  await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
  await expect(page.locator('.plan-table')).toHaveCount(0);
  await expect(page.locator('.plan-table .table-plan-button')).toHaveCount(0);
  await expect(
    page.getByText('Planning suppressed -- no response plans to compare.'),
  ).toBeVisible();

  await expect(
    page.getByRole('heading', { name: 'Verified response Pareto frontier' }),
  ).toBeVisible();
  await expect(
    page.getByText('Planning suppressed -- no Pareto frontier to display.'),
  ).toBeVisible();
  // Neither the exposure-aware scatter chart nor either frontier table --
  // which would carry the illustrative demoParetoFrontier plan labels
  // ("Aggressive isolation", "Isolate + controlled flush", etc.) -- may
  // render while planning is suppressed.
  await expect(page.locator('.frontier-chart')).toHaveCount(0);
  await expect(page.getByText('Aggressive isolation')).toHaveCount(0);

  await expect(page.getByRole('heading', { name: 'Compare plans' })).toBeVisible();
  await expect(
    page.getByText('Planning suppressed -- no plan comparison available.'),
  ).toBeVisible();
});

// UI-11 required scenario: stale verification disables approval. Uses
// the `?demo=stale_verification` deterministic variant -- the guarded
// gate itself (ApprovalWorkspace's canReview = isVerified && isCurrent)
// predates this test and is never touched here; this only proves a
// real STALE plan reaches the UI and the gate actually holds.
test('stale verification blocks the approval hierarchy from advancing past simulator-verified', async ({
  page,
}) => {
  await page.goto('/?demo=stale_verification');
  await expect(page.getByText('ILLUSTRATIVE DEMO / DEMO_FALLBACK')).toBeVisible();

  await page.getByRole('button', { name: /^Approval/ }).click();
  await expect(page.getByRole('heading', { name: 'Operator approval' })).toBeVisible();
  await expect(page.getByText('Verification is stale.')).toBeVisible();
  await expect(page.getByText(/Re-verify before approval/i)).toBeVisible();

  const hierarchy = page.getByRole('list', { name: 'Approval authority hierarchy' });
  await expect(hierarchy).toBeVisible();
  await expect(hierarchy.getByText('Simulator verified')).toHaveClass(/reached/);
  await expect(hierarchy.getByText('Current context')).not.toHaveClass(/reached/);
  await expect(hierarchy.getByText('Human approved')).not.toHaveClass(/reached/);
});
