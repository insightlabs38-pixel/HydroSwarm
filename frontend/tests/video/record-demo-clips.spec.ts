import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

/**
 * Dedicated final-presentation recording flow -- NOT part of the regular
 * E2E/visual-regression suite (see ../e2e/*.spec.ts, which stays
 * untouched). Produces the 9 clean demo source clips for the final
 * video, captured against a real running HydroSwarm v0.2.1 release
 * instance (see global-setup.ts for the preflight that enforces this --
 * no mocked `/api/**` routes are used anywhere in this file).
 *
 * Every interaction here is deterministic (authored reference milestones,
 * real buttons) or a real backend call (Live Example) -- nothing is a
 * fabricated wait for a fabricated outcome. Holds are plain
 * `waitForTimeout` calls used only to leave enough clean footage at an
 * important state for later editing, per the recording brief.
 */

const RAW_DIR = new URL('../../../artifacts/video-recording/raw', import.meta.url).pathname;

function ffmpegAvailable(): boolean {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/** Closes the page (finalizing its video), saves the WebM under its clip
 * name, and -- if ffmpeg is on PATH -- transcodes it to a clean MP4
 * alongside it. The WebM is kept either way as the original capture. */
async function finalizeClip(page: Page, name: string): Promise<void> {
  const video = page.video();
  if (!video) {
    throw new Error(`Video recording was not enabled for clip "${name}".`);
  }
  await page.close();
  fs.mkdirSync(RAW_DIR, { recursive: true });
  const webmPath = path.join(RAW_DIR, `${name}.webm`);
  await video.saveAs(webmPath);

  if (!ffmpegAvailable()) {
    console.warn(`[record-demo-clips] ffmpeg not found -- "${name}" left as WebM only.`);
    return;
  }
  const mp4Path = path.join(RAW_DIR, `${name}.mp4`);
  execFileSync(
    'ffmpeg',
    [
      '-y',
      '-i',
      webmPath,
      '-c:v',
      'libx264',
      '-preset',
      'medium',
      '-crf',
      '18',
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      mp4Path,
    ],
    { stdio: 'inherit' },
  );
}

/** Opens the real Reference Incident (real checksummed artifact served by
 * the real backend -- no route mocking) and steps it to an authored
 * milestone using only the product's own Pause/Next/"Replay ..." controls. */
async function openReferenceAtMilestone(page: Page, milestone: number): Promise<void> {
  await page.goto('/?experience=reference');
  await expect(page.getByText('REFERENCE INCIDENT · CHECKSUMMED REPLAY')).toBeVisible();
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

test.describe('final-presentation demo clips', () => {
  test('01-gateway', async ({ page }) => {
    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: /Local incident decision support/i }),
    ).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run Reference Incident' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run Live Example' })).toBeVisible();
    await page.waitForTimeout(4000);
    await finalizeClip(page, '01-gateway');
  });

  test('02-reference-source', async ({ page }) => {
    // Milestone 2 (evidence_insufficient): broad, unnarrowed candidate
    // region -- the early source-localization/uncertainty state.
    await openReferenceAtMilestone(page, 2);
    await page.getByRole('button', { name: /^Source/ }).click();
    await expect(page.getByRole('heading', { name: 'Ranked source candidates' })).toBeVisible();
    await expect(page.locator('.map-canvas[role="img"]')).toBeVisible();
    const inspector = page.getByRole('complementary', { name: 'Decision inspector' });
    await expect(inspector).toBeVisible();
    await page.waitForTimeout(5000);
    await finalizeClip(page, '02-reference-source');
  });

  test('03-reference-sampling', async ({ page }) => {
    // Milestone 3 (sample_recommended): the authored evidence-insufficient
    // pause where ordinary Next is disabled and only the authored sample
    // collection control can advance the replay.
    await openReferenceAtMilestone(page, 3);
    const collect = page.getByRole('button', { name: 'Replay sample collection' });
    await expect(collect).toHaveClass(/mode-banner-pause-action/);
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    await page.waitForTimeout(4000);
    await collect.click();
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeEnabled();
    await page.waitForTimeout(3000);
    await finalizeClip(page, '03-reference-sampling');
  });

  test('04-reference-posterior', async ({ page }) => {
    // Milestone 4 (sample_received, region still broad) -> 5
    // (posterior_contracted, region narrows to the true source) --
    // captures the actual before/after contraction, not just an end state.
    await openReferenceAtMilestone(page, 4);
    await page.getByRole('button', { name: /^Source/ }).click();
    await expect(page.getByRole('heading', { name: 'Ranked source candidates' })).toBeVisible();
    await page.waitForTimeout(2500);
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    await page.waitForTimeout(6000);
    await finalizeClip(page, '04-reference-posterior');
  });

  test('05-reference-response-rejected-verified', async ({ page }) => {
    // Milestone 8 (safe_plan_verified): both the rejected candidate and
    // the verified alternative are present in the plan comparison.
    await openReferenceAtMilestone(page, 8);
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
    await expect(page.getByText('REJECTED').first()).toBeVisible();
    await page.waitForTimeout(5000);
    // Select the rejected plan first to surface its real rejection reason
    // (pressure-constraint explanation), then the verified one.
    const rejectedRow = page.locator('tr').filter({ hasText: 'REJECTED' }).first();
    await rejectedRow.locator('.table-plan-button').click();
    await expect(page.getByText('REJECTED').first()).toBeVisible();
    await page.waitForTimeout(4000);
    await expect(page.getByText('VERIFIED').first()).toBeVisible();
    await page.waitForTimeout(4000);
    await finalizeClip(page, '05-reference-response-rejected-verified');
  });

  test('06-reference-approval', async ({ page }) => {
    // Milestone 9 (human_approval_boundary): hold on the pre-approval
    // boundary first, then trigger the legitimate reference
    // operator-approval action and capture the resulting transition.
    await openReferenceAtMilestone(page, 9);
    await page.getByRole('button', { name: /^Approval/ }).click();
    await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
      'Exact verification',
    );
    await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
      'Verification context',
    );
    await expect(page.getByRole('region', { name: 'Decision gate' })).toContainText(
      'Infrastructure actuation',
    );
    const approve = page.getByRole('button', { name: 'Replay operator approval' });
    await expect(approve).toHaveClass(/mode-banner-pause-action/);
    await page.waitForTimeout(5000);
    await approve.click();
    await page.waitForTimeout(4000);
    await finalizeClip(page, '06-reference-approval');
  });

  test('07-technical-provenance', async ({ page }) => {
    await openReferenceAtMilestone(page, 8);
    await page.getByRole('button', { name: /^Response/ }).click();
    await expect(page.getByRole('heading', { name: 'Verified plan comparison' })).toBeVisible();
    const dockTrigger = page.getByRole('button', { name: 'Expand technical dock' });
    if (await dockTrigger.isVisible()) await dockTrigger.click();
    await page.getByRole('tab', { name: 'Verification' }).click();
    await expect(page.getByText('Verification status')).toBeVisible();
    await page.waitForTimeout(3500);
    await page.getByRole('tab', { name: 'Provenance' }).click();
    await expect(page.getByText('Network hash')).toBeVisible();
    await expect(page.getByText('Model checkpoint hash')).toBeVisible();
    await page.waitForTimeout(3500);
    await finalizeClip(page, '07-technical-provenance');
  });

  test('08-live-v5', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Run Live Example' }).click();
    await expect(page.getByText('LIVE COMPUTATION · REFERENCE INPUTS')).toBeVisible();
    // Real, sequential production-pipeline stages against the real
    // released HydroCore-v5 runtime and real WNTR/EPANET path -- no
    // mocked Live API. The real deterministic evidence/sampling policy
    // decides for itself, from the real Evidence Certificate, whether
    // another sample is worth collecting -- this harness captures
    // whichever real, governed outcome it reaches instead of forcing a
    // particular branch. A governed stop here (planning not currently
    // permitted) is a legitimate, honest outcome, not an application
    // failure, and must never be forced past into a fabricated sample
    // or plan.
    const samplingReady = page.getByText('Real sampling recommendation ready');
    const governedStop = page.getByText('Planning is not currently permitted for this incident');
    await expect(samplingReady.or(governedStop)).toBeVisible({ timeout: 60_000 });

    if (await samplingReady.isVisible()) {
      await page.waitForTimeout(3000);
      await page.getByRole('button', { name: 'Collect reference sample' }).click();
      await expect(page.getByText('Verified plan ready:')).toBeVisible({ timeout: 60_000 });
      await page.waitForTimeout(6000);
    } else {
      await page.waitForTimeout(8000);
    }
    await finalizeClip(page, '08-live-v5');
  });

  // v2: captures the real LIVE progression itself (not just the final
  // state) so a viewer can see HydroSwarm actually executing its current
  // governed policy against the real published release -- real network
  // import, real incident creation, real HydroCore-v5 analysis, real
  // Evidence Certificate, then whichever real governed outcome results.
  // No artificial per-stage delays: fast real stages stay fast. Kept as
  // an additional clip alongside '08-live-v5' (not a replacement).
  test('08-live-v5-v2', async ({ page }) => {
    await page.goto('/?experience=live');
    await expect(page.getByText('LIVE COMPUTATION · REFERENCE INPUTS')).toBeVisible();
    await page.waitForTimeout(3000);

    const samplingReady = page.getByText('Real sampling recommendation ready');
    const governedStop = page.getByText('Planning is not currently permitted for this incident');
    await expect(samplingReady.or(governedStop)).toBeVisible({ timeout: 60_000 });

    if (await governedStop.isVisible()) {
      // Real, governed terminal state -- hold it so the frame clearly
      // reads as a deliberate decision, not an error.
      await page.waitForTimeout(6000);
    } else {
      // The real deterministic policy asked for a sample this run --
      // still a real, honest outcome; follow it through.
      await page.waitForTimeout(3000);
      await page.getByRole('button', { name: 'Collect reference sample' }).click();
      await expect(page.getByText('Verified plan ready:')).toBeVisible({ timeout: 60_000 });
      await page.waitForTimeout(5000);
    }
    await finalizeClip(page, '08-live-v5-v2');
  });

  test('09-validation-model-network', async ({ page }) => {
    await openReferenceAtMilestone(page, 9);

    // A. Validation -- HydroCore-v5 final evaluation evidence.
    await page.getByRole('button', { name: 'Validation' }).click();
    await expect(
      page.getByRole('heading', { name: 'HydroCore-v5 final evaluation evidence' }),
    ).toBeVisible();
    await expect(page.getByText(/hard safety counters violated/)).toBeVisible();
    await expect(page.getByText('HydroCore-v5 M11.6 final evaluation summary')).toBeVisible();
    await page.waitForTimeout(6000);

    // B. Model & Authority -- Decision authority path.
    await page.getByRole('button', { name: 'Model & Authority' }).click();
    await expect(page.getByRole('heading', { name: 'Decision authority path' })).toBeVisible();
    await expect(
      page.locator('.authority-path-label').filter({ hasText: 'HydroCore-v5 Sentinel' }),
    ).toBeVisible();
    await expect(
      page.locator('.authority-path-label').filter({ hasText: 'Human operator' }),
    ).toBeVisible();
    await page.waitForTimeout(6000);

    // C. Network -- standard EPANET .inp import/validation surface.
    await page.getByRole('button', { name: 'Network' }).click();
    await expect(page.getByRole('heading', { name: 'Import network' })).toBeVisible();
    await expect(page.getByLabel('EPANET .inp file')).toBeVisible();
    await page.waitForTimeout(4000);

    await finalizeClip(page, '09-validation-model-network');
  });
});
