import { defineConfig, devices } from '@playwright/test';

/**
 * Dedicated config for the final-presentation demo-clip recording flow.
 * Deliberately separate from ../../playwright.config.ts (the real E2E /
 * visual-regression suite): this one targets a real running release
 * instance instead of a locally built preview server, records video for
 * every test instead of only on failure, and is never part of the
 * regression gate. Running it must never change anything about the
 * existing E2E config or its snapshots.
 *
 * Target app: the real published `ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.0`
 * release, started separately via `docker compose -f docker-compose.release.yml up`
 * (or an equivalent real v0.2.0 runtime) and expected at HYDROSWARM_RECORDING_BASE_URL
 * (default http://127.0.0.1:8765). No webServer block here on purpose --
 * this harness never boots its own (dev-built, mockable) server.
 */
const BASE_URL = process.env.HYDROSWARM_RECORDING_BASE_URL ?? 'http://127.0.0.1:8765';

export default defineConfig({
  testDir: '.',
  testMatch: 'record-demo-clips.spec.ts',
  globalSetup: './global-setup.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  use: {
    baseURL: BASE_URL,
    viewport: { width: 1920, height: 1080 },
    video: {
      mode: 'on',
      size: { width: 1920, height: 1080 },
    },
    trace: 'off',
    screenshot: 'off',
  },
  projects: [
    {
      name: 'chromium-demo-recording',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1920, height: 1080 } },
    },
  ],
});
