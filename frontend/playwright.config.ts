import { defineConfig, devices } from '@playwright/test';

declare const process: { arch: string };

// Chromium's text and canvas rasterization differs between the native ARM64
// development runner and GitHub's Linux x86_64 E2E runner. Keep strict visual
// regression comparison on each architecture rather than weakening the
// threshold or treating one architecture's raster output as the other one's
// baseline.
// Existing `chromium-linux` snapshots are the native ARM64 capture set.
// GitHub's x86_64 runner receives its own explicit project/baselines.
const chromiumProjectName = process.arch === 'x64' ? 'chromium-x64' : 'chromium';

export default defineConfig({
  testDir: './tests/e2e',
  use: { baseURL: 'http://127.0.0.1:4173', trace: 'retain-on-failure' },
  webServer: {
    command: 'npm run build && npm exec vite preview -- --host 127.0.0.1 --port 4173',
    port: 4173,
    reuseExistingServer: true,
  },
  projects: [{ name: chromiumProjectName, use: { ...devices['Desktop Chrome'] } }],
});
