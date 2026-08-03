import { expect, test } from '@playwright/test';

test('operator understands the fallback incident and verified response', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('DETERMINISTIC DEMO FALLBACK')).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Verified response awaiting approval' }),
  ).toBeVisible();
  await expect(page.getByText('Collect sample at J123')).toBeVisible();
  await page.getByRole('button', { name: 'Validation' }).click();
  await expect(page.getByRole('heading', { name: 'Benchmarks and operating range' })).toBeVisible();
});
