import { expect, test } from '@playwright/test';

test.describe('Self-Serve Audit Hub smoke flow', () => {
  test('loads, runs analysis, and emits no page errors', async ({ page }) => {
    const pageErrors: string[] = [];

    page.on('pageerror', (err) => {
      pageErrors.push(err.message);
    });

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        pageErrors.push(msg.text());
      }
    });

    await page.goto('/index.html');

    await expect(page.getByRole('heading', { name: 'Self-Serve Audit Hub' })).toBeVisible();

    // Use sample helper to exercise UX and seeded text path.
    await page.getByRole('button', { name: 'Try Example Input' }).click();

    const input = page.locator('#audit-input');
    await expect(input).toHaveValue(/Hamiltonian|GUE|Berry-Keating/i);

    await page.getByRole('button', { name: 'Run Spectral Analysis' }).click();

    await expect(page.locator('#audit-result')).toBeVisible();
    await expect(page.locator('#audit-result')).toContainText(/Strong Coherence \(Tier 1\)|Weak Evidence \(Tier 3\)/);

    // Quick XSS sanity: injected-looking string should remain inert and not crash page.
    await input.fill('<img src=x onerror=alert(1)> GUE');
    await page.getByRole('button', { name: 'Run Spectral Analysis' }).click();
    await expect(page.locator('#audit-result')).toContainText('Keywords:');

    // Validate notebook button resolved to either local or Colab URL.
    const notebookHref = await page.locator('#notebook-btn').getAttribute('href');
    expect(notebookHref).toBeTruthy();
    expect(
      notebookHref === 'notebooks/gemma4_starter.ipynb' ||
      notebookHref?.includes('colab.research.google.com/github/holeyfield33-art/unitarity-lab/blob/main/notebooks/gemma4_starter.ipynb')
    ).toBeTruthy();

    expect(pageErrors, `Console/page errors found: ${pageErrors.join(' | ')}`).toEqual([]);
  });
});
