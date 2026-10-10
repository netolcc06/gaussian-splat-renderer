// Visitor smoke test of the published page (W3/W4): opens the page in normal (non-test) mode,
// waits for the auto-loaded demo scene, checks errors/credits/stats, exercises the panel (grid
// demo, renderer switch, touch-free camera drag) and saves screenshots of the whole page.
//
//   node tools/web/smoke_page.mjs [--url https://...]   (default: serve build-web/dist locally)
//   writes build-web/web_checks/smoke_*.png and smoke_result.json; exit 1 on failure.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const args = process.argv.slice(2);
const urlArg = args.includes('--url') ? args[args.indexOf('--url') + 1] : null;
const outDir = 'build-web/web_checks';
fs.mkdirSync(outDir, { recursive: true });
const chrome = process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

let server = null;
let url = urlArg;
if (!url) {
  const port = 8137;
  server = spawn(process.execPath, ['tools/web/serve.mjs', 'build-web/dist', String(port)], { stdio: 'ignore' });
  await new Promise((r) => setTimeout(r, 800));
  url = `http://localhost:${port}/`;
}

const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: 'new',
  args: ['--force_high_performance_gpu', '--use-webgpu-adapter=d3d12', '--enable-unsafe-webgpu', '--window-size=1400,850'],
  defaultViewport: { width: 1400, height: 850 },
});
const consoleErrors = [];
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(String(e)));
  await page.goto(url, { waitUntil: 'load' });
  const state = await page.evaluate(() => window.gsrTest.ready);
  check('page ready', state === 'ready', state);

  // The demo loads by itself.
  await page.waitForFunction(() => window.gsrTest.lastLoad !== undefined, { timeout: 30000 });
  const load = await page.evaluate(() => window.gsrTest.lastLoad);
  check('demo scene loads', load && load.ok, load && (load.ok ? `${load.scene.count} gaussians in ${(load.ms / 1000).toFixed(2)} s` : load.error));
  await page.evaluate(() => window.gsrTest.frames(30));
  await new Promise((r) => setTimeout(r, 1200));  // let the stats panel refresh
  const credit = await page.$eval('#credit-knock', (e) => !e.hidden && e.textContent.includes('CC BY 4.0'));
  check('CC BY credit shown with the demo', credit);
  const stats = await page.$eval('#stats', (e) => e.textContent);
  check('stats panel filled', /Gaussians/.test(stats) && /FPS/.test(stats), stats.replace(/\s+/g, ' ').slice(0, 160));
  const s = await page.evaluate(() => window.gsrTest.stats());
  check('splat-tiles renderer active', s.renderer === 'splat-tiles' && !s.tilesError, `${s.renderer} ${s.tilesError || ''}`);
  await page.screenshot({ path: path.join(outDir, 'smoke_demo.png') });

  // Camera drag (right of the panel) changes the view; reset returns to the demo camera.
  const cam0 = await page.evaluate(() => window.gsrTest.camera());
  await page.mouse.move(900, 400);
  await page.mouse.down();
  await page.mouse.move(1000, 420, { steps: 5 });
  await page.mouse.up();
  const cam1 = await page.evaluate(() => window.gsrTest.camera());
  check('drag orbits', Math.abs(cam1.yaw_deg - cam0.yaw_deg) > 1);
  await page.click('#reset-camera');
  const cam2 = await page.evaluate(() => window.gsrTest.camera());
  check('reset returns to the demo view', Math.abs(cam2.yaw_deg - cam0.yaw_deg) < 1e-3);

  // Points renderer, then back.
  await page.select('#opt-renderer', 'points');
  await page.evaluate(() => window.gsrTest.frames(5));
  check('switch to points', (await page.evaluate(() => window.gsrTest.stats().renderer)) === 'points');
  await page.screenshot({ path: path.join(outDir, 'smoke_points.png') });
  await page.select('#opt-renderer', 'splat-tiles');

  // Synthetic grid demo button; the credit disappears.
  await page.click('#demo-grid');
  const grid = await page.evaluate(() => window.gsrTest.lastLoad);
  check('grid demo loads', grid && grid.ok && grid.scene.count === 512);
  check('credit hidden for the grid', await page.$eval('#credit-knock', (e) => e.hidden));
  await page.evaluate(() => window.gsrTest.frames(10));
  await page.screenshot({ path: path.join(outDir, 'smoke_grid.png') });

  const errs = await page.evaluate(() => window.gsrTest.stats());
  check('0 WebGPU errors', (errs.surfaceErrors || 0) === 0 && (errs.shaderErrors || 0) === 0 && !errs.failure, JSON.stringify({ surface: errs.surfaceErrors, shader: errs.shaderErrors }));
  check('0 console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
} catch (e) {
  check('smoke test ran', false, String(e && e.stack || e));
} finally {
  await browser.close();
  if (server) server.kill();
}
fs.writeFileSync(path.join(outDir, 'smoke_result.json'), JSON.stringify({ url, results, consoleErrors }, null, 2));
const failed = results.filter((r) => !r.ok).length;
console.log(failed ? `FAILED (${failed})` : 'ALL PASSED');
process.exit(failed ? 1 : 0);
