// Ad-hoc debugging helper (not part of the checks): serves build-web/dist (+ data/ under
// /__testdata/), opens the page in headless Chrome on the NVIDIA GPU and runs a script file of page
// steps exported as `default async function (page, { repo, log })`.
//
//   node tools/web/debug_page.mjs <script.mjs> [query]
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const script = process.argv[2];
const query = process.argv[3] || 'test=1&w=960&h=540';
const server = await startServer(path.join(repo, 'build-web', 'dist'), 0, { '/__testdata/': path.join(repo, 'data') });
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: true,
  protocolTimeout: 600000,
  args: ['--force_high_performance_gpu', '--use-webgpu-adapter=d3d12', '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features', '--no-first-run', '--no-default-browser-check'],
});
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(600000);
  page.on('console', (m) => { if (m.type() === 'error' || process.env.VERBOSE) console.log(`[page ${m.type()}] ${m.text()}`); });
  await page.goto(new URL(`index.html?${query}`, server.url).href, { waitUntil: 'load' });
  const state = await page.evaluate(() => window.gsrTest.ready);
  console.log(`page: ${state}`);
  const mod = await import(pathToFileURL(path.resolve(script)).href);
  await mod.default(page, { repo, log: console.log });
} finally {
  await browser.close();
  await server.close();
}
