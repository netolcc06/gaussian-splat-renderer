// Automated checks of the WebGPU build in headless Chrome (puppeteer-core + the installed
// Chrome). One command per web phase, like the native tools/run_*.ps1 scripts:
//
//   node tools/web/run_web_checks.mjs --test w0
//
// Options:
//   --test w0|w1|w2      which phase's checks to run (W0: gradient + hygiene; W1: loading + points;
//                        W2: splat-tiles, tools/web/w2_checks.mjs)
//   --dist <dir>         site to serve (default build-web/dist)
//   --url <url>          test an already served site instead of serving --dist
//   --chrome <exe>       Chrome binary (default: $CHROME_PATH or the standard install path)
//   --any-gpu            do not require the NVIDIA adapter
//   --headful            show the browser window
//   --out <dir>          logs and result JSON (default build-web/web_checks)
//   --skip-big           W1/W2: skip bonsai_30k (309 MB)
//   --quick              W2: in-page self-tests only
//   --skip-perf          W2: skip the 1920x1080 performance runs
//
// W0 checks (docs/web/DESIGN.md section 10): the page reaches "ready" on the NVIDIA adapter;
// the shader and preprocessor self-tests pass (no compiler errors or warnings); the captured
// offscreen image equals the gradient formula exactly at 960x540 and at an odd size; 0 WebGPU
// errors, no device loss and no console errors after the frames ran. The 960x540 capture is
// written to docs/web/img/w0_gradient.png.
//
// W1 checks (docs/web/W1.md): the GPU scene decoder equals the CPU decoder in every web format
// (decode self-test); grid_512_sh3 (URL stream, f32/f16/q8, and a .gz File), utsuho_plush and
// bonsai_30k (file picker) load, and their points images match the native `gsplat --renderer
// points` screenshot of the same scene/camera/format (imgdiff, gate below; utsuho also with an
// orbited camera exported as JSON); truncated files give the native error texts; the wasm heap
// peak while loading bonsai stays < 200 MB; 0 WebGPU errors, 0 console errors. Needs the native
// Release build (gsplat.exe, imgdiff.exe). Test scenes are served from data/ under /__testdata/,
// never from dist/. Images: docs/web/img/w1_*.png.
//
// Exit code 0 = all passed, 1 = a check failed.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');

function parseArgs(argv) {
  const opts = {
    test: 'w0',
    dist: path.join(repo, 'build-web', 'dist'),
    url: '',
    chrome: process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    anyGpu: false,
    headful: false,
    out: path.join(repo, 'build-web', 'web_checks'),
    skipBig: false,
    quick: false,
    skipPerf: false,
  };
  for (let i = 2; i < argv.length; ++i) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--test') opts.test = next().toLowerCase();
    else if (a === '--dist') opts.dist = path.resolve(next());
    else if (a === '--url') opts.url = next();
    else if (a === '--chrome') opts.chrome = next();
    else if (a === '--any-gpu') opts.anyGpu = true;
    else if (a === '--headful') opts.headful = true;
    else if (a === '--out') opts.out = path.resolve(next());
    else if (a === '--skip-big') opts.skipBig = true;
    else if (a === '--quick') opts.quick = true;        // W2: self-tests only
    else if (a === '--skip-perf') opts.skipPerf = true; // W2: no 1080p performance runs
    else throw new Error(`unknown option ${a}`);
  }
  return opts;
}

// --- PNG writer (RGBA8, no dependency) --------------------------------------------------------
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; ++n) {
    let c = n;
    for (let k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function encodePng(width, height, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // RGBA
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; ++y) {
    raw[y * (width * 4 + 1)] = 0;  // filter: none
    rgba.copy(raw, y * (width * 4 + 1) + 1, y * width * 4, (y + 1) * width * 4);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

// --- Expected images --------------------------------------------------------------------------
// The CPU twin of gradientValue() in shaders/wgsl/gradient.wgsl (integer math, so exact).
function expectedGradient(width, height) {
  const out = Buffer.alloc(width * height * 4);
  const dx = Math.max(width - 1, 1), dy = Math.max(height - 1, 1);
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      const i = (y * width + x) * 4;
      out[i] = Math.floor((x * 255) / dx);
      out[i + 1] = Math.floor((y * 255) / dy);
      out[i + 2] = (x + y) & 255;
      out[i + 3] = 255;
    }
  }
  return out;
}

function compareExact(got, expected, width) {
  let mismatches = 0, maxDiff = 0, first = null;
  for (let i = 0; i < expected.length; ++i) {
    const d = Math.abs(got[i] - expected[i]);
    if (d) {
      ++mismatches;
      maxDiff = Math.max(maxDiff, d);
      if (!first) {
        const p = i >> 2;
        first = { x: p % width, y: Math.floor(p / width), channel: 'rgba'[i & 3], got: got[i], expected: expected[i] };
      }
    }
  }
  return { mismatches, maxDiff, first };
}

// --- Runner -----------------------------------------------------------------------------------
const results = [];
function check(name, pass, detail = '') {
  results.push({ name, pass: !!pass, detail });
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

const mb = (n) => `${(n / (1 << 20)).toFixed(1)} MiB`;

// Common start-of-page checks: ready, NVIDIA adapter, shader and preprocessor self-tests.
async function checkStartup(ctx, page, state, label = '') {
  const reason = await page.evaluate(() => window.gsrTest.reason);
  check(`page ready${label}`, state === 'ready', state === 'ready' ? '' : `${state}: ${reason}`);
  if (state !== 'ready') throw new Error('page did not start');
  const stats0 = await page.evaluate(() => window.gsrTest.stats());
  const a = stats0.adapter;
  const adapterText = `${a.vendor} / ${a.architecture} / ${a.description || '(no description)'} (${a.adapterType}, backend ${a.backend})`;
  console.log(`Adapter: ${adapterText}`);
  console.log(`Device features: ${stats0.deviceFeatures.join(' ')}`);
  check('adapter is NVIDIA', ctx.opts.anyGpu || /nvidia/i.test(a.vendor), adapterText);
  const shaders = await page.evaluate(() => window.gsrTest.selftest('shaders'));
  const shaderMsgs = shaders.modules.flatMap((m) => m.messages.map((x) => `${m.label}: ${x}`));
  check('shader self-test (0 errors, 0 warnings)', shaders.pass,
    `${shaders.modules.length} modules${shaderMsgs.length ? '; ' + shaderMsgs.join(' | ') : ''}`);
  const pre = await page.evaluate(() => window.gsrTest.selftest('preprocessor'));
  check('preprocessor self-test', pre.pass, `${pre.details.filter((d) => d.pass).length}/${pre.details.length} cases`);
  return { stats0, shaders, pre };
}

async function checkHygiene(page, label = '') {
  await page.evaluate(() => window.gsrTest.frames(60));
  const stats = await page.evaluate(() => window.gsrTest.stats());
  check(`frames rendered${label}`, stats.frames >= 60, `${stats.frames} frames`);
  check(`0 WebGPU errors${label}`, stats.errors.total === 0,
    `validation ${stats.errors.validation}, oom ${stats.errors.outOfMemory}, internal ${stats.errors.internal}` +
    (stats.errors.recent.length ? `; ${stats.errors.recent.join(' | ')}` : ''));
  check(`device not lost${label}`, !stats.deviceLost, stats.deviceLost ? stats.deviceLostMessage : '');
  check(`canvas surface frames OK${label}`, stats.surfaceErrors === 0, `${stats.surfaceErrors} getCurrentTexture failures`);
  return stats;
}

async function capturePng(page, file) {
  const cap = await page.evaluate(() => window.gsrTest.capture());
  const pixels = Buffer.from(cap.base64, 'base64');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, encodePng(cap.width, cap.height, pixels));
  return { ...cap, pixels };
}

// --- W0: gradient + hygiene ---------------------------------------------------------------------
async function runW0(ctx) {
  const W = 960, H = 540;
  const { page, state } = await ctx.openPage(`test=1&w=${W}&h=${H}`);
  const { shaders, pre } = await checkStartup(ctx, page, state);

  const cap = await page.evaluate(() => window.gsrTest.capture());
  const pixels = Buffer.from(cap.base64, 'base64');
  check('capture size', cap.width === W && cap.height === H && pixels.length === W * H * 4, `${cap.width}x${cap.height}`);
  const cmp = compareExact(pixels, expectedGradient(W, H), W);
  check('gradient exact (960x540)', cmp.mismatches === 0,
    cmp.mismatches ? `${cmp.mismatches} values differ, max ${cmp.maxDiff}, first ${JSON.stringify(cmp.first)}` : 'all 2,073,600 values equal');
  const pngPath = path.join(repo, 'docs', 'web', 'img', 'w0_gradient.png');
  fs.mkdirSync(path.dirname(pngPath), { recursive: true });
  fs.writeFileSync(pngPath, encodePng(W, H, pixels));
  console.log(`Saved ${path.relative(repo, pngPath)}`);

  const stats = await checkHygiene(page);
  ctx.summary.pages.push({ size: `${W}x${H}`, stats, shaders, preprocessor: pre, gradient: cmp });
  await page.close();

  // Odd size: partial workgroups and 256-byte row padding in the readback.
  const W2 = 333, H2 = 187;
  const p2 = await ctx.openPage(`test=1&w=${W2}&h=${H2}`);
  check(`page ready (${W2}x${H2})`, p2.state === 'ready');
  if (p2.state === 'ready') {
    const cap2 = await p2.page.evaluate(() => window.gsrTest.capture());
    const px2 = Buffer.from(cap2.base64, 'base64');
    const cmp2 = compareExact(px2, expectedGradient(W2, H2), W2);
    check(`gradient exact (${W2}x${H2})`, cap2.width === W2 && cap2.height === H2 && cmp2.mismatches === 0,
      cmp2.mismatches ? `${cmp2.mismatches} differ, first ${JSON.stringify(cmp2.first)}` : `${cap2.width}x${cap2.height}`);
    const s2 = await p2.page.evaluate(() => window.gsrTest.stats());
    check(`0 WebGPU errors (${W2}x${H2})`, s2.errors.total === 0 && !s2.deviceLost, `${s2.errors.total} errors`);
    ctx.summary.pages.push({ size: `${W2}x${H2}`, stats: s2, gradient: cmp2 });
  }
  await p2.page.close();
}

// --- W1: streamed loading, scene formats, points renderer vs native, memory -------------------
//
// Test assets come from the repo's data/ directory, served by the test server under /__testdata/
// (never part of dist/). The native side is build/bin/Release/gsplat.exe --renderer points with
// the same scene, camera, size and scene format; tools imgdiff compares the two PNGs.

// Points are opaque discs drawn by the same vertex math, so the images should be identical up to
// (a) the 8-bit rounding of colours that go through the rgba16float target on the web but are
// written straight to RGBA8 natively (at most 1 level), and (b) pixels where a different point
// wins the depth test (depth near-ties, or a disc-edge pixel decided differently by D3D12 and
// Vulkan in front of another point): rare but large. Measured (docs/web/W1.md): synthetic grid
// identical in f32/f16/q8; utsuho 62.9 dB, bonsai 54.7 dB, < 0.04% of the pixels above 2/255 and
// none of them on a silhouette (always point vs point, never point vs background).
// Gate: PSNR >= 50 dB and at most 0.1% of the pixels off by more than 2/255.
const POINTS_MIN_PSNR = 50;
const POINTS_MAX_OVER_PCT = 0.1;

function runNative(ctx, args, logName) {
  const exe = path.join(repo, 'build', 'bin', 'Release', 'gsplat.exe');
  const r = spawnSync(exe, args, { cwd: repo, encoding: 'utf8', timeout: 300000 });
  fs.writeFileSync(path.join(ctx.opts.out, logName), `${exe} ${args.join(' ')}\n${r.stdout || ''}${r.stderr || ''}`);
  return r.status;
}

function imgdiff(ctx, test, ref, name) {
  const exe = path.join(repo, 'build', 'bin', 'Release', 'imgdiff.exe');
  const json = path.join(ctx.opts.out, `${name}_diff.json`);
  const heat = path.join(ctx.opts.out, `${name}_heatmap.png`);
  const r = spawnSync(exe, [test, ref, '--json', json, '--heatmap', heat], { cwd: repo, encoding: 'utf8' });
  if (r.status > 1 || !fs.existsSync(json)) throw new Error(`imgdiff failed: ${r.stdout}${r.stderr}`);
  return JSON.parse(fs.readFileSync(json, 'utf8'));
}

const psnrText = (d) => (d.psnr_db >= 999 ? 'inf' : d.psnr_db.toFixed(2));

// One web-vs-native points comparison. `web` = {page, load: () => promise of the load result}.
async function comparePoints(ctx, page, c) {
  const W = 960, H = 540;
  const webPng = path.join(ctx.opts.out, `w1_${c.name}_web.png`);
  const nativePng = path.join(ctx.opts.out, `w1_${c.name}_native.png`);
  const cap = await capturePng(page, webPng);
  if (c.docImage) fs.copyFileSync(webPng, path.join(repo, 'docs', 'web', 'img', c.docImage));
  const args = ['--scene', c.scene, '--renderer', 'points', '--width', `${W}`, '--height', `${H}`,
    '--screenshot', nativePng, ...c.nativeArgs];
  const code = runNative(ctx, args, `w1_${c.name}_native.log`);
  check(`native points screenshot ${c.name}`, code === 0 && fs.existsSync(nativePng), `exit ${code}`);
  if (code !== 0) return null;
  const d = imgdiff(ctx, webPng, nativePng, `w1_${c.name}`);
  const ok = d.psnr_db >= POINTS_MIN_PSNR && d.pixels_over_threshold_pct <= POINTS_MAX_OVER_PCT;
  check(`points ${c.name}: web vs native`, ok && cap.width === W && cap.height === H,
    `PSNR ${psnrText(d)} dB, max ${(d.max_abs * 255).toFixed(0)}/255, mean ${(d.mean_abs * 255).toFixed(4)}/255, ` +
    `${d.pixels_over_threshold_pct.toFixed(4)}% px > 2/255`);
  ctx.summary.points.push({ name: c.name, psnr: d.psnr_db, maxAbs255: d.max_abs * 255, meanAbs255: d.mean_abs * 255,
    overPct: d.pixels_over_threshold_pct, nativeArgs: c.nativeArgs });
  return d;
}

async function loadByUrl(page, url) {
  return page.evaluate((u) => window.gsrTest.loadUrl(u), url);
}

async function loadByFile(page, file) {
  const input = await page.$('#file-input');
  await input.uploadFile(file);
  await page.waitForFunction(() => window.gsrTest.lastLoad, { timeout: 60000 });
  const r = await page.evaluate(() => window.gsrTest.lastLoad);
  await page.evaluate(() => { window.gsrTest.lastLoad = null; });
  return r;
}

async function runW1(ctx) {
  const W = 960, H = 540;
  ctx.summary.points = [];
  // W2 made splat-tiles the default renderer of a loaded scene: these checks are about points.
  const { page, state } = await ctx.openPage(`test=1&w=${W}&h=${H}&renderer=points`);
  const { shaders } = await checkStartup(ctx, page, state);
  ctx.summary.shaders = shaders;
  const heap0 = (await page.evaluate(() => window.gsrTest.stats())).heap;
  console.log(`wasm heap at start: ${mb(heap0.heapSize)} (malloc in use ${mb(heap0.mallocInUse)})`);

  // GPU decoder vs CPU decoder, every web scene format.
  const decode = await page.evaluate(() => window.gsrTest.selftestAsync('decode'));
  ctx.summary.decode = decode;
  for (const c of decode.cases || []) {
    check(`decode self-test: ${c.name}`, c.pass,
      c.pass ? `${c.bytesPerGaussian} B/gaussian, max quaternion error ${c.maxQuaternionError.toExponential(1)}`
        : `${c.error || ''} ${c.mismatches || 0} mismatches ${JSON.stringify(c.firstMismatch || {})}`);
  }
  check('decode self-test (all cases)', decode.pass && (decode.cases || []).length >= 13, `${(decode.cases || []).length} cases`);

  const synthCamera = JSON.parse(fs.readFileSync(path.join(repo, 'tools', 'cameras', 'synthetic.json'), 'utf8'));
  const gridUrl = '/__testdata/synthetic/grid_512_sh3.ply';

  // grid_512_sh3 through fetch (stream reader), in the three web formats, synthetic camera.
  for (const format of ['f32', 'f16', 'q8']) {
    check(`set scene-format ${format}`, await page.evaluate((f) => window.gsrTest.setOption('scene-format', f), format));
    const r = await loadByUrl(page, gridUrl);
    check(`load grid_512_sh3 (${format}, URL stream)`, r.ok && r.scene.count === 512 && r.scene.shDegree === 3 &&
      r.scene.format === format && r.scene.warnings.length === 0,
      r.ok ? `${r.scene.count} gaussians, ${r.scene.format}/${r.scene.layout}, ${r.scene.bytesPerGaussian} B/gaussian, ` +
        `up ${r.scene.upAxis}, ${r.ms.toFixed(1)} ms` : r.error);
    check(`set camera (${format})`, await page.evaluate((cam) => window.gsrTest.setCamera(cam), synthCamera));
    await comparePoints(ctx, page, {
      name: `grid_${format}`, scene: 'data/synthetic/grid_512_sh3.ply',
      nativeArgs: ['--camera', 'tools/cameras/synthetic.json', ...(format === 'f32' ? [] : ['--scene-format', format])],
      docImage: format === 'f32' ? 'w1_grid_points.png' : null,
    });
  }
  await page.evaluate(() => window.gsrTest.setOption('scene-format', 'f32'));

  // utsuho_plush through the file picker (File.slice reader): auto-framed camera (as native
  // without cameras.json: up -y, framing of the robust box), then an orbited camera exported as
  // JSON, set back on the web and given to native with --camera.
  const utsuho = path.join(repo, 'data', 'scenes', 'utsuho_plush', 'point_cloud.ply');
  if (fs.existsSync(utsuho)) {
    const r = await loadByFile(page, utsuho);
    check('load utsuho_plush (file picker)', r.ok && r.scene.count === 53671 && r.scene.shDegree === 3 && r.scene.upAxis === '-y',
      r.ok ? `${r.scene.count} gaussians, ${mb(r.scene.gpuBytes)} on the GPU, ${r.ms.toFixed(0)} ms` : r.error);
    ctx.summary.utsuho = r;
    await comparePoints(ctx, page, { name: 'utsuho_auto', scene: utsuho, nativeArgs: [], docImage: 'w1_utsuho_points.png' });
    await page.evaluate(() => { window.gsrTest.orbit(-160, 60); window.gsrTest.module._gsr_input_zoom(3); });
    const cam = await page.evaluate(() => window.gsrTest.camera());
    const camFile = path.join(ctx.opts.out, 'w1_utsuho_orbit_camera.json');
    fs.writeFileSync(camFile, JSON.stringify(cam, null, 2));
    check('set exported camera', await page.evaluate((c) => window.gsrTest.setCamera(c), cam));
    await comparePoints(ctx, page, { name: 'utsuho_orbit', scene: utsuho, nativeArgs: ['--camera', camFile],
      docImage: 'w1_utsuho_points_orbit.png' });

    // Format fallback: with an 8 MiB buffer limit the f32 SH array (10.3 MiB) does not fit, the
    // f16 colour records (5.2 MiB) do: the scene loads as f16 and says why.
    await page.evaluate(() => window.gsrTest.setOption('debug-max-buffer-mb', '8'));
    const fb = await loadByFile(page, utsuho);
    await page.evaluate(() => window.gsrTest.setOption('debug-max-buffer-mb', '0'));
    check('format fallback f32 -> f16 (8 MiB buffer limit)', fb.ok && fb.scene.format === 'f16' &&
      fb.scene.requestedFormat === 'f32' && fb.scene.fallback.length > 0, fb.ok ? fb.scene.fallback : fb.error);
    if (fb.ok) {
      await page.evaluate((c) => window.gsrTest.setCamera(c), cam);
      await comparePoints(ctx, page, { name: 'utsuho_fallback_f16', scene: utsuho,
        nativeArgs: ['--camera', camFile, '--scene-format', 'f16'] });
    }
  } else {
    console.log('utsuho_plush not present: skipped');
  }

  // Bad files: same PlyError text as native (tests/test_ply_stream.cpp proves the stream and file
  // paths agree; the texts here are the native ones), and the previous scene stays.
  const before = await page.evaluate(() => window.gsrTest.scene().count);
  const badCases = [
    { name: 'grid_truncated.ply', keep: 1, expect: 'grid_truncated.ply: file is truncated: header declares 512 gaussians x 248 bytes = 126976 bytes of vertex data, but only 63488 bytes follow the header (256 complete records)' },
    { name: 'grid_short_header.ply', keepBytes: 200, expect: "grid_short_header.ply: PLY header has no 'end_header' line (file truncated or not a PLY)" },
  ];
  for (const b of badCases) {
    const r = await page.evaluate(async (u, b) => {
      const bytes = new Uint8Array(await (await fetch(u)).arrayBuffer());
      const header = new TextDecoder().decode(bytes.subarray(0, 4096)).indexOf('end_header\n') + 11;
      const n = b.keepBytes ?? header + Math.floor((bytes.length - header) / 2);
      return window.gsrTest.loadFile(new File([bytes.subarray(0, n)], b.name));
    }, gridUrl, b);
    check(`bad file ${b.name}: native error text`, !r.ok && r.error === b.expect, r.ok ? 'loaded' : r.error);
  }
  const trailing = await page.evaluate(async (u) => {
    const bytes = new Uint8Array(await (await fetch(u)).arrayBuffer());
    const garbage = new Uint8Array(1000).map((_, i) => (i * 37) & 255);
    return window.gsrTest.loadFile(new File([bytes, garbage], 'grid_trailing.ply'));
  }, gridUrl);
  check('trailing garbage: loads with the native warning', trailing.ok &&
    trailing.scene.warnings.join('|') === '1000 unexpected trailing bytes after the last element',
    trailing.ok ? trailing.scene.warnings.join('|') : trailing.error);
  const after = await page.evaluate(() => window.gsrTest.scene().count);
  check('failed loads keep the previous scene', before > 0 && after === 512, `before ${before}, after ${after}`);

  // gzip: the page compresses the file (CompressionStream) and loads it as a .ply.gz File, which
  // goes through DecompressionStream with an unknown total size (checks at the end of the stream).
  const gz = await page.evaluate(async (u) => {
    const bytes = await (await fetch(u)).arrayBuffer();
    const gzBlob = await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).blob();
    const r = await window.gsrTest.loadFile(new File([gzBlob], 'grid_512_sh3.ply.gz'));
    return { ...r, gzBytes: gzBlob.size };
  }, gridUrl);
  check('gzip file (DecompressionStream, size unknown)', gz.ok && gz.scene.count === 512 && gz.scene.name === 'grid_512_sh3.ply' &&
    gz.scene.fileBytes === 128594, gz.ok ? `${gz.gzBytes} B gzip -> ${gz.scene.fileBytes} B` : gz.error);
  if (gz.ok) {
    await page.evaluate((cam) => window.gsrTest.setCamera(cam), synthCamera);
    await comparePoints(ctx, page, { name: 'grid_gz', scene: 'data/synthetic/grid_512_sh3.ply',
      nativeArgs: ['--camera', 'tools/cameras/synthetic.json'] });
  }

  // bonsai_30k (309 MB) through the file picker: wasm heap and JS heap during the load, load time.
  const bonsai = path.join(repo, 'data', 'scenes', 'bonsai_30k', 'point_cloud.ply');
  if (fs.existsSync(bonsai) && !ctx.opts.skipBig) {
    const r = await loadByFile(page, bonsai);
    ctx.summary.bonsai = r;
    if (r.ok) {
      const h = r.scene.heap;
      const t = r.scene.timingsMs;
      console.log(`bonsai: ${(r.bytes / 1e6).toFixed(1)} MB in ${r.ms.toFixed(0)} ms (${(r.bytes / 1e6 / (r.ms / 1000)).toFixed(0)} MB/s); ` +
        `inside push ${t.insidePush.toFixed(0)} ms (decode ${t.decode.toFixed(0)}, encode ${t.encode.toFixed(0)}, upload ${t.upload.toFixed(0)}), stats ${t.stats.toFixed(0)} ms`);
      console.log(`bonsai wasm heap: size ${mb(h.heapSizeBefore)} -> peak ${mb(h.heapSizePeak)}; malloc in use ${mb(h.mallocInUseBefore)} -> peak ${mb(h.mallocInUsePeak)} -> after ${mb(h.mallocInUseAfter)}; JS heap peak ${mb(r.jsHeapPeak)}`);
    }
    check('load bonsai_30k (file picker)', r.ok && r.scene.count === 1244819, r.ok ? `${r.scene.count} gaussians, ${mb(r.scene.gpuBytes)} on the GPU, ${(r.ms / 1000).toFixed(2)} s` : r.error);
    check('bonsai: wasm heap peak < 200 MB', r.ok && r.scene.heap.heapSizePeak < 200 * (1 << 20),
      r.ok ? `heap ${mb(r.scene.heap.heapSizePeak)}, malloc peak ${mb(r.scene.heap.mallocInUsePeak)}` : '');
    if (r.ok) {
      // Native with the web's default scene transform (no cameras.json on the web: up -y).
      await comparePoints(ctx, page, { name: 'bonsai_auto', scene: bonsai, nativeArgs: ['--up', '-y'], docImage: 'w1_bonsai_points.png' });
      // The same file as q8: the per-batch encoder (encodeScene, single-threaded wasm) runs now.
      await page.evaluate(() => window.gsrTest.setOption('scene-format', 'q8'));
      const q = await loadByFile(page, bonsai);
      await page.evaluate(() => window.gsrTest.setOption('scene-format', 'f32'));
      ctx.summary.bonsaiQ8 = q;
      if (q.ok) {
        const t = q.scene.timingsMs;
        console.log(`bonsai q8: ${(q.ms / 1000).toFixed(2)} s; decode ${t.decode.toFixed(0)} ms, encode ${t.encode.toFixed(0)} ms, ` +
          `upload ${t.upload.toFixed(0)} ms; ${mb(q.scene.gpuBytes)} on the GPU; malloc peak ${mb(q.scene.heap.mallocInUsePeak)}`);
      }
      check('load bonsai_30k as q8', q.ok && q.scene.format === 'q8' && q.scene.bytesPerGaussian === 96,
        q.ok ? `${q.scene.bytesPerGaussian} B/gaussian, ${(q.ms / 1000).toFixed(2)} s` : q.error);
      if (q.ok) {
        await comparePoints(ctx, page, { name: 'bonsai_q8', scene: bonsai, nativeArgs: ['--up', '-y', '--scene-format', 'q8'] });
      }
    }
  } else {
    console.log('bonsai_30k not present (or --skip-big): skipped');
  }

  // Pointer input through real mouse events (W3 adds touch): a 100 px drag orbits by 25 degrees
  // (0.25 deg/px, as native), one wheel notch zooms the orbit distance by 0.9.
  await page.evaluate(() => window.gsrTest.resetCamera());
  const box = await (await page.$('#canvas')).boundingBox();
  const cam0 = await page.evaluate(() => window.gsrTest.camera());
  // Drag right of the control panel (it covers the top-left ~360 px).
  await page.mouse.move(box.x + 500, box.y + 200);
  await page.mouse.down();
  await page.mouse.move(box.x + 600, box.y + 200, { steps: 5 });
  await page.mouse.up();
  const cam1 = await page.evaluate(() => window.gsrTest.camera());
  await page.mouse.wheel({ deltaY: -100 });
  await page.evaluate(() => window.gsrTest.frames(2));
  const cam2 = await page.evaluate(() => window.gsrTest.camera());
  const dist = (c) => Math.hypot(...c.position.map((p, i) => p - c.target[i]));
  const dYaw = cam1.yaw_deg - cam0.yaw_deg;
  check('pointer drag orbits the camera', Math.abs(dYaw - 25) < 1e-3 && Math.abs(cam1.pitch_deg - cam0.pitch_deg) < 1e-3,
    `yaw ${cam0.yaw_deg.toFixed(2)} -> ${cam1.yaw_deg.toFixed(2)}`);
  check('wheel zooms the camera', Math.abs(dist(cam2) / dist(cam1) - 0.9) < 1e-4,
    `distance ${dist(cam1).toFixed(4)} -> ${dist(cam2).toFixed(4)}`);

  ctx.summary.stats = await checkHygiene(page);
  await page.close();
}

// --- W2: splat-tiles (see the W2 section further down) ------------------------------------------
import { runW2 } from './w2_checks.mjs';

async function main() {
  const opts = parseArgs(process.argv);
  const tests = { w0: runW0, w1: runW1, w2: (ctx) => runW2(ctx, { check, checkStartup, checkHygiene, capturePng, encodePng,
    runNative, imgdiff, psnrText, loadByUrl, loadByFile, repo, mb }) };
  if (!tests[opts.test]) throw new Error(`unknown --test ${opts.test} (available: ${Object.keys(tests).join(', ')})`);
  fs.mkdirSync(opts.out, { recursive: true });
  const logPath = path.join(opts.out, `${opts.test}_console.log`);
  const log = fs.createWriteStream(logPath);

  let server = null;
  let baseUrl = opts.url;
  if (!baseUrl) {
    if (!fs.existsSync(path.join(opts.dist, 'index.html'))) throw new Error(`no build at ${opts.dist} (run tools/web/build_web.ps1)`);
    // Test assets (data/) are mounted beside the site, never copied into it.
    server = await startServer(opts.dist, 0, { '/__testdata/': path.join(repo, 'data') });
    baseUrl = server.url;
  }
  console.log(`Site: ${baseUrl}\nChrome: ${opts.chrome}`);

  // Flags (DESIGN.md section 9): new headless mode keeps WebGPU; --force_high_performance_gpu
  // picks the NVIDIA GPU on this hybrid laptop (Chrome takes the Intel iGPU without it).
  // --enable-precise-memory-info: unquantized performance.memory (JS heap during loads).
  const browser = await puppeteer.launch({
    executablePath: opts.chrome,
    headless: !opts.headful,
    protocolTimeout: 600000,
    args: [
      '--force_high_performance_gpu',
      '--use-webgpu-adapter=d3d12',
      '--enable-unsafe-webgpu',
      '--enable-precise-memory-info',
      // W2: unquantized timestamp-query results (Chrome rounds them to 100 us otherwise).
      ...(opts.test === 'w2' ? ['--enable-webgpu-developer-features'] : []),
      '--no-first-run',
      '--no-default-browser-check',
    ],
  });
  const consoleErrors = [];
  const summary = { test: opts.test, chrome: await browser.version(), pages: [] };

  async function openPage(query) {
    const page = await browser.newPage();
    page.setDefaultTimeout(300000);
    const tag = `[${query}]`;
    page.on('console', (m) => {
      log.write(`${tag} ${m.type()}: ${m.text()}\n`);
      if (m.type() === 'error') consoleErrors.push(`${tag} ${m.text()}`);
    });
    page.on('pageerror', (e) => {
      log.write(`${tag} pageerror: ${e.message}\n`);
      consoleErrors.push(`${tag} pageerror: ${e.message}`);
    });
    page.on('response', (r) => {
      if (r.status() >= 400) log.write(`${tag} HTTP ${r.status()}: ${r.url()}\n`);
    });
    page.on('requestfailed', (r) => {
      log.write(`${tag} requestfailed: ${r.url()} ${r.failure()?.errorText}\n`);
      consoleErrors.push(`${tag} requestfailed: ${r.url()}`);
    });
    await page.goto(new URL(`index.html?${query}`, baseUrl).href, { waitUntil: 'load' });
    // Wait on the page's own promise (adapter/device creation is asynchronous), not a fixed time.
    const state = await page.evaluate(() => window.gsrTest.ready);
    return { page, state };
  }

  const ctx = { opts, openPage, summary };
  try {
    await tests[opts.test](ctx);
    check('0 console errors', consoleErrors.length === 0, consoleErrors.slice(0, 5).join(' | '));
  } finally {
    await browser.close();
    if (server) await server.close();
    log.end();
  }

  summary.results = results;
  summary.pass = results.every((r) => r.pass);
  fs.writeFileSync(path.join(opts.out, `${opts.test}_result.json`), JSON.stringify(summary, null, 2));
  console.log(`\nConsole log: ${path.relative(repo, logPath)}\nResult: ${path.relative(repo, path.join(opts.out, `${opts.test}_result.json`))}`);
  console.log(`\n| check | result |\n|---|---|\n${results.map((r) => `| ${r.name} | ${r.pass ? 'PASS' : 'FAIL'} |`).join('\n')}`);
  console.log(summary.pass ? '\nALL PASSED' : '\nFAILED');
  return summary.pass ? 0 : 1;
}

main().then((code) => process.exit(code)).catch((e) => {
  console.error(e);
  process.exit(1);
});
