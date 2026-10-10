// W2 checks (docs/web/W2.md): the splat-tiles renderer in WebGPU/WGSL.
//
//   node tools/web/run_web_checks.mjs --test w2 [--quick] [--skip-big] [--skip-perf]
//
// 1. Start-up: every WGSL variant (portable and, when the device has them, subgroup builds)
//    compiles with 0 errors / 0 warnings; the GLSL -> WGSL port manifest is in sync
//    (tools/web/check_shader_sync.py).
// 2. In-page GPU self-tests (src/webgpu/wgpu_tile_selftest.cpp): scan, sort, coverage, tiles,
//    image32, with portable and subgroup kernels.
// 3. Reference images at 960x540: the synthetic cases of tools/run_reference_checks.ps1 and the real
//    scenes in data/ (utsuho auto-framed; train/bonsai with their cameras.json and training camera
//    0, exactly the native checks), captured with subgroups and with the portable shaders (the two
//    must be identical), compared with gsplat_ref (gate >= 50 dB, like native) and with the native
//    `gsplat --renderer splat-tiles` screenshot. Every captured frame must have overflow = 0 and
//    coverage mismatch = 0.
// 4. Performance at 1920x1080 (train, bonsai): GPU timestamps per pass during a 360-degree orbit from
//    training camera 0 and the back-to-back throughput, with and without subgroups.
// 5. Hygiene: 0 WebGPU errors, device not lost, 0 console errors.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const W = 960, H = 540;
const REF_MIN_PSNR = 50;  // the native gate (tools/run_reference_checks.ps1)

// The 5 synthetic cases and the real scenes of tools/run_reference_checks.ps1.
const SYNTHETIC = ['single_sh0', 'single_sh3', 'overlap_sh0', 'grid_512_sh3', 'shdir_sh3'];
const REAL = ['utsuho_plush', 'train_7k', 'bonsai_30k'];

function runTool(ctx, exe, args, logName) {
  const r = spawnSync(exe, args, { cwd: ctx.h.repo, encoding: 'utf8', timeout: 600000 });
  fs.writeFileSync(path.join(ctx.opts.out, logName), `${exe} ${args.join(' ')}\n${r.stdout || ''}${r.stderr || ''}`);
  return r;
}

async function selftests(ctx, page) {
  const { check } = ctx.h;
  const out = {};
  for (const name of ['scan', 'sort', 'coverage', 'tiles', 'image32']) {
    const t0 = Date.now();
    const r = await page.evaluate((n) => window.gsrTest.selftestAsync(n), name);
    out[name] = r;
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    if (name === 'scan' || name === 'sort') {
      const cases = r.cases || [];
      const failed = cases.filter((c) => !c.pass);
      for (const c of failed) check(`${name} self-test: ${c.name} (${c.kernels})`, false, JSON.stringify(c));
      const kernels = [...new Set(cases.map((c) => c.kernels))].join(' + ');
      check(`${name} self-test (${cases.length} cases, ${kernels})`, r.pass && cases.length > 0 && failed.length === 0,
        `${cases.length - failed.length}/${cases.length} bit-exact, ${secs} s${r.error ? '; ' + r.error : ''}`);
    } else if (name === 'coverage') {
      check('coverage self-test (stored spans vs CPU tile_coverage)', r.pass,
        `${r.visible} visible, ${r.recordsEqualCpu} records equal, ${r.borderlineDifferences} borderline, ` +
        `${r.hardDifferences} hard differences, ${r.countMismatches} count mismatches, ` +
        `${r.exactTilesMissing} exact tiles missing, visibility CPU/GPU mismatches ${r.visibleCpuGpuMismatches}, ` +
        `mismatch counter ${r.coverageMismatchCounter}, stored entries +${(r.storedOverheadPct ?? 0).toFixed(3)}% vs exact` +
        `${r.error ? '; ' + r.error : ''}`);
    } else if (name === 'tiles') {
      for (const [k, v] of Object.entries(r)) {
        if (typeof v !== 'object' || v === null || !('pass' in v)) continue;
        check(`tiles self-test: ${k}`, v.pass, JSON.stringify(v));
      }
      check('tiles self-test (all)', r.pass, `${secs} s${r.error ? '; ' + r.error : ''}`);
    } else if (name === 'image32') {
      check('image32 self-test (rgba32f vs CPU reference, > 60 dB)', r.pass,
        `PSNR ${r.psnr?.toFixed(2)} dB, max ${r.maxAbs255?.toFixed(2)}/255, ${r.channelsOver2} channels > 2/255, ` +
        `reference ${r.referenceMs?.toFixed(0)} ms${r.error ? '; ' + r.error : ''}`);
    }
  }
  return out;
}

export async function runW2(ctx0, h) {
  const ctx = { ...ctx0, h };
  const { check, checkStartup, checkHygiene } = h;
  const summary = ctx.summary;
  summary.w2 = {};

  // Port manifest in sync with the GLSL sources.
  const sync = spawnSync('python', [path.join(h.repo, 'tools', 'web', 'check_shader_sync.py')], { cwd: h.repo, encoding: 'utf8' });
  check('GLSL -> WGSL port manifest in sync (check_shader_sync.py)', sync.status === 0, (sync.stdout + sync.stderr).trim().split('\n').pop());

  const { page, state } = await ctx.openPage(`test=1&w=${W}&h=${H}`);
  const { shaders } = await checkStartup(ctx, page, state);
  const stats0 = await page.evaluate(() => window.gsrTest.stats());
  summary.w2.shaders = shaders;
  console.log(`Subgroups available: ${stats0.subgroupsAvailable}, active: ${stats0.subgroupsActive}; timestamps: ${stats0.timestampsAvailable}`);
  check('splat-tiles renderer created', !stats0.tilesError, stats0.tilesError || `subgroups ${stats0.subgroupsActive}`);

  summary.w2.selftests = await selftests(ctx, page);
  if (ctx.opts.quick) {
    summary.w2.stats = await checkHygiene(page);
    await page.close();
    return;
  }

  summary.w2.reference = await referenceImages(ctx, page, stats0.subgroupsAvailable);
  if (!ctx.opts.skipPerf) summary.w2.perf = await perfRuns(ctx, page, stats0.subgroupsAvailable);

  const final = await page.evaluate(() => window.gsrTest.stats());
  check('splat-tiles: no coverage mismatch in any frame read back', final.tiles.mismatchTotal === 0,
    `${final.tiles.framesRead} frames read back, mismatch total ${final.tiles.mismatchTotal}`);
  summary.w2.stats = await checkHygiene(page);
  await page.close();
  printTables(summary.w2);
}

// --- Reference images ---------------------------------------------------------------------------

async function waitFrames(page, n) {
  await page.evaluate((k) => window.gsrTest.frames(k), n);
}

// Captures the next frame and returns the splat-tiles counters of that very frame. Retries (after
// more frames) while the adaptive list is still growing (overflow > 0), as native's 2-frame-late
// capacity does in the first frames.
async function captureChecked(ctx, page, file) {
  let last = null;
  for (let attempt = 0; attempt < 5; ++attempt) {
    await waitFrames(page, 6);
    const cap = await ctx.h.capturePng(page, file);
    const frame = await page.evaluate(() => window.gsrTest.captureFrame());
    const counters = await page.evaluate((f) => window.gsrTest.tilesFrameAsync(f), frame);
    last = { cap, counters, attempts: attempt + 1 };
    if (counters.overflow === 0) break;
  }
  return last;
}

async function setSubgroups(page, on) {
  const ok = await page.evaluate((v) => window.gsrTest.setOption('subgroups', v), on ? 'auto' : 'off');
  await waitFrames(page, 3);  // the renderer is recreated before the next frame
  const s = await page.evaluate(() => window.gsrTest.stats());
  return ok && s.subgroupsActive === on;
}

function sameBytes(a, b) {
  return fs.readFileSync(a).equals(fs.readFileSync(b));
}

async function referenceImages(ctx, page, subgroupsAvailable) {
  const { check, imgdiff, psnrText, repo } = ctx.h;
  const out = ctx.opts.out;
  const bin = path.join(repo, 'build', 'bin', 'Release');
  const synthCamera = fs.readFileSync(path.join(repo, 'tools', 'cameras', 'synthetic.json'), 'utf8');
  const cases = [];
  for (const n of SYNTHETIC) {
    cases.push({ name: n, kind: 'synthetic', scene: `data/synthetic/${n}.ply`, url: `/__testdata/synthetic/${n}.ply`,
      camArgs: ['--camera', 'tools/cameras/synthetic.json'], cameraJson: synthCamera });
  }
  for (const n of REAL) {
    const ply = path.join(repo, 'data', 'scenes', n, 'point_cloud.ply');
    if (!fs.existsSync(ply)) { console.log(`${n}: not present, skipped`); continue; }
    if (n === 'bonsai_30k' && ctx.opts.skipBig) { console.log('bonsai_30k: --skip-big'); continue; }
    const cams = path.join(repo, 'data', 'scenes', n, 'cameras.json');
    const hasCams = fs.existsSync(cams);
    cases.push({ name: n, kind: 'real', scene: `data/scenes/${n}/point_cloud.ply`, url: `/__testdata/scenes/${n}/point_cloud.ply`,
      camArgs: hasCams ? ['--camera-index', '0'] : [], camerasUrl: hasCams ? `/__testdata/scenes/${n}/cameras.json` : null });
  }

  const rows = [];
  for (const c of cases) {
    // Real scenes through the file picker (File.slice, as W1): Chrome aborts a fetch of a 309 MB
    // body once it outgrows its cache, which counts as a failed request even though every byte arrived.
    const r = c.kind === 'real' ? await ctx.h.loadByFile(page, path.join(repo, c.scene)) : await ctx.h.loadByUrl(page, c.url);
    check(`load ${c.name}`, r.ok, r.ok ? `${r.scene.count} gaussians, ${r.ms.toFixed(0)} ms` : r.error);
    if (!r.ok) continue;
    if (c.cameraJson) {
      check(`camera ${c.name}`, await page.evaluate((cam) => window.gsrTest.setCamera(cam), c.cameraJson));
    } else if (c.camerasUrl) {
      const ok = await page.evaluate(async (u) => {
        const text = await (await fetch(u)).text();
        return window.gsrTest.setCamerasJson(text) && window.gsrTest.trainingCamera(0);
      }, c.camerasUrl);
      check(`cameras.json + training camera 0: ${c.name}`, ok);
    }

    // Web captures: subgroup variants (when available) and the portable shaders.
    const modes = subgroupsAvailable ? [true, false] : [false];
    const shots = {};
    for (const sg of modes) {
      check(`${c.name}: subgroups ${sg ? 'on' : 'off'}`, await setSubgroups(page, sg));
      const file = path.join(out, `w2_${c.name}_web_${sg ? 'subgroups' : 'portable'}.png`);
      const shot = await captureChecked(ctx, page, file);
      const k = shot.counters;
      check(`${c.name} (${sg ? 'subgroups' : 'portable'}): captured frame overflow 0, mismatch 0`,
        k.overflow === 0 && k.mismatch === 0 && shot.cap.width === W && shot.cap.height === H,
        `V ${k.visible}, M ${k.duplicates}, entries ${k.entries}, capacity ${k.capacity}, overflow ${k.overflow}, ` +
        `mismatch ${k.mismatch}, attempts ${shot.attempts}`);
      shots[sg ? 'subgroups' : 'portable'] = { file, counters: k };
    }
    if (shots.subgroups) {
      check(`${c.name}: subgroups and portable images identical`, sameBytes(shots.subgroups.file, shots.portable.file));
    }
    const webPng = shots.portable.file;
    const docName = { train_7k: 'w2_train.png', bonsai_30k: 'w2_bonsai.png', utsuho_plush: 'w2_utsuho.png', grid_512_sh3: 'w2_grid.png' }[c.name];
    if (docName) fs.copyFileSync(webPng, path.join(repo, 'docs', 'web', 'img', docName));

    // Coverage analysis of this real frame (stored spans vs CPU, rows per splat, tail overhead).
    if (c.kind === 'real') {
      const a = await page.evaluate(() => { window.gsrTest.selftest('scene:new'); return window.gsrTest.selftestAsync('scene'); });
      check(`${c.name}: coverage of the real frame (stored spans vs CPU tile_coverage)`, a.pass,
        `${a.visible} visible, ${a.recordsEqualCpu} equal, ${a.borderlineDifferences} borderline, ${a.hardDifferences} hard, ` +
        `${a.exactTilesMissing} exact tiles missing, stored +${(a.storedOverheadPct ?? 0).toFixed(3)}% vs exact` +
        `${a.error ? '; ' + a.error : ''}`);
      shots.coverage = a;
    }

    // CPU reference (the flags of tools/run_reference_checks.ps1) and native splat-tiles.
    const common = ['--scene', c.scene, ...c.camArgs, '--width', `${W}`, '--height', `${H}`];
    const refPfm = path.join(out, `w2_${c.name}_ref.pfm`);
    const ref = runTool(ctx, path.join(bin, 'gsplat_ref.exe'),
      [...common, '--bound', 'tile', '--out', path.join(out, `w2_${c.name}_ref.png`), '--pfm', refPfm], `w2_${c.name}_ref.log`);
    const nativePng = path.join(out, `w2_${c.name}_native.png`);
    await page.evaluate(() => window.gsrTest.setOption('paused', '1'));
    const nat = runTool(ctx, path.join(bin, 'gsplat.exe'),
      [...common, '--renderer', 'splat-tiles', '--splat-bound', 'tile', '--screenshot', nativePng], `w2_${c.name}_native.log`);
    await page.evaluate(() => window.gsrTest.setOption('paused', '0'));
    check(`${c.name}: gsplat_ref and native gsplat ran`, ref.status === 0 && nat.status === 0, `exit ${ref.status} / ${nat.status}`);
    if (ref.status !== 0 || nat.status !== 0) continue;
    const dWeb = imgdiff(ctx, webPng, refPfm, `w2_${c.name}_web_vs_ref`);
    const dNat = imgdiff(ctx, nativePng, refPfm, `w2_${c.name}_native_vs_ref`);
    const dWN = imgdiff(ctx, webPng, nativePng, `w2_${c.name}_web_vs_native`);
    check(`${c.name}: web vs CPU reference >= ${REF_MIN_PSNR} dB`, dWeb.psnr_db >= REF_MIN_PSNR,
      `web ${psnrText(dWeb)} dB (max ${(dWeb.max_abs * 255).toFixed(1)}/255), native ${psnrText(dNat)} dB; ` +
      `web vs native ${psnrText(dWN)} dB, max ${(dWN.max_abs * 255).toFixed(0)}/255, ${dWN.pixels_over_threshold_pct.toFixed(4)}% px > 2/255`);
    rows.push({ name: c.name, kind: c.kind, webPsnr: dWeb.psnr_db, webMax: dWeb.max_abs * 255, nativePsnr: dNat.psnr_db,
      nativeMax: dNat.max_abs * 255, webVsNativePsnr: dWN.psnr_db, webVsNativeMax: dWN.max_abs * 255,
      webVsNativeOverPct: dWN.pixels_over_threshold_pct,
      identicalSubgroups: shots.subgroups ? sameBytes(shots.subgroups.file, shots.portable.file) : null,
      counters: shots.portable.counters, coverage: shots.coverage || null });
  }
  return rows;
}

// --- Performance --------------------------------------------------------------------------------

const PERF_W = 1920, PERF_H = 1080, PERF_FRAMES = 300;

// Native benchmark (same scene, training camera 0, 1920x1080, 300-frame orbit): per-pass GPU ms.
function nativeBenchmark(ctx, scene, name) {
  const json = path.join(ctx.opts.out, `w2_perf_${name}_native.json`);
  const r = runTool(ctx, path.join(ctx.h.repo, 'build', 'bin', 'Release', 'gsplat.exe'),
    ['--scene', scene, '--camera-index', '0', '--width', `${PERF_W}`, '--height', `${PERF_H}`, '--benchmark', json,
      '--frames', `${PERF_FRAMES}`], `w2_perf_${name}_native.log`);
  if (r.status !== 0 || !fs.existsSync(json)) return null;
  const d = JSON.parse(fs.readFileSync(json, 'utf8'));
  const g = (k) => (d.gpu_ms[k] ? d.gpu_ms[k].avg : null);
  return {
    frame: g('frame (total)'), cpuFrame: d.cpu_frame_ms.avg, fps: d.fps_avg,
    stages: {
      preprocess: g('preprocess + cull + tile counts (compute)'), scanFlags: g('scan: visible flags'),
      compact: g('compact (index order)'), depthSort: g('depth sort (total)'), scanTiles: g('scan: tiles touched'),
      setup: g('tile setup (1 thread)'), duplicate: g('duplicate (compute)'), tileSort: g('tile sort (total)'),
      ranges: g('tile ranges (compute)'), render: g('render tiles (compute)'), resolve: g('resolve (blit to rgba8)'),
    },
    entries: d.renderer_settings.entries_last_frame, capacity: d.renderer_settings.list_capacity_entries,
  };
}

const WEB_STAGES = {
  preprocess: 'preprocess', scanFlags: 'scan: visible flags', compact: 'compact (index order)', depthSort: 'depth sort (8 passes)',
  scanTiles: 'scan: tiles touched', setup: 'tile setup (1 thread)', duplicate: 'duplicate', tileSort: 'tile sort',
  ranges: 'tile ranges', render: 'render tiles',
};

async function webPerf(ctx, page, sg) {
  await setSubgroups(page, sg);
  await page.evaluate(() => window.gsrTest.trainingCamera(0));
  await waitFrames(page, 40);  // adaptive list settled, pipelines warm
  await page.evaluate(() => window.gsrTest.timestampsReset());
  // 360-degree orbit over PERF_FRAMES rAF frames (the native --benchmark path): 0.25 deg per pixel.
  const raf = await page.evaluate(async (n) => {
    const t0 = performance.now();
    for (let i = 0; i < n; ++i) {
      if (i > 0) window.gsrTest.orbit((360 / n) / 0.25, 0);
      await window.gsrTest.frames(1);
    }
    return { ms: performance.now() - t0, frames: n };
  }, PERF_FRAMES);
  await waitFrames(page, 4);
  const ts = await page.evaluate(() => window.gsrTest.timestamps());
  const tiles = (await page.evaluate(() => window.gsrTest.stats())).tiles;
  // Throughput: the same orbit, frames submitted back to back (no vsync).
  await page.evaluate(() => window.gsrTest.trainingCamera(0));
  await page.evaluate(() => window.gsrTest.bench(30, 0));  // warm-up
  await page.evaluate(() => window.gsrTest.trainingCamera(0));
  const bench = await page.evaluate((n) => window.gsrTest.bench(n, 360), PERF_FRAMES);
  const pass = (n) => (ts.passes.find((p) => p.name === n) || {}).meanMs ?? null;
  const stages = {};
  for (const [k, n] of Object.entries(WEB_STAGES)) stages[k] = pass(n);
  return { subgroups: sg, frame: pass('frame (first pass begin .. last pass end)'), stages, samples: ts.frames,
    quantized: ts.quantized100us, rafMsPerFrame: raf.ms / raf.frames, benchMsPerFrame: bench.msPerFrame, tiles };
}

async function perfRuns(ctx, page, subgroupsAvailable) {
  const { check, repo } = ctx.h;
  await page.evaluate((w, h) => window.gsrTest.module._gsr_resize(w, h), PERF_W, PERF_H);
  await waitFrames(page, 3);
  const rows = [];
  for (const n of ['train_7k', 'bonsai_30k']) {
    const ply = path.join(repo, 'data', 'scenes', n, 'point_cloud.ply');
    if (!fs.existsSync(ply) || (n === 'bonsai_30k' && ctx.opts.skipBig)) continue;
    const r = await ctx.h.loadByFile(page, ply);
    const camsOk = r.ok && await page.evaluate(async (u) => window.gsrTest.setCamerasJson(await (await fetch(u)).text()),
      `/__testdata/scenes/${n}/cameras.json`);
    check(`perf: load ${n} + cameras.json`, r.ok && camsOk);
    if (!r.ok) continue;
    const row = { name: n, web: [] };
    for (const sg of subgroupsAvailable ? [false, true] : [false]) {
      const p = await webPerf(ctx, page, sg);
      row.web.push(p);
      check(`perf ${n} (${sg ? 'subgroups' : 'portable'}): no coverage mismatch`, p.tiles.mismatchTotal === 0,
        `GPU frame ${p.frame?.toFixed(2)} ms (${p.samples} frames${p.quantized ? ', QUANTIZED' : ''}), ` +
        `back-to-back ${p.benchMsPerFrame.toFixed(2)} ms/frame, rAF ${p.rafMsPerFrame.toFixed(2)} ms/frame, ` +
        `capacity ${p.tiles.capacity}, M ${p.tiles.last?.duplicates}, overflow frames ${p.tiles.overflowFrames}`);
      if (!sg) {
        await page.evaluate(() => window.gsrTest.trainingCamera(0));
        const doc = n === 'train_7k' ? 'w2_train_1080p.png' : 'w2_bonsai_1080p.png';
        await ctx.h.capturePng(page, path.join(repo, 'docs', 'web', 'img', doc));
      }
    }
    // The page must not render while the native benchmark runs (same GPU).
    await page.evaluate(() => window.gsrTest.setOption('paused', '1'));
    row.native = nativeBenchmark(ctx, `data/scenes/${n}/point_cloud.ply`, n);
    await page.evaluate(() => window.gsrTest.setOption('paused', '0'));
    check(`perf: native benchmark ${n}`, !!row.native, row.native ? `GPU frame ${row.native.frame.toFixed(2)} ms` : 'failed');
    rows.push(row);
  }
  await page.evaluate(() => window.gsrTest.module._gsr_resize(960, 540));
  await waitFrames(page, 3);
  return rows;
}

// --- Tables -------------------------------------------------------------------------------------

function printTables(w2) {
  const f = (x, d = 2) => (x === null || x === undefined ? '-' : x >= 999 ? 'inf' : x.toFixed(d));
  if (w2.reference && w2.reference.length) {
    console.log('\n| scene | web vs ref (dB) | web max (/255) | native vs ref (dB) | native max (/255) | web vs native (dB) | max (/255) | px > 2/255 (%) | subgroups = portable | V | M |');
    console.log('|---|---|---|---|---|---|---|---|---|---|---|');
    for (const r of w2.reference) {
      console.log(`| ${r.name} | ${f(r.webPsnr)} | ${f(r.webMax, 1)} | ${f(r.nativePsnr)} | ${f(r.nativeMax, 1)} | ${f(r.webVsNativePsnr)} | ` +
        `${f(r.webVsNativeMax, 0)} | ${f(r.webVsNativeOverPct, 4)} | ` +
        `${r.identicalSubgroups === null ? 'n/a' : r.identicalSubgroups ? 'identical' : 'DIFFERENT'} | ` +
        `${r.counters.visible} | ${r.counters.duplicates} |`);
    }
  }
  if (w2.perf && w2.perf.length) {
    const keys = ['preprocess', 'scanFlags', 'compact', 'depthSort', 'scanTiles', 'setup', 'duplicate', 'tileSort', 'ranges', 'render'];
    for (const r of w2.perf) {
      console.log(`\n${r.name} @ 1920x1080, training camera 0, 300-frame orbit (GPU ms, mean)`);
      const cols = r.web.map((p) => `web ${p.subgroups ? 'subgroups' : 'portable'}`);
      console.log(`| stage | ${cols.join(' | ')} | native |`);
      console.log(`|---|${cols.map(() => '---|').join('')}---|`);
      for (const k of keys) console.log(`| ${k} | ${r.web.map((p) => f(p.stages[k], 3)).join(' | ')} | ${f(r.native?.stages[k], 3)} |`);
      console.log(`| GPU frame (passes) | ${r.web.map((p) => f(p.frame, 3)).join(' | ')} | ${f(r.native?.frame, 3)} |`);
      console.log(`| back-to-back ms/frame | ${r.web.map((p) => f(p.benchMsPerFrame, 3)).join(' | ')} | ${f(r.native?.cpuFrame, 3)} (CPU frame) |`);
    }
  }
}
