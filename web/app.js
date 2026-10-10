// Page logic for the WebGPU build: WebGPU check, module start-up, canvas sizing, scene loading
// (file picker, drag and drop, URL), pointer camera, and the test hooks used by
// tools/web/run_web_checks.mjs (window.gsrTest).
//
// URL parameters: ?test=1 fixes the canvas backing size (default 960x540, or &w=..&h=..) so
// captures are reproducible; otherwise the canvas follows its CSS size x devicePixelRatio.
// &scene=<url> loads a scene after start-up; &renderer=points|splat-tiles (default splat-tiles);
// &nosubgroups=1 forces the portable shaders.
//
// Loading (docs/web/DESIGN.md 2.3): the file is read in chunks (File.slice for local files, the
// response body stream for URLs, DecompressionStream for .gz) and each chunk is copied into the
// module's load buffer and decoded + uploaded right away (gsr_load_push). The whole file is never
// in the wasm heap. The UI here is minimal; phase W3 builds the real one.

const params = new URLSearchParams(location.search);
const testMode = params.get('test') === '1';
const statusEl = document.getElementById('status');
const CHUNK_BYTES = 8 << 20;  // File.slice size; stream chunks are pushed as they come

// Test hooks. `state`: 'loading' -> 'ready' | 'unsupported' | 'failed'.
const gsrTest = (window.gsrTest = { state: 'loading', reason: '' });
let resolveReady;
gsrTest.ready = new Promise((resolve) => { resolveReady = resolve; });

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

function finish(state, reason = '') {
  gsrTest.state = state;
  gsrTest.reason = reason;
  resolveReady(state);
}

function showUnsupported(reason) {
  document.getElementById('app').hidden = true;
  document.getElementById('unsupported-reason').textContent = reason;
  document.getElementById('unsupported').hidden = false;
  finish('unsupported', reason);
}

// Checked before the wasm is downloaded: no point fetching it without WebGPU.
async function checkWebGPU() {
  if (!('gpu' in navigator)) return 'navigator.gpu is missing (this browser has no WebGPU, or it is disabled)';
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return 'navigator.gpu.requestAdapter() returned no adapter (GPU or driver blocklisted?)';
  } catch (e) {
    return `navigator.gpu.requestAdapter() failed: ${e}`;
  }
  return '';
}

function canvasSize(canvas) {
  if (testMode) {
    return [parseInt(params.get('w') || '960', 10), parseInt(params.get('h') || '540', 10)];
  }
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  let w = Math.max(1, Math.round(rect.width * dpr));
  let h = Math.max(1, Math.round(rect.height * dpr));
  const maxSide = 2560;  // cap the long side (memory and fill rate on high-DPI screens)
  const scale = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.floor(w * scale)), Math.max(1, Math.floor(h * scale))];
}

function base64FromBytes(bytes) {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function formatBytes(n) {
  return n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${(n / 1024).toFixed(0)} KB`;
}

// JS heap (Chrome only; coarse unless --enable-precise-memory-info). ArrayBuffers of file chunks
// are external memory and may not be counted.
function jsHeapBytes() {
  return performance.memory ? performance.memory.usedJSHeapSize : 0;
}

async function main() {
  if (testMode) document.body.classList.add('test-mode');
  const unsupported = await checkWebGPU();
  if (unsupported) return showUnsupported(unsupported);

  setStatus('Loading module...');
  const { default: createGsplatModule } = await import('./gsplat.mjs');
  const Module = await createGsplatModule({
    print: (text) => console.log(text),
    printErr: (text) => console.error(text),
  });
  gsrTest.module = Module;

  const callString = (fn, s, ...rest) => {
    const ptr = Module.stringToNewUTF8(s);
    try { return fn(ptr, ...rest); } finally { Module._free(ptr); }
  };
  const json = (ptr) => JSON.parse(Module.UTF8ToString(ptr));

  // UI callbacks used by the loader; the panel section below replaces these no-ops.
  const ui = { loadStarted() {}, loadProgress() {}, loadFinished() {} };

  // --- Scene loading -------------------------------------------------------------------------
  let loadBusy = Promise.resolve();

  // Pushes the chunks produced by `nextChunk()` (Uint8Array, or null at the end) into the module.
  async function pushScene(name, totalBytes, nextChunk, progressTotal = totalBytes) {
    const t0 = performance.now();
    ui.loadStarted(progressTotal);
    let jsPeak = jsHeapBytes();
    if (callString(Module._gsr_load_begin, name, totalBytes) !== 0) throw new Error('renderer not ready');
    let done = 0;
    let lastStatus = 0;
    for (;;) {
      const chunk = await nextChunk();
      if (!chunk) break;
      for (let off = 0; off < chunk.length; off += CHUNK_BYTES) {
        const part = chunk.subarray(off, Math.min(chunk.length, off + CHUNK_BYTES));
        const ptr = Module._gsr_load_buffer(part.length);
        Module.HEAPU8.set(part, ptr);  // HEAPU8 is re-read: the heap may have grown
        if (Module._gsr_load_push(part.length) !== 0) throw new Error(Module.UTF8ToString(Module._gsr_load_error()));
        done += part.length;
      }
      jsPeak = Math.max(jsPeak, jsHeapBytes());
      const now = performance.now();
      if (now - lastStatus > 100) {
        lastStatus = now;
        const pct = progressTotal ? ` ${Math.min(100, Math.floor((100 * done) / progressTotal))}%` : '';
        setStatus(`Loading ${name}:${pct} ${formatBytes(done)}`);
        ui.loadProgress(done, progressTotal);
      }
    }
    if (Module._gsr_load_end() !== 0) throw new Error(Module.UTF8ToString(Module._gsr_load_error()));
    const scene = json(Module._gsr_scene_json());
    const ms = performance.now() - t0;
    setStatus(`${name}: ${scene.count.toLocaleString()} gaussians, SH ${scene.shDegree}, ${scene.format}` +
              `${scene.fallback ? ' (' + scene.fallback + ')' : ''}, loaded in ${(ms / 1000).toFixed(2)} s`);
    return { scene, ms, bytes: done, jsHeapPeak: jsPeak };
  }

  // A reader over a stream (fetch body or File.stream()), optionally gunzipped.
  function streamReader(stream, gzip) {
    const reader = (gzip ? stream.pipeThrough(new DecompressionStream('gzip')) : stream).getReader();
    return async () => {
      const { value, done } = await reader.read();
      return done ? null : value;
    };
  }

  function loadFile(file) {
    const gzip = /\.gz$/i.test(file.name);
    const name = file.name.replace(/\.gz$/i, '');
    let offset = 0;
    const sliceReader = async () => {
      if (offset >= file.size) return null;
      const end = Math.min(file.size, offset + CHUNK_BYTES);
      const buf = new Uint8Array(await file.slice(offset, end).arrayBuffer());
      offset = end;
      return buf;
    };
    return pushScene(name, gzip ? 0 : file.size, gzip ? streamReader(file.stream(), true) : sliceReader);
  }

  // `knownBytes`: decoded size for the progress bar when the response length is not the decoded
  // length (gzip, or a server applying Content-Encoding).
  async function loadUrl(url, knownBytes = 0) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const path = new URL(url, location.href).pathname;
    const gzip = /\.gz$/i.test(path);
    const name = decodeURIComponent(path.split('/').pop()).replace(/\.gz$/i, '');
    const length = parseInt(response.headers.get('Content-Length') || '0', 10);
    const encoded = response.headers.get('Content-Encoding');
    const total = gzip || encoded ? 0 : length;
    return pushScene(name, total, streamReader(response.body, gzip), total || knownBytes);
  }

  // One load at a time; errors are shown and returned (the previous scene stays).
  function runLoad(start) {
    const job = loadBusy.then(start).then(
      (r) => { ui.loadFinished(true); return { ok: true, ...r }; },
      (e) => {
        Module._gsr_load_cancel();
        const error = String(e && e.message || e);
        console.warn(`load failed: ${error}`);  // a bad file is not a page error
        setStatus(`Load failed: ${error}`, true);
        ui.loadFinished(false);
        return { ok: false, error };
      });
    loadBusy = job;
    return job;
  }
  gsrTest.loadUrl = (url) => runLoad(() => loadUrl(url));
  gsrTest.loadFile = (file) => runLoad(() => loadFile(file));
  gsrTest.scene = () => json(Module._gsr_scene_json());

  // --- Hooks for the tests ---------------------------------------------------------------------
  gsrTest.stats = () => json(Module._gsr_stats_json());
  gsrTest.selftest = (name) => json(callString(Module._gsr_selftest, name));
  gsrTest.selftestAsync = (name) => new Promise((resolve) => {
    const poll = () => {
      const r = gsrTest.selftest(name);
      if (r.pending) requestAnimationFrame(poll); else resolve(r);
    };
    poll();
  });
  gsrTest.camera = () => json(Module._gsr_camera_json());
  gsrTest.setCamera = (cam) => callString(Module._gsr_set_camera_json, typeof cam === 'string' ? cam : JSON.stringify(cam)) === 0;
  gsrTest.setOption = (key, value) => {
    const k = Module.stringToNewUTF8(key);
    try { return callString((v) => Module._gsr_set_option(k, v), String(value)) === 0; } finally { Module._free(k); }
  };
  gsrTest.orbit = (dx, dy) => Module._gsr_input_orbit(dx, dy);
  // W2: splat-tiles counters of a frame (after its asynchronous readback), GPU pass timestamps,
  // training cameras, the throughput benchmark.
  gsrTest.captureFrame = () => Module._gsr_capture_frame();
  gsrTest.tilesFrame = (frame) => json(Module._gsr_tiles_frame_json(frame));
  gsrTest.tilesFrameAsync = (frame) => new Promise((resolve) => {
    const poll = () => {
      const r = gsrTest.tilesFrame(frame);
      if (r.pending) requestAnimationFrame(poll); else resolve(r);
    };
    poll();
  });
  gsrTest.timestamps = () => json(Module._gsr_timestamps_json());
  gsrTest.timestampsReset = () => Module._gsr_timestamps_reset();
  gsrTest.setCamerasJson = (text) => callString(Module._gsr_set_cameras_json, text) === 0;
  gsrTest.trainingCamera = (i) => Module._gsr_training_camera(i) === 0;
  // Renders `frames` frames back to back (no vsync), orbiting `orbitDeg` degrees in total, and
  // resolves with the wall time once the GPU queue has drained.
  gsrTest.bench = (frames, orbitDeg = 360) => new Promise((resolve, reject) => {
    const t0 = performance.now();
    if (Module._gsr_bench_run(frames, orbitDeg) !== 0) return reject(new Error('benchmark refused'));
    const poll = () => {
      if (Module._gsr_bench_done()) {
        const ms = performance.now() - t0;
        resolve({ frames, ms, msPerFrame: ms / frames });
      } else {
        setTimeout(poll, 0);
      }
    };
    setTimeout(poll, 0);
  });
  gsrTest.resetCamera = () => Module._gsr_reset_camera();
  gsrTest.frames = (n) => new Promise((resolve) => {
    const f = () => (--n > 0 ? requestAnimationFrame(f) : resolve());
    requestAnimationFrame(f);
  });
  // Captures the offscreen image of the next frame: {width, height, base64} (RGBA8, top row first).
  gsrTest.capture = () => new Promise((resolve, reject) => {
    if (Module._gsr_capture_request() !== 0) return reject(new Error('capture request refused'));
    const poll = () => {
      const status = Module._gsr_capture_status();
      if (status === 3) {
        const w = Module._gsr_capture_width();
        const h = Module._gsr_capture_height();
        const ptr = Module._gsr_capture_data();
        resolve({ width: w, height: h, base64: base64FromBytes(Module.HEAPU8.subarray(ptr, ptr + w * h * 4)) });
      } else if (status < 0) {
        reject(new Error('capture failed'));
      } else {
        requestAnimationFrame(poll);
      }
    };
    requestAnimationFrame(poll);
  });

  // --- Canvas, input -------------------------------------------------------------------------
  const canvas = document.getElementById('canvas');
  const [w, h] = canvasSize(canvas);
  callString(Module._gsr_init, '#canvas', w, h);
  if (!testMode) {
    new ResizeObserver(() => {
      const [nw, nh] = canvasSize(canvas);
      Module._gsr_resize(nw, nh);
    }).observe(canvas);
  }

  // --- Camera input --------------------------------------------------------------------------
  // Mouse: left drag orbits, right/middle (or shift) drag pans, wheel zooms, double click resets.
  // Touch: one finger orbits, two fingers pan (centroid) and pinch-zoom. Deltas in CSS pixels.
  let demoCamera = null;  // the curated view of the demo scene (reset returns to it)
  const resetCamera = () => {
    if (demoCamera) gsrTest.setCamera(demoCamera); else Module._gsr_reset_camera();
  };
  const pointers = new Map();
  let mousePan = false;
  const pinchState = () => {
    const [a, b] = [...pointers.values()];
    return { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
  };
  let pinch = null;
  canvas.addEventListener('pointerdown', (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    mousePan = e.pointerType === 'mouse' && (e.button !== 0 || e.shiftKey);
    pinch = pointers.size === 2 ? pinchState() : null;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 1) {
      if (mousePan) Module._gsr_input_pan(dx, dy); else Module._gsr_input_orbit(dx, dy);
    } else if (pointers.size === 2 && pinch) {
      const now = pinchState();
      Module._gsr_input_pan(now.cx - pinch.cx, now.cy - pinch.cy);
      if (pinch.d > 0 && now.d > 0) Module._gsr_input_zoom(Math.log(now.d / pinch.d) / Math.log(1.15));
      pinch = now;
    }
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    pinch = pointers.size === 2 ? pinchState() : null;
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const lines = e.deltaMode === 1 ? e.deltaY : e.deltaY / 100;  // ~1 step per notch
    Module._gsr_input_zoom(-lines);
  }, { passive: false });
  canvas.addEventListener('dblclick', resetCamera);

  // --- Panel ---------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const progressEl = $('progress');
  const loadButtons = [$('demo-knock'), $('demo-grid')];
  ui.loadStarted = (total) => {
    progressEl.hidden = false;
    progressEl.classList.toggle('indeterminate', !total);
    $('progress-bar').style.width = '0';
    loadButtons.forEach((b) => { b.disabled = true; });
  };
  ui.loadProgress = (done, total) => {
    if (total) $('progress-bar').style.width = `${Math.min(100, (100 * done) / total)}%`;
  };
  ui.loadFinished = () => {
    progressEl.hidden = true;
    loadButtons.forEach((b) => { b.disabled = false; });
  };

  const cameraRow = $('row-camera');
  const cameraSelect = $('opt-camera');
  function setTrainingCameras(text) {
    cameraRow.hidden = true;
    cameraSelect.innerHTML = '';
    if (!text || !gsrTest.setCamerasJson(text)) return;
    let count = 0;
    try { count = JSON.parse(text).length || 0; } catch { count = 0; }
    if (!count) return;
    const free = document.createElement('option');
    free.value = '-1';
    free.textContent = 'Free camera';
    cameraSelect.append(free);
    for (let i = 0; i < count; ++i) {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = `#${i}`;
      cameraSelect.append(o);
    }
    cameraRow.hidden = false;
  }
  cameraSelect.addEventListener('change', () => {
    const i = parseInt(cameraSelect.value, 10);
    if (i >= 0) gsrTest.trainingCamera(i); else resetCamera();
  });

  // Applies the panel's settings to a freshly loaded scene.
  function afterLoad(result, { credit = false, camera = null } = {}) {
    if (!result.ok) return result;
    demoCamera = camera;
    if (camera) gsrTest.setCamera(camera);
    $('credit-knock').hidden = !credit;
    gsrTest.setOption('sh-degree', $('opt-sh').value);
    return result;
  }

  async function loadDemoKnock() {
    const camText = await fetch('demo/knock-community-hall.camera.json').then((r) => (r.ok ? r.text() : null)).catch(() => null);
    setTrainingCameras(null);
    const r = await gsrTest.loadUrl('demo/knock-community-hall_300k.ply.gz', 27600912);
    return afterLoad(r, { credit: true, camera: camText });
  }
  async function loadDemoGrid() {
    setTrainingCameras(null);
    return afterLoad(await gsrTest.loadUrl('demo/grid_512_sh3.ply'));
  }
  async function loadFiles(files) {
    const list = [...files];
    const ply = list.find((f) => /\.ply(\.gz)?$/i.test(f.name));
    const cams = list.find((f) => /\.json$/i.test(f.name));
    if (!ply) {
      setStatus('Choose a .ply (or .ply.gz) file, optionally together with its cameras.json', true);
      return { ok: false };
    }
    const r = afterLoad(await gsrTest.loadFile(ply));
    if (r.ok) setTrainingCameras(cams ? await cams.text() : null);
    return r;
  }
  gsrTest.loadDemo = (name) => (name === 'grid' ? loadDemoGrid() : loadDemoKnock());

  $('demo-knock').addEventListener('click', () => { gsrTest.lastLoad = loadDemoKnock(); });
  $('demo-grid').addEventListener('click', () => { gsrTest.lastLoad = loadDemoGrid(); });
  const fileInput = $('file-input');
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) gsrTest.lastLoad = loadFiles(fileInput.files);
    fileInput.value = '';
  });
  const dropHint = $('drop-hint');
  window.addEventListener('dragover', (e) => { e.preventDefault(); dropHint.hidden = false; });
  window.addEventListener('dragleave', (e) => { if (!e.relatedTarget) dropHint.hidden = true; });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dropHint.hidden = true;
    if (e.dataTransfer.files.length) gsrTest.lastLoad = loadFiles(e.dataTransfer.files);
  });

  $('opt-renderer').addEventListener('change', (e) => gsrTest.setOption('renderer', e.target.value));
  $('opt-sh').addEventListener('change', (e) => gsrTest.setOption('sh-degree', e.target.value));
  $('reset-camera').addEventListener('click', resetCamera);
  $('panel-toggle').addEventListener('click', () => {
    const panel = $('toolbar');
    const collapsed = panel.classList.toggle('collapsed');
    $('panel-toggle').innerHTML = collapsed ? '&#x2b;' : '&#x2212;';
    $('panel-toggle').setAttribute('aria-expanded', String(!collapsed));
  });

  // Stats: FPS from requestAnimationFrame, the rest from the module (twice a second).
  let fpsFrames = 0;
  let fpsT0 = performance.now();
  let fps = 0;
  const countFrame = () => { ++fpsFrames; requestAnimationFrame(countFrame); };
  requestAnimationFrame(countFrame);
  function updateStats() {
    const now = performance.now();
    fps = (1000 * fpsFrames) / (now - fpsT0);
    fpsFrames = 0;
    fpsT0 = now;
    const s = gsrTest.stats();
    const rows = [['FPS', fps.toFixed(0)], ['Resolution', `${s.width} x ${s.height}`]];
    if (s.sceneLoaded) {
      const sc = gsrTest.scene();
      rows.push(['Gaussians', (sc.count || 0).toLocaleString()], ['SH stored', String(sc.shDegree)], ['Scene format', sc.format]);
      rows.push(['Renderer', s.renderer]);
      if (s.renderer === 'splat-tiles') rows.push(['Subgroups', s.subgroupsActive ? 'on' : 'off (portable)']);
      const ts = gsrTest.timestamps();
      if (ts.available && ts.passes && ts.passes.length) {
        const total = ts.passes.reduce((sum, p) => sum + (p.meanMs || 0), 0);
        rows.push(['GPU time', `${total.toFixed(2)} ms${ts.quantized100us ? ' (~0.1 ms res.)' : ''}`]);
      }
      if (s.tilesError) rows.push(['Renderer error', s.tilesError]);
    }
    const a = s.adapter || {};
    rows.push(['GPU', `${a.vendor || ''} ${a.architecture || ''}`.trim() || 'unknown']);
    $('stats').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v).replace(/</g, '&lt;')}</dd>`).join('');
  }

  setStatus('Starting WebGPU...');
  const waitReady = () => {
    const state = Module._gsr_state();
    if (state === 1) {
      const a = gsrTest.stats().adapter;
      setStatus(`WebGPU: ${a.vendor} ${a.architecture} ${a.description}`.trim() + ' - open a .ply file');
      document.getElementById('toolbar').hidden = false;
      // ?nosubgroups=1: force the portable shaders (what Firefox and Safari run) in Chrome.
      if (params.get('nosubgroups') === '1') gsrTest.setOption('subgroups', 'off');
      if (params.get('renderer')) {
        gsrTest.setOption('renderer', params.get('renderer'));
        $('opt-renderer').value = params.get('renderer');
      }
      if (!testMode && gsrTest.stats().timestampsAvailable) gsrTest.setOption('timestamps', 'on');
      if (!testMode) setInterval(updateStats, 500);
      updateStats();
      finish('ready');
      const scene = params.get('scene');
      if (scene) gsrTest.lastLoad = gsrTest.loadUrl(scene);
      else if (!testMode && params.get('demo') !== '0') gsrTest.lastLoad = loadDemoKnock();
    } else if (state < 0) {
      const reason = Module.UTF8ToString(Module._gsr_last_error());
      setStatus(`WebGPU error: ${reason}`, true);
      finish('failed', reason);
    } else {
      requestAnimationFrame(waitReady);
    }
  };
  requestAnimationFrame(waitReady);
}

main().catch((e) => {
  console.error(e);
  setStatus(`Error: ${e.message || e}`, true);
  finish('failed', String(e && e.stack || e));
});
