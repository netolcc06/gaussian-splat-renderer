// Web entry point (Emscripten): the C API that web/app.js calls, and the frame loop.
// The web counterpart of src/app/main.cpp + Application. Phase W0: WebGPU start-up, the gradient
// test pattern, the resolve to the canvas, capture and self-tests. Phase W1: streamed scene
// loading (PlyStreamDecoder -> WgpuScene), the points renderer, the camera (auto-framing like
// native, orbit/pan/zoom input, exact camera JSON for tests). Phase W2: the splat-tiles renderer
// (WgpuTileRenderer, the default once a scene is loaded; `points` stays selectable), its in-page GPU
// self-tests, GPU timestamps, training cameras (cameras.json) and a throughput benchmark. See
// docs/web/DESIGN.md section 5 and docs/web/W2.md.
//
// Life cycle: app.js instantiates the module, sizes the canvas and calls gsr_init(). The
// adapter and device arrive asynchronously; then every shader variant is compiled once (the
// shader self-test), the pipelines are created and the requestAnimationFrame loop starts.
// gsr_state() turns Ready once the compiler has reported on every module.
//
// Loading (DESIGN.md 2.3): JS reads the file in chunks and, per chunk, copies it into the
// buffer from gsr_load_buffer() and calls gsr_load_push(). The decoder parses the header, then
// decodes whole records in batches; each batch is encoded to the scene format and written to the
// GPU buffers at once. Only the positions (16 B/gaussian, for the bounding box and auto-framing)
// are kept until gsr_load_end(); the file itself is never whole in the wasm heap. The previous
// scene keeps rendering until the new one is complete.
#include "app/camera.h"
#include "app/scene_camera.h"
#include "scene/gaussian_scene.h"
#include "scene/ply_loader.h"
#include "scene/scene_transform.h"
#include "scene/training_cameras.h"
#include "splat/splat_math.h"
#include "webgpu/wgpu_compute.h"
#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_decode_selftest.h"
#include "webgpu/wgpu_gradient.h"
#include "webgpu/wgpu_points.h"
#include "webgpu/wgpu_resolve.h"
#include "webgpu/wgpu_resources.h"
#include "webgpu/wgpu_scene.h"
#include "webgpu/wgpu_shaders.h"
#include "webgpu/wgpu_tile_renderer.h"
#include "webgpu/wgpu_tile_selftest.h"
#include "webgpu/wgsl_preprocess.h"

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <emscripten/html5.h>
#include <malloc.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace gsr;
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Matches gsr_state() in app.js.
enum class AppState : int { Starting = 0, Ready = 1, Failed = -1 };

// Wasm memory: the heap size (linear memory; it only grows) and the bytes malloc has in use.
struct HeapSample {
    uint64_t heapSize = 0;
    uint64_t inUse = 0;
};
HeapSample sampleHeap() {
    const struct mallinfo mi = mallinfo();
    return {uint64_t(emscripten_get_heap_size()), uint64_t(mi.uordblks)};
}

// A scene being streamed in (gsr_load_begin .. gsr_load_end).
struct LoadJob {
    std::string name;
    uint64_t totalBytes = 0;
    std::unique_ptr<PlyStreamDecoder> decoder;
    std::unique_ptr<WgpuScene> scene;
    std::vector<glm::vec4> positions;  // posAlpha of every gaussian: bounding boxes, auto-framing
    Clock::time_point start = Clock::now();
    double pushMs = 0.0;  // inside gsr_load_push (decode + encode + upload)
    uint32_t pushes = 0;
    HeapSample before, peak;
};

// The scene on screen and what is known about it.
struct LoadedScene {
    std::unique_ptr<WgpuScene> gpu;
    GaussianScene info;  // count, degree, comments, warnings, stats (no per-gaussian arrays)
    std::string name;
    UpAxis upAxis = UpAxis::NegY;
    glm::mat4 model{1.0f};
    std::vector<TrainingCamera> trainingCameras;  // gsr_set_cameras_json (cameras.json)
    nlohmann::json report;  // gsr_scene_json()
};

// Which renderer draws a loaded scene (gsr_set_option("renderer", ...)).
enum class RendererKind { SplatTiles, Points };
// Subgroup variants of the splat-tiles shaders: auto = when the device supports them.
enum class SubgroupMode { Auto, On, Off };

struct WebApp {
    WgpuContext context;
    WgslShaderCache shaders;
    WgpuGradientPass gradient;
    WgpuResolve resolve;
    WgpuPointsRenderer points;
    WgpuDecodeSelftest decodeSelftest;
    WgpuTileRenderer tiles;
    WgpuTimestamps timestamps;
    std::unique_ptr<WgpuTileSelftest> tileSelftest;
    WGPUBindGroupLayout sceneLayout = nullptr;
    WgpuTexture2D target;  // rgba16float render target (the native offscreen image)

    AppState state = AppState::Starting;
    bool pipelinesCreated = false;
    std::string failure;
    uint32_t width = 0, height = 0;  // requested canvas size in pixels
    bool resizePending = false;
    uint64_t frames = 0;
    uint32_t surfaceErrors = 0;  // getCurrentTexture failures (not WebGPU errors)
    std::string scratch;         // backing store of strings returned to JS

    // Scene, load, camera, options.
    LoadedScene scene;
    std::unique_ptr<LoadJob> load;
    std::vector<uint8_t> loadBuffer;  // JS copies each chunk here (reused)
    std::string loadError;
    uint32_t loadsCompleted = 0;
    Camera camera = Camera::defaultCamera();
    WgpuSceneOptions sceneOptions;

    // Renderer selection and splat-tiles settings (W2).
    RendererKind renderer = RendererKind::SplatTiles;
    SubgroupMode subgroupMode = SubgroupMode::Auto;
    bool subgroupsAvailable = false;  // device feature + the subgroup_id WGSL builtin
    WgpuTileSettings tileSettings;
    bool tilesDirty = false;          // settings changed: recreate the renderer before the next frame
    bool timestampsOn = true;
    SplatCamera lastCamera;           // the splat camera of the last splat-tiles frame
    uint64_t captureFrame = 0;        // frame index of the last capture
    std::string tilesError;

    // Benchmark (gsr_bench_run): frames submitted back to back, done when the queue drains.
    bool benchRunning = false;
    bool benchDone = false;
    bool paused = false;  // gsr_set_option("paused", "1"): no GPU work (tests: native runs meanwhile)
};

WebApp* gApp = nullptr;

void fail(const std::string& message) {
    gApp->state = AppState::Failed;
    gApp->failure = message;
    std::fprintf(stderr, "gsplat: %s\n", message.c_str());
}

void applySize() {
    WebApp& app = *gApp;
    app.context.configureSurface(app.width, app.height);
    // The render target is what the renderers write (rgba16float, like native). TextureBinding
    // for the resolve's textureLoad, StorageBinding for compute writes, RenderAttachment for
    // raster renderers (points).
    app.target.create(app.context.device(), app.width, app.height, WGPUTextureFormat_RGBA16Float,
                      WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc |
                          WGPUTextureUsage_RenderAttachment,
                      "render target");
    app.resizePending = false;
}

bool subgroupsWanted(const WebApp& app) {
    return app.subgroupsAvailable && app.subgroupMode != SubgroupMode::Off;
}

// (Re)creates the splat-tiles renderer with the current settings (pipelines, layouts); its
// scene-sized buffers follow on the first frame.
void createTiles() {
    WebApp& app = *gApp;
    WgpuTileSettings s = app.tileSettings;
    s.subgroups = subgroupsWanted(app);
    app.tiles.release();
    if (!app.tiles.create(app.context.device(), app.context.queue(), app.shaders, app.sceneLayout,
                          app.context.deviceLimits(), s)) {
        app.tilesError = "cannot create the splat-tiles renderer";
        std::fprintf(stderr, "gsplat: %s\n", app.tilesError.c_str());
    } else {
        app.tilesError.clear();
    }
    app.tilesDirty = false;
}

// The splat camera of the current view (INRIA conventions, splat_math.h), as native's
// SplatTileRenderer::render computes it from the viewer matrices.
SplatCamera currentSplatCamera() {
    WebApp& app = *gApp;
    const float aspect = float(app.target.width) / float(app.target.height);
    return splatCameraFromViewer(app.camera.view(), app.camera.projection(aspect), app.scene.model, app.target.width,
                                 app.target.height);
}

// Records the scene into the render target with the selected renderer (no canvas, no capture).
// Returns true if the splat-tiles renderer recorded the frame (its afterSubmit() is then due).
bool encodeScene(WGPUCommandEncoder encoder, WgpuTimestamps* timestamps) {
    WebApp& app = *gApp;
    WGPUDevice device = app.context.device();
    if (app.renderer == RendererKind::SplatTiles && app.tiles.valid()) {
        app.lastCamera = currentSplatCamera();
        app.tiles.encode(encoder, *app.scene.gpu, app.target, app.lastCamera, app.frames, timestamps);
        return true;
    }
    // Same matrices as the native app: the Vulkan projection; the shader flips clip y.
    const float aspect = float(app.target.width) / float(app.target.height);
    const glm::mat4 mvp = app.camera.projection(aspect) * app.camera.view() * app.scene.model;
    app.points.encode(device, app.context.queue(), encoder, *app.scene.gpu, app.target, mvp);
    return false;
}

bool frame(double /*timeMs*/, void* /*userData*/) {
    WebApp& app = *gApp;
    if (app.context.errors().deviceLost) {
        fail("WebGPU device lost: " + app.context.errors().deviceLostMessage);
        return false;  // stop the loop
    }
    if (app.state == AppState::Starting && app.pipelinesCreated && app.shaders.pendingInfo() == 0) {
        app.state = AppState::Ready;  // every module compiled and reported
        std::printf("gsplat: ready (%ux%u)\n", app.width, app.height);
    }
    if (app.resizePending) applySize();
    if (app.tilesDirty) createTiles();
    if (app.benchRunning || app.paused) return true;  // the benchmark owns the GPU queue until it drains

    WGPUSurfaceTexture surfaceTexture = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(app.context.surface(), &surfaceTexture);
    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (surfaceTexture.texture) wgpuTextureRelease(surfaceTexture.texture);
        ++app.surfaceErrors;
        return true;  // try again next frame
    }
    WGPUTextureView canvasView = wgpuTextureCreateView(surfaceTexture.texture, nullptr);

    WGPUDevice device = app.context.device();
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WgpuTimestamps* timestamps = app.timestampsOn && app.timestamps.enabled() ? &app.timestamps : nullptr;
    if (timestamps) timestamps->beginFrame();
    bool tilesFrame = false;
    if (app.scene.gpu && app.scene.gpu->valid()) {
        tilesFrame = encodeScene(encoder, timestamps);
    } else {
        app.gradient.encode(device, encoder, app.target);  // no scene yet: the W0 test pattern
    }
    if (timestamps) timestamps->endFrame(encoder);
    app.resolve.encodeToCanvas(device, encoder, app.target, canvasView);
    const bool capturing = app.resolve.encodeCapture(device, encoder, app.target);
    if (capturing) app.captureFrame = app.frames;
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(app.context.queue(), 1, &commands);
    if (capturing) app.resolve.startMapping();  // after the submit that fills the buffer
    if (tilesFrame) app.tiles.afterSubmit();
    if (timestamps) timestamps->afterSubmit();

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(canvasView);
    wgpuTextureRelease(surfaceTexture.texture);
    // No wgpuSurfacePresent: the browser presents the canvas when the rAF callback returns.
    ++app.frames;
    return true;
}

void onContextReady(bool ok) {
    WebApp& app = *gApp;
    if (!ok) return fail(app.context.failure());
    const WgpuAdapterDescription& a = app.context.adapterDescription();
    std::printf("gsplat: adapter \"%s\" / \"%s\" (%s, %s), canvas format %d\n", a.vendor.c_str(),
                a.architecture.c_str(), a.description.c_str(), a.adapterType.c_str(),
                int(app.context.surfaceFormat()));

    // Shader self-test: compile every variant once, so that compiler messages for all of them
    // are collected even before the code paths that use them run.
    WGPUDevice device = app.context.device();
    // Subgroup variants need the device feature and the subgroup_id / num_subgroups builtins.
    const auto& wgsl = app.context.wgslLanguageFeatures();
    app.subgroupsAvailable = app.context.hasFeature(WGPUFeatureName_Subgroups) &&
                             std::find(wgsl.begin(), wgsl.end(), "subgroup_id") != wgsl.end();
    for (const WgslVariant& v : webShaderVariants(app.subgroupsAvailable)) app.shaders.get(device, v);

    applySize();
    app.sceneLayout = WgpuScene::createBindGroupLayout(device);
    if (!app.gradient.create(device, app.shaders)) return fail("cannot create the gradient pipeline");
    if (!app.resolve.create(device, app.shaders, app.context.surfaceFormat())) {
        return fail("cannot create the resolve pipelines");
    }
    if (!app.points.create(device, app.shaders, app.sceneLayout)) return fail("cannot create the points renderer");
    createTiles();
    if (!app.tiles.valid()) return fail(app.tilesError);
    app.timestamps.create(device, app.context.hasFeature(WGPUFeatureName_TimestampQuery));
    app.tileSelftest = std::make_unique<WgpuTileSelftest>(
        device, app.context.queue(), app.shaders, app.sceneLayout, app.subgroupsAvailable, [] {
            WgpuTileSelftest::SceneAccess a;
            if (gApp->scene.gpu && gApp->tiles.valid() && gApp->renderer == RendererKind::SplatTiles) {
                a.renderer = &gApp->tiles;
                a.camera = gApp->lastCamera;
                a.width = gApp->target.width;
                a.height = gApp->target.height;
            }
            return a;
        });
    app.pipelinesCreated = true;
    emscripten_request_animation_frame_loop(&frame, nullptr);
}

// --- Scene loading --------------------------------------------------------------------------

void notePeak(LoadJob& job) {
    const HeapSample s = sampleHeap();
    job.peak.heapSize = std::max(job.peak.heapSize, s.heapSize);
    job.peak.inUse = std::max(job.peak.inUse, s.inUse);
}

void abortLoad(const std::string& message) {
    WebApp& app = *gApp;
    app.loadError = message;
    app.load.reset();  // frees the partial GPU scene; the current scene stays
    // stdout, not stderr: a bad file is an expected user error, not a page error (the tests
    // count console errors). JS gets the text from gsr_load_error() and shows it.
    std::printf("gsplat: load failed: %s\n", message.c_str());
}

nlohmann::json vec3Json(const glm::vec3& v) { return {v.x, v.y, v.z}; }

// The job is complete: statistics, scene transform and camera exactly as the native app does
// them for a scene without cameras.json (auto up axis, auto-framing), then the swap.
void installLoadedScene(LoadJob& job) {
    WebApp& app = *gApp;
    const auto tStats = Clock::now();
    const GaussianScene& info = job.decoder->info();
    LoadedScene next;
    next.name = job.name;
    next.info.sourcePath = job.name;
    next.info.count = info.count;
    next.info.shDegree = info.shDegree;
    next.info.comments = info.comments;
    next.info.warnings = info.warnings;
    computePositionStats(job.positions.data(), job.positions.size(), next.info.stats);
    next.info.stats.nonFiniteValues = info.stats.nonFiniteValues;
    next.info.stats.zeroQuaternions = info.stats.zeroQuaternions;
    next.upAxis = defaultUpAxis(info.comments, false);  // cameras.json: gsr_set_cameras_json afterwards
    next.model = sceneModelMatrix(next.upAxis, {});
    const double statsMs = msSince(tStats);
    std::vector<glm::vec4>().swap(job.positions);  // not needed any more
    notePeak(job);
    const HeapSample after = sampleHeap();

    const WgpuScene& gpu = *job.scene;
    const SceneStats& st = next.info.stats;
    next.report = {
        {"name", job.name},
        {"count", info.count},
        {"shDegree", info.shDegree},
        {"storedShDegree", gpu.shDegree},
        {"format", sceneFormatName(gpu.format)},
        {"requestedFormat", sceneFormatName(gpu.requestedFormat)},
        {"layout", sceneLayoutName(gpu.sceneLayout)},
        {"fallback", gpu.fallbackNote},
        {"gpuBytes", gpu.totalBytes},
        {"bytesPerGaussian", info.count ? double(gpu.totalBytes) / info.count : 0.0},
        {"fileBytes", job.decoder->bytesReceived()},
        {"warnings", info.warnings},
        {"comments", info.comments},
        {"upAxis", upAxisName(next.upAxis)},
        {"stats", {{"bboxMin", vec3Json(st.bboxMin)}, {"bboxMax", vec3Json(st.bboxMax)},
                   {"robustMin", vec3Json(st.robustMin)}, {"robustMax", vec3Json(st.robustMax)},
                   {"alphaMean", st.alphaMean}, {"nonFinite", st.nonFiniteValues},
                   {"zeroQuaternions", st.zeroQuaternions}}},
        {"timingsMs", {{"total", msSince(job.start)}, {"insidePush", job.pushMs}, {"header", info.timings.headerMs},
                       {"decode", info.timings.decodeMs}, {"encode", gpu.encodeMs}, {"upload", gpu.uploadMs},
                       {"stats", statsMs}}},
        {"pushes", job.pushes},
        {"batches", gpu.batches},
        {"heap", {{"heapSizeBefore", job.before.heapSize}, {"heapSizeAfter", after.heapSize},
                  {"heapSizePeak", job.peak.heapSize}, {"mallocInUseBefore", job.before.inUse},
                  {"mallocInUsePeak", job.peak.inUse}, {"mallocInUseAfter", after.inUse},
                  {"loadBufferBytes", app.loadBuffer.capacity()}}},
    };
    next.gpu = std::move(job.scene);
    app.scene = std::move(next);  // the old scene's buffers are released here
    app.camera = framedSceneCamera(app.scene.info, app.scene.model);
    ++app.loadsCompleted;
    std::printf("gsplat: loaded %s: %u gaussians, SH %d, %s, %.1f MiB on the GPU, %.0f ms\n", job.name.c_str(),
                info.count, info.shDegree, sceneFormatName(gpu.format), double(gpu.totalBytes) / 1048576.0,
                msSince(job.start));
}

const char* keep(std::string s) {
    gApp->scratch = std::move(s);
    return gApp->scratch.c_str();
}

// --- In-page self-tests (CPU side) ---------------------------------------------------------

nlohmann::json preprocessorSelftest() {
    // In-memory files: include once, nested conditionals, #else, #define inside a file, and
    // the error cases.
    const std::map<std::string, std::string> files = {
        {"main.wgsl", "#include \"a.wgsl\"\n#include \"a.wgsl\"\nfn main() {}\n#ifdef FOO\nfoo\n#ifndef BAR\nfoo_not_bar\n"
                      "#else\nfoo_bar\n#endif\n#else\nnot_foo\n#endif\n#ifdef FROM_A\nfrom_a\n#endif\n"},
        {"a.wgsl", "#define FROM_A\n  #include \"b.wgsl\"\nconst a = 1;"},  // no final newline
        {"b.wgsl", "#include \"a.wgsl\"\nconst b = 2;\n"},                    // cycle: a is already in
        {"bad_endif.wgsl", "#endif\n"},
        {"bad_missing.wgsl", "#ifdef X\n"},
        {"bad_include.wgsl", "#include \"nope.wgsl\"\n"},
        {"bad_directive.wgsl", "#pragma once\n"},
        {"bad_define.wgsl", "#define N 4\n"},
    };
    const WgslFileLoader loader = [&](std::string_view name) -> std::optional<std::string_view> {
        auto it = files.find(std::string(name));
        if (it == files.end()) return std::nullopt;
        return std::string_view(it->second);
    };
    struct Case {
        std::string root;
        std::vector<std::string> defines;
        bool ok;
        std::string expected;  // code if ok, error prefix otherwise
    };
    const Case cases[] = {
        {"main.wgsl", {}, true, "const b = 2;\nconst a = 1;\nfn main() {}\nnot_foo\nfrom_a\n"},
        {"main.wgsl", {"FOO"}, true, "const b = 2;\nconst a = 1;\nfn main() {}\nfoo\nfoo_not_bar\nfrom_a\n"},
        {"main.wgsl", {"FOO", "BAR"}, true, "const b = 2;\nconst a = 1;\nfn main() {}\nfoo\nfoo_bar\nfrom_a\n"},
        {"bad_endif.wgsl", {}, false, "bad_endif.wgsl:1: #endif without"},
        {"bad_missing.wgsl", {}, false, "bad_missing.wgsl:1: missing #endif"},
        {"bad_include.wgsl", {}, false, "bad_include.wgsl:1: cannot find included file \"nope.wgsl\""},
        {"bad_directive.wgsl", {}, false, "bad_directive.wgsl:1: unknown directive #pragma"},
        {"bad_define.wgsl", {}, false, "bad_define.wgsl:1: #define takes one identifier"},
    };
    nlohmann::json details = nlohmann::json::array();
    bool pass = true;
    for (const Case& c : cases) {
        const WgslPreprocessResult r = preprocessWgsl(c.root, c.defines, loader);
        bool ok = r.ok == c.ok;
        if (ok && c.ok) ok = r.code == c.expected;
        if (ok && !c.ok) ok = r.error.rfind(c.expected, 0) == 0;
        // Line map: line 1 of the first case comes from b.wgsl:2.
        if (ok && c.ok) ok = r.describeLine(1) == "b.wgsl:2" && r.lineMap.size() == size_t(std::count(r.code.begin(), r.code.end(), '\n'));
        pass = pass && ok;
        details.push_back({{"root", c.root}, {"defines", c.defines}, {"pass", ok},
                           {"got", c.ok ? r.code : r.error}});
    }
    // The embedded shaders must preprocess too.
    for (const WgslVariant& v : webShaderVariants(gApp->subgroupsAvailable)) {
        const WgslPreprocessResult r = preprocessWgsl(v.file, v.defines, embeddedWgslLoader());
        pass = pass && r.ok;
        details.push_back({{"root", v.label()}, {"pass", r.ok}, {"got", r.ok ? "ok" : r.error}});
    }
    return {{"pass", pass}, {"details", details}};
}

}  // namespace

extern "C" {

// Starts WebGPU on the canvas `canvasSelector` with a backing size of width x height pixels.
EMSCRIPTEN_KEEPALIVE int gsr_init(const char* canvasSelector, int width, int height) {
    if (gApp) return -1;  // once per page
    gApp = new WebApp();  // never freed: lives as long as the page
    gApp->width = uint32_t(std::max(1, width));
    gApp->height = uint32_t(std::max(1, height));
    gApp->context.start(canvasSelector, &onContextReady);
    return 0;
}

// 0 = starting, 1 = ready (rendering), -1 = failed (gsr_last_error()).
EMSCRIPTEN_KEEPALIVE int gsr_state() { return gApp ? int(gApp->state) : 0; }

EMSCRIPTEN_KEEPALIVE const char* gsr_last_error() { return gApp ? gApp->failure.c_str() : ""; }

EMSCRIPTEN_KEEPALIVE void gsr_resize(int width, int height) {
    if (!gApp) return;
    const uint32_t w = uint32_t(std::max(1, width)), h = uint32_t(std::max(1, height));
    if (w == gApp->width && h == gApp->height) return;
    gApp->width = w;
    gApp->height = h;
    gApp->resizePending = gApp->pipelinesCreated;  // before that, the start-up uses the new size
}

// Adapter, features, limits, error counters, frame count, heap (JSON; valid until the next call).
EMSCRIPTEN_KEEPALIVE const char* gsr_stats_json() {
    if (!gApp) return "{}";
    nlohmann::json j = nlohmann::json::parse(gApp->context.describeJson());
    j["state"] = int(gApp->state);
    j["failure"] = gApp->failure;
    j["frames"] = gApp->frames;
    j["width"] = gApp->width;
    j["height"] = gApp->height;
    j["surfaceErrors"] = gApp->surfaceErrors;
    j["shaderErrors"] = gApp->shaders.errorCount();
    j["shaderWarnings"] = gApp->shaders.warningCount();
    const HeapSample heap = sampleHeap();
    j["heap"] = {{"heapSize", heap.heapSize}, {"mallocInUse", heap.inUse}};
    j["sceneLoaded"] = bool(gApp->scene.gpu);
    j["renderer"] = gApp->renderer == RendererKind::SplatTiles ? "splat-tiles" : "points";
    j["subgroupsAvailable"] = gApp->subgroupsAvailable;
    j["subgroupsActive"] = gApp->tiles.valid() && gApp->tiles.settings().subgroups;
    j["timestampsAvailable"] = gApp->timestamps.enabled();
    j["tiles"] = nlohmann::json::parse(gApp->tiles.statsJson());
    j["tilesError"] = gApp->tilesError;
    j["loadsCompleted"] = gApp->loadsCompleted;
    j["loading"] = bool(gApp->load);
    return keep(j.dump());
}

// --- Scene loading (see the file comment) ----------------------------------------------------

// Starts loading a scene file `name` of `totalBytes` bytes (0 = unknown, e.g. gzip). Cancels a
// load in progress. Returns 0, or -1 if the renderer is not ready.
EMSCRIPTEN_KEEPALIVE int gsr_load_begin(const char* name, double totalBytes) {
    if (!gApp || gApp->state != AppState::Ready) return -1;
    WebApp& app = *gApp;
    app.load.reset();
    app.loadError.clear();
    auto job = std::make_unique<LoadJob>();
    job->name = name ? name : "scene.ply";
    job->totalBytes = totalBytes > 0 ? uint64_t(totalBytes) : 0;
    job->before = sampleHeap();
    job->peak = job->before;
    LoadJob* j = job.get();
    auto onHeader = [j, &app](const GaussianScene& info) {
        j->scene = std::make_unique<WgpuScene>();
        std::string error;
        if (!j->scene->create(app.context.device(), app.sceneLayout, app.context.deviceLimits(), info.count,
                              info.shDegree, app.sceneOptions, &error))
            throw std::runtime_error(j->name + ": " + error);
        j->positions.resize(info.count);
    };
    auto onBatch = [j, &app](uint32_t first, const GaussianScene& batch) {
        j->scene->upload(app.context.queue(), first, batch);
        std::copy(batch.posAlpha.begin(), batch.posAlpha.end(), j->positions.begin() + first);
        notePeak(*j);
    };
    job->decoder = std::make_unique<PlyStreamDecoder>(job->name, job->totalBytes, onHeader, onBatch);
    app.load = std::move(job);
    return 0;
}

// A buffer of at least `bytes` bytes for the next chunk (valid until the next call).
EMSCRIPTEN_KEEPALIVE uint8_t* gsr_load_buffer(int bytes) {
    if (!gApp) return nullptr;
    if (gApp->loadBuffer.size() < size_t(std::max(bytes, 1))) gApp->loadBuffer.resize(size_t(std::max(bytes, 1)));
    return gApp->loadBuffer.data();
}

// Decodes and uploads the next `bytes` bytes of the file (in the load buffer). 0 = ok, -1 = the
// load failed (gsr_load_error(); the job is gone, the previous scene stays).
EMSCRIPTEN_KEEPALIVE int gsr_load_push(int bytes) {
    if (!gApp || !gApp->load) return -1;
    LoadJob& job = *gApp->load;
    const auto t = Clock::now();
    try {
        job.decoder->push(std::span<const uint8_t>(gApp->loadBuffer.data(), size_t(std::max(bytes, 0))));
    } catch (const std::exception& e) {
        abortLoad(e.what());
        return -1;
    }
    job.pushMs += msSince(t);
    ++job.pushes;
    notePeak(job);
    return 0;
}

// All bytes pushed: final checks, statistics, camera, swap. 0 = ok, -1 = failed (gsr_load_error()).
EMSCRIPTEN_KEEPALIVE int gsr_load_end() {
    if (!gApp || !gApp->load) return -1;
    LoadJob& job = *gApp->load;
    try {
        job.decoder->finish();
        installLoadedScene(job);
    } catch (const std::exception& e) {
        abortLoad(e.what());
        return -1;
    }
    gApp->load.reset();
    return 0;
}

EMSCRIPTEN_KEEPALIVE void gsr_load_cancel() {
    if (gApp) gApp->load.reset();
}

EMSCRIPTEN_KEEPALIVE const char* gsr_load_error() { return gApp ? gApp->loadError.c_str() : ""; }

// Progress of the load in progress: {"loading", "bytes", "totalBytes", "records", "count"}.
EMSCRIPTEN_KEEPALIVE const char* gsr_load_progress_json() {
    if (!gApp || !gApp->load) return "{\"loading\":false}";
    const LoadJob& job = *gApp->load;
    return keep(nlohmann::json{{"loading", true},
                               {"bytes", job.decoder->bytesReceived()},
                               {"totalBytes", job.totalBytes},
                               {"records", job.decoder->recordsDecoded()},
                               {"count", job.decoder->headerParsed() ? job.decoder->info().count : 0}}
                    .dump());
}

// The scene on screen (name, counts, format, GPU bytes, warnings, stats, load timings, heap).
EMSCRIPTEN_KEEPALIVE const char* gsr_scene_json() {
    if (!gApp || !gApp->scene.gpu) return "{}";
    return keep(gApp->scene.report.dump());
}

// Options: "scene-format" f32|f16|q8 and "scene-sh-degree" 0..3 (next load), "point-size" (px),
// "opacity-threshold" (points renderer), "debug-max-buffer-mb" (tests: lower buffer limit, 0 = off);
// W2: "renderer" splat-tiles|points, "subgroups" auto|on|off, "tile-budget-mb", "tile-list"
// adaptive|fixed, "bounds" tight|rect, "sh-degree" 0..3, "timestamps" on|off, "paused" 0|1.
// 0 = ok, -1 = unknown key or bad value.
EMSCRIPTEN_KEEPALIVE int gsr_set_option(const char* key, const char* value) {
    if (!gApp || !key || !value) return -1;
    const std::string k = key, v = value;
    try {
        if (k == "scene-format") {
            const std::optional<SceneFormat> f = parseSceneFormat(v);
            if (!f || *f == SceneFormat::Codebook) return -1;
            gApp->sceneOptions.format = *f;
        } else if (k == "scene-sh-degree") {
            const int d = std::stoi(v);
            if (d < 0 || d > 3) return -1;
            gApp->sceneOptions.maxShDegree = d;
        } else if (k == "debug-max-buffer-mb") {  // tests: exercise the format fallback
            gApp->sceneOptions.debugMaxBufferBytes = uint64_t(std::stod(v) * 1048576.0);
        } else if (k == "point-size") {
            gApp->points.pointSizePx = std::clamp(std::stof(v), 1.0f, 64.0f);
        } else if (k == "opacity-threshold") {
            gApp->points.opacityThreshold = std::clamp(std::stof(v), 0.0f, 1.0f);
        } else if (k == "renderer") {  // the renderer of a loaded scene
            if (v == "splat-tiles") gApp->renderer = RendererKind::SplatTiles;
            else if (v == "points") gApp->renderer = RendererKind::Points;
            else return -1;
        } else if (k == "subgroups") {  // auto | on | off (off: the portable shaders, the reference)
            if (v == "auto" || v == "on") gApp->subgroupMode = v == "on" ? SubgroupMode::On : SubgroupMode::Auto;
            else if (v == "off") gApp->subgroupMode = SubgroupMode::Off;
            else return -1;
            if (v == "on" && !gApp->subgroupsAvailable) return -1;
            gApp->tilesDirty = true;
        } else if (k == "tile-budget-mb") {
            gApp->tileSettings.budgetMB = uint32_t(std::clamp(std::stoi(v), 1, 4096));
            gApp->tilesDirty = true;
        } else if (k == "tile-list") {  // adaptive | fixed (the whole budget)
            if (v != "adaptive" && v != "fixed") return -1;
            gApp->tileSettings.adaptiveList = v == "adaptive";
            gApp->tilesDirty = true;
        } else if (k == "bounds") {  // tight | rect (identical images; rect bins the whole 3-sigma rectangle)
            if (v != "tight" && v != "rect") return -1;
            gApp->tileSettings.boundsMode = v == "tight" ? 1u : 0u;
            gApp->tilesDirty = true;
        } else if (k == "sh-degree") {  // SH bands evaluated by splat-tiles (<= the stored degree)
            const int d = std::stoi(v);
            if (d < 0 || d > 3) return -1;
            gApp->tileSettings.maxShDegree = d;
            gApp->tilesDirty = true;
        } else if (k == "paused") {  // tests: stop rendering (e.g. while a native benchmark runs)
            gApp->paused = v == "1";
        } else if (k == "timestamps") {
            if (v != "on" && v != "off") return -1;
            gApp->timestampsOn = v == "on";
        } else {
            return -1;
        }
    } catch (const std::exception&) {
        return -1;
    }
    return 0;
}

// --- Camera -----------------------------------------------------------------------------------

// The camera as JSON (the format of --camera files: position, target, yaw/pitch, fov, near, far).
EMSCRIPTEN_KEEPALIVE const char* gsr_camera_json() { return gApp ? keep(gApp->camera.toJson().dump()) : "{}"; }

// Sets the camera from JSON, exactly as `gsplat --camera <file>` reads it. 0 = ok, -1 = invalid.
EMSCRIPTEN_KEEPALIVE int gsr_set_camera_json(const char* json) {
    if (!gApp || !json) return -1;
    try {
        gApp->camera = Camera::fromJson(nlohmann::json::parse(json));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gsplat: bad camera JSON: %s\n", e.what());
        return -1;
    }
    return 0;
}

// Pointer input in CSS pixels / wheel steps, with the native mouse mapping (Application).
EMSCRIPTEN_KEEPALIVE void gsr_input_orbit(float dx, float dy) {
    const float rotateSpeed = 0.25f;  // degrees per pixel
    if (gApp) gApp->camera.orbit(dx * rotateSpeed, -dy * rotateSpeed);
}
EMSCRIPTEN_KEEPALIVE void gsr_input_pan(float dx, float dy) {
    if (!gApp) return;
    const float panScale = gApp->camera.distance * 0.0015f;
    gApp->camera.pan(-dx * panScale, dy * panScale);
}
EMSCRIPTEN_KEEPALIVE void gsr_input_zoom(float steps) {
    if (gApp) gApp->camera.zoom(steps);
}
// Back to the auto-framed view of the scene (or the default camera without a scene).
EMSCRIPTEN_KEEPALIVE void gsr_reset_camera() {
    if (!gApp) return;
    gApp->camera = gApp->scene.gpu ? framedSceneCamera(gApp->scene.info, gApp->scene.model) : Camera::defaultCamera();
}

// --- Capture and self-tests -------------------------------------------------------------------

// Capture of the offscreen image: request, poll the status, then read width*height*4 bytes
// of RGBA8 (top row first) at gsr_capture_data().
EMSCRIPTEN_KEEPALIVE int gsr_capture_request() {
    return gApp && gApp->state == AppState::Ready && gApp->resolve.requestCapture() ? 0 : -1;
}
// 0 none, 1 requested, 2 mapping, 3 ready, -1 failed.
EMSCRIPTEN_KEEPALIVE int gsr_capture_status() { return gApp ? int(gApp->resolve.captureStatus()) : 0; }
EMSCRIPTEN_KEEPALIVE const uint8_t* gsr_capture_data() { return gApp ? gApp->resolve.capturePixels().data() : nullptr; }
EMSCRIPTEN_KEEPALIVE int gsr_capture_width() { return gApp ? int(gApp->resolve.captureWidth()) : 0; }
EMSCRIPTEN_KEEPALIVE int gsr_capture_height() { return gApp ? int(gApp->resolve.captureHeight()) : 0; }

// Frame index of the last capture (gsr_tiles_frame_json(frame) gives that frame's counters).
EMSCRIPTEN_KEEPALIVE double gsr_capture_frame() { return gApp ? double(gApp->captureFrame) : 0.0; }

// In-page self-tests: "shaders" (compiler messages of every variant), "preprocessor", "decode"
// (GPU scene decoder vs the CPU decoder), W2: "scan", "sort", "coverage", "tiles", "image32" and
// "scene" (analysis of the current splat-tiles frame; "scene:new" re-runs it). Asynchronous tests
// return {"pending":true} until done (call again). Returns JSON {"pass": bool, ...}.
EMSCRIPTEN_KEEPALIVE const char* gsr_selftest(const char* name) {
    if (!gApp) return "{\"pass\":false,\"error\":\"not initialized\"}";
    const std::string n = name ? name : "";
    if (n == "shaders") return keep(gApp->shaders.reportJson());
    if (n == "preprocessor") return keep(preprocessorSelftest().dump());
    if (n == "decode") {
        if (gApp->state != AppState::Ready) return "{\"pending\":true}";
        if (!gApp->decodeSelftest.started())
            gApp->decodeSelftest.start(gApp->context.device(), gApp->context.queue(), gApp->shaders, gApp->sceneLayout);
        return keep(gApp->decodeSelftest.resultJson());
    }
    if (n == "scene:new") {  // a fresh analysis of the current frame
        if (gApp->state != AppState::Ready || !gApp->tileSelftest) return "{\"pending\":true}";
        gApp->tileSelftest = std::make_unique<WgpuTileSelftest>(
            gApp->context.device(), gApp->context.queue(), gApp->shaders, gApp->sceneLayout, gApp->subgroupsAvailable,
            [] {
                WgpuTileSelftest::SceneAccess a;
                if (gApp->scene.gpu && gApp->tiles.valid() && gApp->renderer == RendererKind::SplatTiles) {
                    a.renderer = &gApp->tiles;
                    a.camera = gApp->lastCamera;
                    a.width = gApp->target.width;
                    a.height = gApp->target.height;
                }
                return a;
            });
        return keep(gApp->tileSelftest->run("scene"));
    }
    if (WgpuTileSelftest::known(n)) {
        if (gApp->state != AppState::Ready || !gApp->tileSelftest) return "{\"pending\":true}";
        return keep(gApp->tileSelftest->run(n));
    }
    return keep(nlohmann::json{{"pass", false}, {"error", "unknown self-test " + n}}.dump());
}

// --- W2: splat-tiles statistics, timestamps, training cameras, benchmark ---------------------

// The control-block counters of frame `frame` once its asynchronous readback arrived (the last 64
// frames are kept), else {"pending": true}; {"missing": true} if it is older than that history.
EMSCRIPTEN_KEEPALIVE const char* gsr_tiles_frame_json(double frame) {
    if (!gApp) return "{}";
    const uint64_t f = uint64_t(frame);
    const auto s = gApp->tiles.statsForFrame(f);
    if (!s) {
        const auto& last = gApp->tiles.lastStats();
        return last.valid && last.frameIndex > f + 64 ? "{\"missing\":true}" : "{\"pending\":true}";
    }
    const web::WebTileControl& c = s->control;
    return keep(nlohmann::json{{"frame", s->frameIndex},
                               {"gaussians", c.gaussianCount},
                               {"visible", c.visible},
                               {"duplicates", c.duplicates},
                               {"entries", c.entries},
                               {"overflow", c.overflow},
                               {"mismatch", c.coverageMismatch},
                               {"capacity", s->capacity},
                               {"mismatchExample", std::vector<uint32_t>(c.mismatchExample, c.mismatchExample + 3)},
                               {"cullCounts", std::vector<uint32_t>(c.cullCounts, c.cullCounts + 5)}}
                    .dump());
}

// GPU time per pass (timestamp-query), averaged since the last reset.
EMSCRIPTEN_KEEPALIVE const char* gsr_timestamps_json() {
    if (!gApp) return "{}";
    nlohmann::json passes = nlohmann::json::array();
    for (const WgpuTimestamps::PassStats& p : gApp->timestamps.stats())
        passes.push_back({{"name", p.name}, {"meanMs", p.meanMs()}, {"lastMs", p.lastMs}, {"samples", p.samples}});
    return keep(nlohmann::json{{"available", gApp->timestamps.enabled()},
                               {"frames", gApp->timestamps.framesMeasured()},
                               {"quantized100us", gApp->timestamps.quantized()},
                               {"passes", passes}}
                    .dump());
}
EMSCRIPTEN_KEEPALIVE void gsr_timestamps_reset() {
    if (gApp) gApp->timestamps.resetStats();
}

// The training cameras of the scene on screen, from the text of its cameras.json (the native
// loader parses it). As native without --up: the up axis becomes the cameras' mean up, unless the
// PLY says otherwise (defaultUpAxis); the view is re-framed. 0 = ok, -1 = error (stderr).
EMSCRIPTEN_KEEPALIVE int gsr_set_cameras_json(const char* text) {
    if (!gApp || !gApp->scene.gpu || !text) return -1;
    LoadedScene& sc = gApp->scene;
    try {
        const char* path = "/tmp/cameras.json";  // MEMFS: loadTrainingCameras reads a file
        {
            std::ofstream f(path, std::ios::binary);
            f << text;
        }
        sc.trainingCameras = loadTrainingCameras(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gsplat: %s\n", e.what());
        return -1;
    }
    sc.upAxis = defaultUpAxis(sc.info.comments, !sc.trainingCameras.empty());
    sc.model = sceneModelMatrix(sc.upAxis, sc.trainingCameras);
    sc.report["upAxis"] = upAxisName(sc.upAxis);
    sc.report["trainingCameras"] = sc.trainingCameras.size();
    gApp->camera = framedSceneCamera(sc.info, sc.model);
    return int(sc.trainingCameras.size()) > 0 ? 0 : -1;
}

// The view of training camera `index` (native --camera-index). 0 = ok, -1 = no such camera.
EMSCRIPTEN_KEEPALIVE int gsr_training_camera(int index) {
    if (!gApp || index < 0 || size_t(index) >= gApp->scene.trainingCameras.size()) return -1;
    SceneSetup setup;
    setup.trainingCameras = gApp->scene.trainingCameras;
    setup.upAxis = gApp->scene.upAxis;
    setup.model = gApp->scene.model;
    gApp->camera = trainingCameraView(setup, uint32_t(index));
    return 0;
}

// Throughput benchmark: renders `frames` frames back to back into the offscreen target (no canvas,
// no vsync), the camera orbiting `orbitDegTotal` degrees over the run (native --benchmark: 360),
// each frame its own submit; the rAF loop pauses until the queue has drained. JS times it and
// polls gsr_bench_done(). Returns 0, or -1 if no scene is loaded.
EMSCRIPTEN_KEEPALIVE int gsr_bench_run(int frames, double orbitDegTotal) {
    if (!gApp || gApp->state != AppState::Ready || !gApp->scene.gpu || gApp->benchRunning || frames <= 0) return -1;
    WebApp& app = *gApp;
    if (app.tilesDirty) createTiles();
    app.benchRunning = true;
    app.benchDone = false;
    WGPUDevice device = app.context.device();
    for (int i = 0; i < frames; ++i) {
        if (i > 0) app.camera.orbit(float(orbitDegTotal / frames), 0.0f);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        const bool tilesFrame = encodeScene(encoder, nullptr);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(app.context.queue(), 1, &commands);
        wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);
        if (tilesFrame) app.tiles.afterSubmit();
        ++app.frames;
    }
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void*, void*) {
        gApp->benchDone = true;
        gApp->benchRunning = false;
    };
    wgpuQueueOnSubmittedWorkDone(app.context.queue(), cb);
    return 0;
}
EMSCRIPTEN_KEEPALIVE int gsr_bench_done() { return gApp && gApp->benchDone ? 1 : 0; }

}  // extern "C"
