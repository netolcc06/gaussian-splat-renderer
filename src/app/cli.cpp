#include "app/cli.h"

#include "scene/scene_transform.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>

namespace gsr {

namespace {

void printUsage() {
    std::printf(
        "Usage: gsplat [options]\n"
        "  --scene <file.ply>        3DGS scene to load (standard INRIA .ply); interactively, scenes can\n"
        "                            also be opened from the UI (Open scene..., Ctrl+O, drag and drop)\n"
        "  --up <+y|-y|+z|-z|cameras> scene direction that points up in the viewer. Default:\n"
        "                            'cameras' (mean training-camera up) if cameras.json exists,\n"
        "                            +y for tools/gen_synthetic_ply.py files, else -y (COLMAP)\n"
        "  --camera <file.json>      initial camera (see README for the format)\n"
        "  --camera-index <i>        start from training camera i of cameras.json\n"
        "  --cameras <cameras.json>  training cameras (default: cameras.json next to the scene)\n"
        "                            Without --camera/--camera-index the scene is auto-framed.\n"
        "  --renderer <name>         'splat-tiles' (default with --scene since Phase 6: compute tile\n"
        "                            rasterizer, front to back), 'splat' (Phase 2: CPU sort + quads),\n"
        "                            'splat-gpusort' (GPU culling + radix sort + indirect draw), 'points', 'test'\n"
        "  --sh-degree <0-3>         splat: highest SH band evaluated (default: the scene's degree)\n"
        "  --splat-format <f>        splat: blend target rgba8 | rgba16f (default) | rgba32f\n"
        "  --splat-bound <b>         splat: quad extent tile (default, INRIA coverage) | radius\n"
        "  --sort-every <N>          splat-gpusort: re-sort every N frames, reusing the order in\n"
        "                            between (default 1 = every frame; 0 = only on --sort-threshold)\n"
        "  --sort-threshold <deg>    splat-gpusort: also re-sort once the camera rotated (or moved\n"
        "                            with a parallax at the scene centre) by more than <deg>\n"
        "  --radix-variant <v>       radix scatter 'classic' | 'reorder' (default: classic for\n"
        "                            splat-gpusort, reorder for splat-tiles)\n"
        "  --tile-sort <m>           splat-tiles: 'binning' (default: depth sort, then stable sort\n"
        "                            by tile) | 'global' (one 64-bit tile|depth sort, as INRIA)\n"
        "  --tile-duplicate <m>      splat-tiles: 'per-entry' (default: one thread per list entry)\n"
        "                            | 'per-splat' (one thread per splat, as INRIA)\n"
        "  --tile-budget-mb <MB>     splat-tiles: memory for the (tile, splat) list (default 256)\n"
        "  --tile-stats <prefix>     splat-tiles: instrumented render; writes <prefix>.json and\n"
        "                            <prefix>_heatmap.png (+ _entries.png, Phase 6) at the end of a run\n"
        "  --tile-stats-max <v>      scale of the _entries.png heatmap (list entries per pixel; 0 = auto)\n"
        "  --render-splat <m>        splat-tiles: per-entry splat record 'full' (48 B, default) | 'p32'\n"
        "                            (32 B, fp16 colour) | 'p24' (24 B, fp16 conic/opacity/colour)\n"
        "  --tile-list <m>           splat-tiles: 'adaptive' (default: follows the measured entry count;\n"
        "                            budget = maximum) | 'fixed' (capacity = --tile-budget-mb)\n"
        "  --scene-format <f>        scene storage on the GPU: 'f32' (default) | 'f16' | 'q8' (SH rest\n"
        "                            8-bit, per-band scales) | 'cb' (SH rest k-means codebook)\n"
        "  --scene-layout <l>        'soa' (f32 only, default for f32) | 'split' (default otherwise:\n"
        "                            geometry + colour records) | 'aos' (one record per gaussian)\n"
        "  --scene-sh-degree <0-3>   drop stored SH bands above this degree at load (memory)\n"
        "  --sh-codebook-size <K>    cb: codebook entries (default 4096, max 65536)\n"
        "  --sh-codebook-iterations <n> cb: k-means iterations (default 10)\n"
        "  --debug-fp16-positions    round positions through fp16 (demonstrates why they stay fp32)\n"
        "  --bounds <b>              Phase 6 binning / quad extent: 'tight' (default: INRIA's 3-sigma tile\n"
        "                            rectangle clipped to the alpha >= 1/255 ellipse: identical image) | 'rect' |\n"
        "                            'cutoff-only' (the ellipse without the 3-sigma clip: not INRIA)\n"
        "  --tile-size <8|16>        splat-tiles: binning and render tile size (default 16)\n"
        "  --sort-window <K>         splat-tiles: StopThePop-style per-pixel re-sorting window (0 = off)\n"
        "  --render-batch <n>        splat-tiles experiment: splats per shared-memory batch\n"
        "  --render-vote-every <n>   splat-tiles experiment: whole-tile exit vote every n batches (0 = never)\n"
        "  --render-shared <soa|aos> splat-tiles experiment: shared-memory layout of the batch\n"
        "  --render-fp16             splat-tiles experiment: exp/alpha in fp16\n"
        "  --sh-threshold <x>        per-gaussian SH degree: drop band l when opacity x its max colour\n"
        "                            contribution is below x/255 (stored compactly; default off)\n"
        "  --pipeline-stats <f.json> create the renderer, write register/shared-memory statistics of its\n"
        "                            compute/graphics pipelines (VK_KHR_pipeline_executable_properties), exit\n"
        "  --sequence <prefix>       with --screenshot: also write every frame as <prefix>_NNNN.png\n"
        "  --orbit-step <deg>        --screenshot: orbit the camera by <deg> per frame\n"
        "  --width <px> --height <px> window / render size (default 1280x720)\n"
        "  --screenshot <out.png>    render --frames frames (default 10), save PNG, exit\n"
        "  --benchmark <out.json>    render --frames frames (default 300) on an orbit path,\n"
        "                            write GPU/CPU timings as JSON, exit\n"
        "  --frames <N>              frame count for --screenshot / --benchmark\n"
        "  --validation              enable Vulkan validation layers (always on in Debug)\n"
        "  --no-vsync                interactive mode: do not wait for vertical sync\n"
        "  --help                    show this text\n"
        "Test / automation of the UI's scene loading (no human needed):\n"
        "  --load-after <N> <f.ply>  after N frames, load <f.ply> through the same code path as the UI's\n"
        "                            Open scene... / drag and drop (worker thread, swap between frames;\n"
        "                            frames rendered while it loads do not count towards --frames)\n"
        "  --debug-fail-scene-install simulate a GPU failure while installing a UI-loaded scene\n");
}

bool parseUint(std::string_view text, uint32_t& out) {
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc() && ptr == text.data() + text.size();
}

bool parseFloat(std::string_view text, float& out) {
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc() && ptr == text.data() + text.size() && std::isfinite(out);
}

}  // namespace

uint32_t Options::frameCount() const {
    if (frames) return *frames;
    if (!benchmarkPath.empty()) return 300;
    return 10;
}

std::optional<Options> parseCommandLine(int argc, char** argv, int& exitCode) {
    Options opt;
    exitCode = 0;
    auto fail = [&](const std::string& message) -> std::optional<Options> {
        std::fprintf(stderr, "error: %s\n\n", message.c_str());
        printUsage();
        exitCode = 2;
        return std::nullopt;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        // Returns the value after a flag, or nullptr if it is missing.
        auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };

        if (arg == "--help" || arg == "-h") {
            printUsage();
            return std::nullopt;
        } else if (arg == "--validation") {
            opt.validation = true;
        } else if (arg == "--no-vsync") {
            opt.vsync = false;
        } else if (arg == "--debug-fp16-positions") {
            opt.debugFp16Positions = true;
        } else if (arg == "--render-fp16") {
            opt.renderFp16 = true;
        } else if (arg == "--debug-fail-scene-install") {
            opt.debugFailSceneInstall = true;
        } else if (arg == "--load-after") {
            const char* n = value();
            const char* file = n ? value() : nullptr;
            if (!n || !parseUint(n, opt.loadAfterFrames) || !file)
                return fail("--load-after needs a frame count and a .ply file");
            opt.loadAfterPath = file;
        } else if (arg == "--bounds" || arg == "--render-shared" || arg == "--pipeline-stats" || arg == "--sequence") {
            const char* v = value();
            if (!v) return fail(std::string(arg) + " needs a value");
            if (arg == "--bounds") opt.bounds = v;
            else if (arg == "--render-shared") opt.renderShared = v;
            else if (arg == "--pipeline-stats") opt.pipelineStatsPath = v;
            else opt.sequencePrefix = v;
        } else if (arg == "--tile-size" || arg == "--sort-window" || arg == "--render-batch" ||
                   arg == "--render-vote-every") {
            const char* v = value();
            uint32_t n = 0;
            if (!v || !parseUint(v, n)) return fail(std::string(arg) + " needs a non-negative integer");
            if (arg == "--tile-size") opt.tileSize = n;
            else if (arg == "--sort-window") opt.sortWindow = n;
            else if (arg == "--render-batch") opt.renderBatch = n;
            else opt.renderVoteEvery = n;
        } else if (arg == "--tile-stats-max") {
            const char* v = value();
            if (!v || !parseFloat(v, opt.tileStatsMax) || opt.tileStatsMax < 0.0f)
                return fail("--tile-stats-max needs a number >= 0");
        } else if (arg == "--sh-threshold") {
            const char* v = value();
            float f = 0.0f;
            if (!v || !parseFloat(v, f) || f < 0.0f) return fail("--sh-threshold needs a number >= 0");
            opt.shThreshold = f;
        } else if (arg == "--scene" || arg == "--camera" || arg == "--renderer" ||
                   arg == "--screenshot" || arg == "--benchmark" || arg == "--cameras" || arg == "--up" ||
                   arg == "--splat-format" || arg == "--splat-bound" || arg == "--radix-variant" ||
                   arg == "--tile-sort" || arg == "--tile-stats" || arg == "--tile-duplicate" ||
                   arg == "--render-splat" || arg == "--tile-list" || arg == "--scene-format" ||
                   arg == "--scene-layout") {
            const char* v = value();
            if (!v) return fail(std::string(arg) + " needs a value");
            if (arg == "--scene") opt.scenePath = v;
            else if (arg == "--camera") opt.cameraPath = v;
            else if (arg == "--cameras") opt.camerasPath = v;
            else if (arg == "--up") opt.upAxis = v;
            else if (arg == "--renderer") {
                opt.renderer = v;
                opt.rendererExplicit = true;
            }
            else if (arg == "--splat-format") opt.splatFormat = v;
            else if (arg == "--splat-bound") opt.splatBound = v;
            else if (arg == "--radix-variant") opt.radixVariant = v;
            else if (arg == "--tile-sort") opt.tileSort = v;
            else if (arg == "--tile-duplicate") opt.tileDuplicate = v;
            else if (arg == "--tile-stats") opt.tileStatsPrefix = v;
            else if (arg == "--render-splat") opt.renderSplat = v;
            else if (arg == "--tile-list") opt.tileList = v;
            else if (arg == "--scene-format") opt.sceneFormat = v;
            else if (arg == "--scene-layout") opt.sceneLayout = v;
            else if (arg == "--screenshot") opt.screenshotPath = v;
            else opt.benchmarkPath = v;
        } else if (arg == "--sort-every") {
            const char* v = value();
            if (!v || !parseUint(v, opt.sortEvery)) return fail("--sort-every needs a non-negative integer");
        } else if (arg == "--sort-threshold" || arg == "--orbit-step") {
            const char* v = value();
            float f = 0.0f;
            if (!v || !parseFloat(v, f) || !(f >= 0.0f || arg == "--orbit-step"))
                return fail(std::string(arg) + " needs a number" + (arg == "--sort-threshold" ? " >= 0" : ""));
            if (arg == "--sort-threshold") opt.sortThresholdDeg = f;
            else opt.orbitStepDeg = f;
        } else if (arg == "--sh-degree" || arg == "--scene-sh-degree") {
            const char* v = value();
            uint32_t n = 0;
            if (!v || !parseUint(v, n) || n > 3) return fail(std::string(arg) + " needs an integer 0..3");
            if (arg == "--sh-degree") opt.shDegree = n;
            else opt.sceneShDegree = n;
        } else if (arg == "--sh-codebook-size" || arg == "--sh-codebook-iterations") {
            const char* v = value();
            uint32_t n = 0;
            if (!v || !parseUint(v, n) || n == 0 || (arg == "--sh-codebook-size" && n > 65536))
                return fail(std::string(arg) + " needs a positive integer" +
                            (arg == "--sh-codebook-size" ? " <= 65536" : ""));
            if (arg == "--sh-codebook-size") opt.codebookSize = n;
            else opt.codebookIterations = n;
        } else if (arg == "--camera-index") {
            const char* v = value();
            uint32_t n = 0;
            if (!v || !parseUint(v, n)) return fail("--camera-index needs a non-negative integer");
            opt.cameraIndex = n;
        } else if (arg == "--tile-budget-mb") {
            const char* v = value();
            if (!v || !parseUint(v, opt.tileBudgetMB) || opt.tileBudgetMB == 0)
                return fail("--tile-budget-mb needs a positive integer");
        } else if (arg == "--frames" || arg == "--width" || arg == "--height") {
            const char* v = value();
            uint32_t n = 0;
            if (!v || !parseUint(v, n) || n == 0)
                return fail(std::string(arg) + " needs a positive integer");
            if (arg == "--frames") opt.frames = n;
            else if (arg == "--width") opt.width = n;
            else opt.height = n;
        } else {
            return fail("unknown argument '" + std::string(arg) + "'");
        }
    }
    if (!opt.screenshotPath.empty() && !opt.benchmarkPath.empty())
        return fail("--screenshot and --benchmark are mutually exclusive");
    if (!opt.cameraPath.empty() && opt.cameraIndex)
        return fail("--camera and --camera-index are mutually exclusive");
    if (opt.cameraIndex && opt.scenePath.empty()) return fail("--camera-index needs --scene");
    if (!opt.upAxis.empty() && !parseUpAxis(opt.upAxis))
        return fail("--up must be one of +y, -y, +z, -z, cameras");
    const bool sceneLater = !opt.loadAfterPath.empty();  // (or, interactively, from the UI)
    if (opt.renderer.empty()) opt.renderer = opt.scenePath.empty() && !sceneLater ? "test" : "splat-tiles";
    if (opt.renderer != "test" && opt.renderer != "points" && opt.renderer != "splat" &&
        opt.renderer != "splat-gpusort" && opt.renderer != "splat-tiles")
        return fail("unknown renderer '" + opt.renderer +
                    "' (available: splat, splat-gpusort, splat-tiles, points, test)");
    // Without a scene the app starts with "test" and switches to the requested renderer
    // once one is loaded (--load-after, or the UI in interactive mode).
    if (opt.renderer != "test" && opt.scenePath.empty() && !sceneLater && opt.automated())
        return fail("--renderer " + opt.renderer + " needs --scene (or --load-after)");
    if (opt.splatFormat != "rgba8" && opt.splatFormat != "rgba16f" && opt.splatFormat != "rgba32f")
        return fail("--splat-format must be rgba8, rgba16f or rgba32f");
    if (opt.splatBound != "tile" && opt.splatBound != "radius") return fail("--splat-bound must be tile or radius");
    if (opt.tileSort != "binning" && opt.tileSort != "global") return fail("--tile-sort must be binning or global");
    if (opt.tileDuplicate != "per-entry" && opt.tileDuplicate != "per-splat")
        return fail("--tile-duplicate must be per-entry or per-splat");
    if (!opt.tileStatsPrefix.empty() && (opt.renderer != "splat-tiles" || !opt.automated()))
        return fail("--tile-stats needs --renderer splat-tiles and --screenshot or --benchmark");
    if (!opt.radixVariant.empty() && opt.radixVariant != "classic" && opt.radixVariant != "reorder")
        return fail("--radix-variant must be classic or reorder");
    if (opt.renderSplat != "full" && opt.renderSplat != "p32" && opt.renderSplat != "p24")
        return fail("--render-splat must be full, p32 or p24");
    if (opt.tileList != "fixed" && opt.tileList != "adaptive") return fail("--tile-list must be fixed or adaptive");
    if (opt.sceneFormat != "f32" && opt.sceneFormat != "f16" && opt.sceneFormat != "q8" && opt.sceneFormat != "cb")
        return fail("--scene-format must be f32, f16, q8 or cb");
    if (!opt.sceneLayout.empty() && opt.sceneLayout != "soa" && opt.sceneLayout != "split" && opt.sceneLayout != "aos")
        return fail("--scene-layout must be soa, split or aos");
    if (opt.sceneLayout == "soa" && opt.sceneFormat != "f32") return fail("--scene-layout soa exists only for f32");
    if (opt.bounds != "rect" && opt.bounds != "tight" && opt.bounds != "cutoff-only")
        return fail("--bounds must be rect, tight or cutoff-only");
    if (opt.bounds == "cutoff-only" && opt.splatBound != "tile") return fail("--bounds cutoff-only needs --splat-bound tile");
    if (opt.tileSize != 8 && opt.tileSize != 16) return fail("--tile-size must be 8 or 16");
    if (opt.sortWindow > 64) return fail("--sort-window must be <= 64");
    if (opt.renderShared != "soa" && opt.renderShared != "aos") return fail("--render-shared must be soa or aos");
    if (!opt.sequencePrefix.empty() && opt.screenshotPath.empty()) return fail("--sequence needs --screenshot");
    if (!opt.pipelineStatsPath.empty() && (!opt.screenshotPath.empty() || !opt.benchmarkPath.empty()))
        return fail("--pipeline-stats is its own mode (no --screenshot/--benchmark)");
    if (sceneLater && !opt.pipelineStatsPath.empty()) return fail("--load-after does not apply to --pipeline-stats");
    if (sceneLater && opt.automated() && opt.loadAfterFrames >= opt.frameCount())
        return fail("--load-after <N> must be smaller than --frames");
    return opt;
}

}  // namespace gsr
