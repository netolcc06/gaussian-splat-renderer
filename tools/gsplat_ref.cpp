// gsplat_ref: CPU reference renderer (INRIA forward pass, see src/splat/reference_renderer.h).
//
// Uses the same scene/camera flags as gsplat and the same camera resolution code
// (app/scene_camera.h), so `gsplat --screenshot` and `gsplat_ref` render the same view:
//
//   gsplat_ref --scene data/scenes/train_7k/point_cloud.ply --camera-index 0 \
//              --width 960 --height 540 --out ref.png --pfm ref.pfm
//
// Outputs: an 8-bit PNG (clamped, rounded) and optionally a float PFM (unclamped) for
// exact diffs with tools/imgdiff. No Vulkan.
#include "app/camera.h"
#include "app/image_io.h"
#include "app/scene_camera.h"
#include "scene/ply_loader.h"
#include "scene/scene_compression.h"
#include "splat/reference_renderer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

namespace {

struct RefOptions {
    std::string scenePath, cameraPath, camerasPath, upAxis;
    std::optional<uint32_t> cameraIndex;
    uint32_t width = 1280, height = 720;
    std::string outPng, outPfm, statsJson;
    gsr::ReferenceOptions render;
    // Phase 5: render the scene as the GPU sees it in a compressed format (encoded, then
    // decoded on the CPU with decodeScene()). Default: the original values.
    gsr::CompressionOptions compression;
    bool compress = false;
    float orbitStep = 0.0f;     // Phase 6: reproduce gsplat --orbit-step (frame i = i orbit steps)
    uint32_t orbitFrames = 0;
};

void printUsage() {
    std::printf(
        "Usage: gsplat_ref --scene <file.ply> --out <out.png> [options]\n"
        "  Scene and camera (same meaning as in gsplat):\n"
        "  --scene <file.ply>  --up <+y|-y|+z|-z|cameras>  --camera <file.json>\n"
        "  --camera-index <i>  --cameras <cameras.json>   --width <px> --height <px> (default 1280x720)\n"
        "  Output:\n"
        "  --out <file.png>        8-bit PNG (clamped to [0,1], rounded)\n"
        "  --pfm <file.pfm>        float image (unclamped), for exact diffs\n"
        "  --stats-json <file>     counts and timings as JSON\n"
        "  Rendering:\n"
        "  --sh-degree <0-3>       highest SH band evaluated (default: the scene's degree)\n"
        "  --bound <tile|radius>   pixels a splat may touch (default tile = INRIA)\n"
        "  --no-early-stop         blend every splat (no T < 1e-4 termination), like\n"
        "                          back-to-front hardware blending\n"
        "  --threads <N>           worker threads (default: all hardware threads)\n"
        "  --sort-window <K|exact> Phase 6 StopThePop-style per-pixel re-sorting window (exact: full\n"
        "                          per-pixel sort by ray depth = the temporal ground truth)\n"
        "  --orbit-step <deg> --orbit-frames <n>  orbit the camera like gsplat --orbit-step for frames 1..n\n"
        "  Scene format (Phase 5; renders the decoded values, i.e. what the GPU sees):\n"
        "  --scene-format <f32|f16|q8|cb>  --scene-layout <soa|split|aos>  --scene-sh-degree <0-3>\n"
        "  --sh-codebook-size <K>  --sh-codebook-iterations <n>  --debug-fp16-positions\n"
        "  --sh-threshold <x>      Phase 6 per-gaussian SH degree (bands below x/255 dropped)\n");
}

bool parseUint(std::string_view text, uint32_t& out) {
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc() && ptr == text.data() + text.size();
}

// Returns false (and prints why) on a bad command line.
bool parseArgs(int argc, char** argv, RefOptions& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        auto fail = [&](const char* msg) {
            std::fprintf(stderr, "error: %s\n\n", msg);
            return false;
        };
        if (arg == "--help" || arg == "-h") return false;
        if (arg == "--no-early-stop") {
            o.render.earlyTermination = false;
            continue;
        }
        if (arg == "--debug-fp16-positions") {
            o.compression.fp16Positions = true;
            o.compress = true;
            continue;
        }
        const char* v = value();
        if (!v) return fail("missing value after a flag");
        uint32_t n = 0;
        if (arg == "--scene") o.scenePath = v;
        else if (arg == "--camera") o.cameraPath = v;
        else if (arg == "--cameras") o.camerasPath = v;
        else if (arg == "--up") o.upAxis = v;
        else if (arg == "--out") o.outPng = v;
        else if (arg == "--pfm") o.outPfm = v;
        else if (arg == "--stats-json") o.statsJson = v;
        else if (arg == "--camera-index") {
            if (!parseUint(v, n)) return fail("--camera-index needs a non-negative integer");
            o.cameraIndex = n;
        } else if (arg == "--width" || arg == "--height" || arg == "--threads") {
            if (!parseUint(v, n) || n == 0) return fail("--width/--height/--threads need a positive integer");
            if (arg == "--width") o.width = n;
            else if (arg == "--height") o.height = n;
            else o.render.threads = n;
        } else if (arg == "--sh-degree") {
            if (!parseUint(v, n) || n > 3) return fail("--sh-degree needs an integer 0..3");
            o.render.maxShDegree = int(n);
        } else if (arg == "--scene-format") {
            const auto f = gsr::parseSceneFormat(v);
            if (!f) return fail("--scene-format must be f32, f16, q8 or cb");
            o.compression.format = *f;
            o.compress = true;
        } else if (arg == "--scene-layout") {
            const auto l = gsr::parseSceneLayout(v);
            if (!l) return fail("--scene-layout must be soa, split or aos");
            o.compression.layout = *l;
            o.compress = true;
        } else if (arg == "--scene-sh-degree") {
            if (!parseUint(v, n) || n > 3) return fail("--scene-sh-degree needs an integer 0..3");
            o.compression.maxShDegree = int(n);
            o.compress = true;
        } else if (arg == "--sh-codebook-size" || arg == "--sh-codebook-iterations") {
            if (!parseUint(v, n) || n == 0) return fail("--sh-codebook-size/-iterations need a positive integer");
            if (arg == "--sh-codebook-size") o.compression.codebookSize = n;
            else o.compression.codebookIterations = n;
        } else if (arg == "--sh-threshold") {
            o.compression.shThreshold = std::stof(v);
            o.compress = true;
        } else if (arg == "--sort-window") {
            if (std::string_view(v) == "exact") o.render.sortWindow = gsr::ReferenceOptions::kExactPixelSort;
            else if (!parseUint(v, n)) return fail("--sort-window needs an integer or 'exact'");
            else o.render.sortWindow = n;
        } else if (arg == "--orbit-step") {
            o.orbitStep = std::stof(v);
        } else if (arg == "--orbit-frames") {
            if (!parseUint(v, n)) return fail("--orbit-frames needs an integer");
            o.orbitFrames = n;
        } else if (arg == "--bounds") {
            const auto b = gsr::parseTileBounds(v);
            if (!b) return fail("--bounds must be rect, tight or cutoff-only");
            o.render.bounds = *b;
        } else if (arg == "--bound") {
            if (std::string_view(v) == "tile") o.render.bound = gsr::CoverageBound::Tile;
            else if (std::string_view(v) == "radius") o.render.bound = gsr::CoverageBound::Radius;
            else return fail("--bound must be tile or radius");
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n\n", argv[i - 1]);
            return false;
        }
    }
    if (o.scenePath.empty() || (o.outPng.empty() && o.outPfm.empty())) {
        std::fprintf(stderr, "error: --scene and --out (or --pfm) are required\n\n");
        return false;
    }
    if (!o.cameraPath.empty() && o.cameraIndex) {
        std::fprintf(stderr, "error: --camera and --camera-index are mutually exclusive\n\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    RefOptions o;
    if (!parseArgs(argc, argv, o)) {
        printUsage();
        return 2;
    }
    try {
        gsr::GaussianScene scene = gsr::loadGaussianPly(o.scenePath);
        std::printf("Scene: %s, %u gaussians, SH degree %d (loaded in %.0f ms)\n", o.scenePath.c_str(), scene.count,
                    scene.shDegree, scene.timings.totalMs);
        if (o.compress) {
            if (o.compression.format != gsr::SceneFormat::F32 && o.compression.layout == gsr::SceneLayout::Soa)
                o.compression.layout = gsr::SceneLayout::Split;
            const gsr::EncodedScene enc = gsr::encodeScene(scene, o.compression);
            std::printf("Scene format %s (%s): %.1f B/gaussian, encoded in %.0f ms; rendering the decoded values\n",
                        gsr::sceneFormatName(enc.format), gsr::sceneLayoutName(enc.layout), enc.bytesPerGaussian(),
                        enc.encodeMs);
            scene = gsr::decodeScene(enc, scene);
        }
        const gsr::SceneSetup setup = gsr::resolveSceneSetup(scene, o.scenePath, o.camerasPath, o.upAxis);
        gsr::Camera camera = gsr::initialSceneCamera(scene, setup, o.cameraPath, o.cameraIndex);
        // gsplat --orbit-step s --frames n renders frame i (0-based) after i orbit steps.
        for (uint32_t i = 0; i < o.orbitFrames; ++i) camera.orbit(o.orbitStep, 0.0f);

        // Exactly the matrices gsplat hands its renderers (Application::recordFrame).
        const float aspect = float(o.width) / float(o.height);
        const gsr::SplatCamera cam =
            gsr::splatCameraFromViewer(camera.view(), camera.projection(aspect), setup.model, o.width, o.height);

        const gsr::ReferenceImage img = gsr::renderReference(scene, cam, o.render);
        const gsr::ReferenceStats& s = img.stats;
        std::printf("Rendered %ux%u with %u threads: preprocess %.1f ms, sort %.1f ms, bin %.1f ms, raster %.1f ms "
                    "(total %.1f ms)\n",
                    img.width, img.height, s.threads, s.preprocessMs, s.sortMs, s.binMs, s.rasterMs, s.totalMs);
        std::printf("  visible %u, culled: near %u, singular %u, off-screen %u; tile instances %llu; "
                    "evaluated fragments %llu; blended fragments %llu (%.1f per pixel); early-stopped pixels %llu\n",
                    s.visible, s.culledNear, s.culledSingular, s.culledOffScreen,
                    static_cast<unsigned long long>(s.tileInstances),
                    static_cast<unsigned long long>(s.evaluatedFragments),
                    static_cast<unsigned long long>(s.blendedFragments),
                    double(s.blendedFragments) / (double(img.width) * img.height),
                    static_cast<unsigned long long>(s.terminatedPixels));

        gsr::ImageRgbF out{img.width, img.height, img.rgb};
        if (!o.outPng.empty()) {
            gsr::writePngRgb(o.outPng, out);
            std::printf("Wrote %s\n", o.outPng.c_str());
        }
        if (!o.outPfm.empty()) {
            gsr::writePfm(o.outPfm, out);
            std::printf("Wrote %s\n", o.outPfm.c_str());
        }
        if (!o.statsJson.empty()) {
            nlohmann::json j = {
                {"scene", o.scenePath},
                {"width", img.width},
                {"height", img.height},
                {"bound", gsr::coverageBoundName(o.render.bound)},
                {"bounds", gsr::tileBoundsName(o.render.bounds)},
                {"early_termination", o.render.earlyTermination},
                {"sh_degree_evaluated", std::min(scene.shDegree, o.render.maxShDegree)},
                {"camera", camera.toJson()},
                {"gaussians", s.gaussians},
                {"visible", s.visible},
                {"culled", {{"near", s.culledNear}, {"singular", s.culledSingular}, {"off_screen", s.culledOffScreen}}},
                {"tile_instances", s.tileInstances},
                {"evaluated_fragments", s.evaluatedFragments},
                {"blended_fragments", s.blendedFragments},
                {"early_stopped_pixels", s.terminatedPixels},
                {"threads", s.threads},
                {"ms", {{"preprocess", s.preprocessMs}, {"sort", s.sortMs}, {"bin", s.binMs}, {"raster", s.rasterMs},
                        {"total", s.totalMs}}},
            };
            const std::filesystem::path p(o.statsJson);
            if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
            std::ofstream(p) << j.dump(2) << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
