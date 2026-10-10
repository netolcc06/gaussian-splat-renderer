// GPU tests for Phase 5 (CTest `scene_formats`), headless Vulkan compute:
//
//   1. Scene formats: for every format x layout, the GPU preprocess (splat-tiles, which
//      decodes the scene with shaders/scene_decode.glsl) gives the same splats as the CPU
//      preprocess (splat_math) on the CPU-decoded scene (decodeScene): same visible set,
//      centre/conic/colour/radius within float tolerance; and the rendered image matches
//      the CPU reference renderer on the decoded scene (> 60 dB before 8-bit quantization).
//      So the GPU decoder is checked against its C++ twin, format by format.
//   2. Packed render splats (`--render-splat p32|p24`) vs the 48-byte Splat of the same
//      frame: p32 centre/conic/opacity bit-exact, colour within the fp16 bound, radius
//      equal; p24 conic/opacity/colour within the fp16 bounds; tile rectangles and depths
//      equal; identical touched-tile counts and identical sorted (tile, splat) lists; image
//      PSNR vs the full mode.
//   3. Adaptive list capacity: a too-small first capacity overflows, grows (no stall) and
//      ends with exactly the fixed-capacity list; a too-large one shrinks after 120 frames.
//
// Exit code: 0 = all checks passed, 1 = a check failed, 2 = no usable GPU / setup error.
#include "core/context.h"
#include "core/profiler.h"
#include "core/resources.h"
#include "render/gpu_scene.h"
#include "render/render_splat.h"
#include "render/splat_tile_renderer.h"
#include "scene/gaussian_scene.h"
#include "scene/half.h"
#include "scene/scene_compression.h"
#include "splat/reference_renderer.h"
#include "splat/splat_math.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace gsr;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

#ifdef NDEBUG
constexpr bool kDebugBuild = false;
#else
constexpr bool kDebugBuild = true;
#endif

constexpr uint32_t W = 640, H = 360;

// SH degree 3, a box of gaussians around the camera (as tests/test_tile_raster.cpp) with
// some very large and some very thin splats (the fp16 conic of p24 is exercised).
GaussianScene makeScene(uint32_t count, uint32_t seed) {
    GaussianScene sc;
    sc.count = count;
    sc.shDegree = 3;
    sc.posAlpha.resize(count);
    sc.scales.resize(count);
    sc.rotations.resize(count);
    sc.sh.resize(size_t(count) * 48);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (uint32_t i = 0; i < count; ++i) {
        glm::vec3 p(u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f);
        float alpha = u(rng);
        if (i % 7 == 0) alpha *= 0.01f;
        if (i % 3 == 0) alpha = 0.95f + 0.04f * u(rng);
        sc.posAlpha[i] = glm::vec4(p, alpha);
        float s = std::exp(u(rng) * 4.0f - 5.5f);
        if (i % 997 == 0) s = 2.0f;  // huge splats (radius of hundreds of pixels)
        glm::vec3 sigma(s, s * (0.2f + u(rng)), s * (0.2f + u(rng)));
        if (i % 13 == 0) sigma.y *= 0.01f;  // needles
        sc.scales[i] = glm::vec4(sigma, 0.0f);
        glm::vec4 q(u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f);
        sc.rotations[i] = glm::normalize(q);
        for (uint32_t k = 0; k < 48; ++k) sc.sh[size_t(i) * 48 + k] = (u(rng) - 0.5f) * (k < 3 ? 2.0f : 0.5f);
    }
    sc.stats.robustMin = glm::vec3(-8.0f);
    sc.stats.robustMax = glm::vec3(8.0f);
    return sc;
}

struct Camera3 {
    glm::mat4 view, proj;
    SplatCamera cam;
};
Camera3 makeCamera() {
    Camera3 c;
    c.view = glm::lookAt(glm::vec3(0.3f, 0.5f, 6.0f), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    c.proj = glm::perspective(glm::radians(60.0f), float(W) / float(H), 0.01f, 100.0f);
    c.proj[1][1] *= -1.0f;
    c.cam = splatCameraFromViewer(c.view, c.proj, glm::mat4(1.0f), W, H);
    return c;
}

struct RunResult {
    std::vector<uint32_t> keys, values, ranges;
    TileControl control{};
    std::vector<float> image;
    std::vector<ProjectedSplat> splats;  // full mode
    std::vector<uint32_t> packed, rects, depths, tileCounts;
    uint32_t capacity = 0, resizes = 0;
    uint64_t overflowFrames = 0;
};

RunResult run(VulkanContext& ctx, const GpuScene& gpu, const TileSettings& ts, uint32_t frames) {
    const Camera3 c = makeCamera();
    GpuProfiler profiler;
    profiler.init(ctx, 1);
    CpuProfiler cpuProfiler;
    Image color = createImage2D(ctx, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                "test: color");
    Image depth{};
    SplatTileRenderer renderer;
    SplatSettings settings;
    settings.format = SplatTargetFormat::Rgba32f;
    renderer.init(ctx, gpu, 1, settings, ts);
    renderer.onRenderTargetsChanged(ctx, color, depth);
    ImageState colorState{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    for (uint64_t frame = 0; frame < frames; ++frame) {
        profiler.beginFrame(0, frame);
        ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const FrameInput in{cmd,    profiler, cpuProfiler, color, depth, colorState,
                                {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE},
                                c.view, c.proj,   glm::mat4(1.0f), frame, 0};
            colorState = renderer.render(in).color;
        });
    }
    RunResult r;
    renderer.collectPending();
    renderer.readBack(r.keys, r.values, r.ranges, r.control);
    renderer.readBackImage(r.image);
    if (ts.renderSplat == RenderSplatMode::Full) renderer.readBackSplats(r.splats, r.tileCounts);
    else renderer.readBackPackedSplats(r.packed, r.rects, r.depths, r.tileCounts);
    r.capacity = renderer.capacity();
    r.resizes = renderer.listResizes();
    r.overflowFrames = renderer.overflowFrames();
    renderer.destroy(ctx);
    destroyImage(ctx, color);
    profiler.destroy();
    return r;
}

double psnr(const std::vector<float>& a, const std::vector<float>& b4orb3, bool bIsRgb) {
    double se = 0.0;
    const size_t pixels = size_t(W) * H;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c) {
            const double x = std::clamp(double(a[p * 4 + c]), 0.0, 1.0);
            const double y = std::clamp(double(b4orb3[p * (bIsRgb ? 3 : 4) + c]), 0.0, 1.0);
            se += (x - y) * (x - y);
        }
    const double mse = se / double(pixels * 3);
    return mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
}

bool relClose(float a, float b, float rel, float absTol) { return std::abs(a - b) <= std::max(absTol, rel * std::abs(b)); }

// ---------------------------------------------------------------------------------------
// 1. Formats
// ---------------------------------------------------------------------------------------
void testFormats(VulkanContext& ctx, const GaussianScene& scene) {
    std::printf("scene formats: GPU decode + preprocess vs CPU decodeScene + preprocess\n");
    const Camera3 c = makeCamera();
    struct Case {
        SceneFormat f;
        SceneLayout l;
        float shThreshold = -1.0f;  // Phase 6: per-gaussian SH degree (>= 0)
    };
    const Case cases[] = {{SceneFormat::F32, SceneLayout::Soa},   {SceneFormat::F32, SceneLayout::Split},
                          {SceneFormat::F32, SceneLayout::Aos},   {SceneFormat::F16, SceneLayout::Split},
                          {SceneFormat::F16, SceneLayout::Aos},   {SceneFormat::Q8, SceneLayout::Split},
                          {SceneFormat::Q8, SceneLayout::Aos},    {SceneFormat::Codebook, SceneLayout::Split},
                          {SceneFormat::Codebook, SceneLayout::Aos},
                          // Phase 6: variable SH degree (compacted soa/split, degree-only aos/cb).
                          {SceneFormat::F32, SceneLayout::Soa, 20.0f}, {SceneFormat::F32, SceneLayout::Split, 20.0f},
                          {SceneFormat::F16, SceneLayout::Split, 20.0f}, {SceneFormat::F16, SceneLayout::Aos, 20.0f},
                          {SceneFormat::Q8, SceneLayout::Split, 20.0f}, {SceneFormat::Codebook, SceneLayout::Split, 20.0f}};
    for (const Case& cs : cases) {
        CompressionOptions o;
        o.format = cs.f;
        o.layout = cs.l;
        o.codebookSize = 256;
        o.shThreshold = cs.shThreshold;
        GpuScene gpu;
        gpu.upload(ctx, scene, o);
        const GaussianScene decoded = decodeScene(encodeScene(scene, o), scene);  // deterministic encode
        const RunResult r = run(ctx, gpu, TileSettings{}, 1);

        uint32_t visibleMismatch = 0, bad = 0, visible = 0;
        double worstColor = 0.0;
        for (uint32_t i = 0; i < scene.count; ++i) {
            const glm::vec4 pa = decoded.posAlpha[i];
            ProjectedSplat cpu;
            bool cpuVisible = false;
            if (viewDepth(c.cam.viewInria, glm::vec3(pa)) > kNearCullDepth && pa.w >= kMinAlpha) {
                std::vector<glm::vec3> coeffs(16);
                for (uint32_t k = 0; k < 16; ++k) coeffs[k] = decoded.shCoeff(i, k);
                cpu = preprocessGaussian(pa, glm::vec3(decoded.scales[i]), decoded.rotations[i], coeffs.data(), 3, c.cam);
                cpuVisible = cpu.cullReason == uint32_t(CullReason::Visible);
            }
            const bool gpuVisible = r.tileCounts[i] != 0;
            visibleMismatch += cpuVisible != gpuVisible;
            if (!gpuVisible || !cpuVisible) continue;
            ++visible;
            const ProjectedSplat& g = r.splats[i];
            bool ok = std::abs(g.center.x - cpu.center.x) < 1e-2f && std::abs(g.center.y - cpu.center.y) < 1e-2f;
            for (int a = 0; a < 3; ++a) ok = ok && relClose(g.conicOpacity[a], cpu.conicOpacity[a], 1e-3f, 1e-6f);
            ok = ok && g.conicOpacity.w == cpu.conicOpacity.w && std::abs(g.radius - cpu.radius) <= 1.0f;
            for (int a = 0; a < 3; ++a) worstColor = std::max(worstColor, double(std::abs(g.color[a] - cpu.color[a])));
            bad += !ok;
        }
        ReferenceOptions ro;
        const ReferenceImage ref = renderReference(decoded, c.cam, ro);
        const double db = psnr(r.image, ref.rgb, true);
        const std::string name = std::string(sceneFormatName(cs.f)) + "/" + sceneLayoutName(cs.l) +
                                 (cs.shThreshold >= 0.0f ? " vsh" : "");
        std::printf("  %-10s visible %u, visibility mismatches %u, splat mismatches %u, worst colour diff %.2e, "
                    "image vs CPU reference on the decoded scene %.2f dB\n",
                    name.c_str(), visible, visibleMismatch, bad, worstColor, db);
        CHECK(visibleMismatch == 0, "%s: %u visibility mismatches", name.c_str(), visibleMismatch);
        CHECK(bad == 0, "%s: %u splats differ", name.c_str(), bad);
        CHECK(worstColor < 1e-4, "%s: colour differs by %.2e", name.c_str(), worstColor);
        CHECK(db > 60.0, "%s: image PSNR %.2f dB", name.c_str(), db);
        gpu.destroy(ctx);
    }
}

// ---------------------------------------------------------------------------------------
// 2. Packed render splats
// ---------------------------------------------------------------------------------------
void testPacked(VulkanContext& ctx, const GaussianScene& scene) {
    std::printf("packed render splats vs the 48-byte Splat\n");
    const Camera3 c = makeCamera();
    GpuScene gpu;
    gpu.upload(ctx, scene);
    const RunResult full = run(ctx, gpu, TileSettings{}, 1);
    for (RenderSplatMode mode : {RenderSplatMode::Packed32, RenderSplatMode::Packed24}) {
        TileSettings ts;
        ts.renderSplat = mode;
        const RunResult r = run(ctx, gpu, ts, 1);
        const bool p24 = mode == RenderSplatMode::Packed24;
        const uint32_t words = p24 ? 6 : 8;
        CHECK(r.tileCounts == full.tileCounts, "%s: touched-tile counts differ", renderSplatModeName(mode));
        uint32_t badExact = 0, badHalf = 0, badRect = 0, badDepth = 0, visible = 0;
        double worstConicRel = 0.0;
        for (uint32_t i = 0; i < scene.count; ++i) {
            if (full.tileCounts[i] == 0) continue;
            ++visible;
            const ProjectedSplat& s = full.splats[i];
            const RenderSplatFields f = decodeRenderSplat(p24, &r.packed[size_t(i) * words]);
            // Bit-exact parts.
            badExact += f.center != s.center || f.radius != std::min(s.radius, 65535.0f);
            if (!p24) badExact += f.conicOpacity != s.conicOpacity;
            // fp16 parts: colour (both), conic and opacity (p24). The GPU's packHalf2x16 may
            // round differently from our RNE encoder: allow one half ulp (2^-10 relative).
            for (int a = 0; a < 3; ++a) badHalf += !relClose(f.color[a], s.color[a], std::ldexp(1.0f, -10), 1e-7f);
            if (p24) {
                for (int a = 0; a < 3; ++a) {
                    const float scaled = std::abs(s.conicOpacity[a]) * kConicScaleP24;
                    const float tol = scaled >= std::ldexp(1.0f, -14) ? scaled * std::ldexp(1.0f, -10)
                                                                       : std::ldexp(1.0f, -24);
                    badHalf += std::abs(f.conicOpacity[a] - s.conicOpacity[a]) * kConicScaleP24 > tol;
                    if (s.conicOpacity[a] != 0.0f)
                        worstConicRel = std::max(worstConicRel, double(std::abs(f.conicOpacity[a] - s.conicOpacity[a]) /
                                                                       std::abs(s.conicOpacity[a])));
                }
                badHalf += !relClose(f.conicOpacity.w, s.conicOpacity.w, std::ldexp(1.0f, -10), 1e-7f);
            }
            const TileRect tr = computeTileRect(s.center, s.radius, c.cam.tilesX, c.cam.tilesY);
            const uint32_t r0 = uint32_t(tr.minX) | (uint32_t(tr.minY) << 16), r1 = uint32_t(tr.maxX) | (uint32_t(tr.maxY) << 16);
            badRect += r.rects[size_t(i) * 2] != r0 || r.rects[size_t(i) * 2 + 1] != r1;
            badDepth += r.depths[i] != floatToBits(s.depth);
        }
        const double db = psnr(r.image, full.image, false);
        std::printf("  %s: %u visible, exact-field mismatches %u, fp16-field mismatches %u, rect %u, depth %u; "
                    "worst conic relative error %.2e; list %s; image vs full %.2f dB\n",
                    renderSplatModeName(mode), visible, badExact, badHalf, badRect, badDepth, worstConicRel,
                    (r.keys == full.keys && r.values == full.values && r.ranges == full.ranges) ? "identical" : "DIFFERENT",
                    db);
        CHECK(badExact == 0, "%s: %u exact fields differ", renderSplatModeName(mode), badExact);
        CHECK(badHalf == 0, "%s: %u fp16 fields outside the bound", renderSplatModeName(mode), badHalf);
        CHECK(badRect == 0 && badDepth == 0, "%s: rects %u, depths %u differ", renderSplatModeName(mode), badRect,
              badDepth);
        CHECK(r.keys == full.keys && r.values == full.values && r.ranges == full.ranges, "%s: list differs",
              renderSplatModeName(mode));
        CHECK(db > (p24 ? 70.0 : 75.0), "%s: image PSNR vs full %.2f dB", renderSplatModeName(mode), db);
        // CPU encoder == GPU encoder for the exact parts (p32 words 0-5).
        if (!p24 && visible > 0) {
            uint32_t w[8];
            uint32_t badWords = 0;
            for (uint32_t i = 0; i < scene.count; ++i) {
                if (full.tileCounts[i] == 0) continue;
                encodeRenderSplat(false, full.splats[i], w);
                for (int k = 0; k < 6; ++k) badWords += w[k] != r.packed[size_t(i) * 8 + k];
                badWords += (w[7] >> 16) != (r.packed[size_t(i) * 8 + 7] >> 16);
            }
            CHECK(badWords == 0, "p32: %u words differ from the CPU encoder", badWords);
        }
    }
    gpu.destroy(ctx);
}

// ---------------------------------------------------------------------------------------
// 3. Adaptive list
// ---------------------------------------------------------------------------------------
void testAdaptive(VulkanContext& ctx, const GaussianScene& scene) {
    std::printf("adaptive list capacity\n");
    GpuScene gpu;
    gpu.upload(ctx, scene);
    const RunResult fixed = run(ctx, gpu, TileSettings{}, 1);
    const uint32_t m = fixed.control.duplicates;

    TileSettings grow;
    grow.adaptiveList = true;
    grow.adaptiveInitialEntries = 16384;  // far too small
    const RunResult g = run(ctx, gpu, grow, 4);
    std::printf("  grow: M = %u, capacity 16384 -> %u after %u resize(s), %llu overflowing frame(s); final list %s\n", m,
                g.capacity, g.resizes, static_cast<unsigned long long>(g.overflowFrames),
                (g.keys == fixed.keys && g.values == fixed.values) ? "identical to fixed" : "DIFFERENT");
    CHECK(g.resizes >= 1 && g.capacity >= m, "grow: capacity %u for M %u", g.capacity, m);
    CHECK(g.overflowFrames >= 1 && g.control.overflow == 0, "grow: overflow frames %llu, last overflow %u",
          static_cast<unsigned long long>(g.overflowFrames), g.control.overflow);
    CHECK(g.keys == fixed.keys && g.values == fixed.values && g.image == fixed.image, "grow: final list/image differ");

    TileSettings shrink;
    shrink.adaptiveList = true;
    shrink.adaptiveInitialEntries = 16u * 1024u * 1024u;  // far too large
    const RunResult s = run(ctx, gpu, shrink, 130);
    std::printf("  shrink: capacity %u -> %u after %u resize(s) (130 frames); final list %s\n", 16u * 1024u * 1024u,
                s.capacity, s.resizes, (s.keys == fixed.keys && s.values == fixed.values) ? "identical" : "DIFFERENT");
    CHECK(s.resizes == 1 && s.capacity < 4u * 1024u * 1024u && s.capacity >= m, "shrink: capacity %u, resizes %u",
          s.capacity, s.resizes);
    CHECK(s.keys == fixed.keys && s.values == fixed.values && s.overflowFrames == 0, "shrink: list differs / overflow");
    gpu.destroy(ctx);
}

}  // namespace

int main(int argc, char** argv) {
    bool validation = kDebugBuild;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--validation") validation = true;
        else if (a == "--no-validation") validation = false;
    }
    VulkanContext ctx;
    try {
        ctx.init(nullptr, validation);
    } catch (const std::exception& e) {
        std::printf("cannot create a headless Vulkan 1.3 device: %s\n", e.what());
        return 2;
    }
    std::printf("Device: %s, validation %s\n", ctx.deviceName.c_str(), ctx.validationEnabled ? "on" : "off");
    try {
        const GaussianScene scene = makeScene(60'000, 21);
        testFormats(ctx, scene);
        testPacked(ctx, scene);
        testAdaptive(ctx, scene);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        ctx.destroy();
        return 2;
    }
    ctx.destroy();
    const auto& vc = VulkanContext::validationCounters();
    std::printf("Validation messages: %u errors, %u warnings\n", vc.errors.load(), vc.warnings.load());
    CHECK(vc.errors == 0 && vc.warnings == 0, "validation messages");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
