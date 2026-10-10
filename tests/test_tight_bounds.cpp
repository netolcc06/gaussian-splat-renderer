// Phase 6 tests (CTest `tight_bounds`): opacity-aware tight tile bounds, smaller tiles and
// the StopThePop window.
//
//   CPU (no GPU needed):
//   1. Coverage geometry on random splats (incl. needles, tiny/huge, near-transparent):
//      tight tiles are a subset of INRIA's rectangle, and EVERY pixel of every dropped tile has
//      splatAlphaAt() == 0 (the exactness claim, pixel by pixel); every pixel with alpha > 0
//      lies in a kept tile; count == number of enumerated tiles; 8-px bins likewise.
//   2. The CPU reference renderer with tight bounds produces a BIT-IDENTICAL float image and
//      the same blended/terminated counts as with the rectangle (synthetic scene).
//   GPU (headless Vulkan, exit 2 without a GPU):
//   3. splat-tiles with --bounds tight / cutoff-only, --tile-size 8|16, binning/global sort,
//      per-entry/per-splat duplication, full/p32/p24 render splats:
//      - per-gaussian counts == the CPU coverage count of the GPU's own splats;
//      - the sorted (tile, splat) list == a CPU construction from those counts' tiles;
//      - control.coverageMismatch == 0 (count pass and write pass agree);
//      - tight images are bit-identical (rgba32f) to the rect image of the same mode;
//      - cutoff-only matches the CPU reference with cutoff-only bounds (float PSNR).
//   4. StopThePop window K = 4 and 16: GPU image vs the CPU reference with the same window.
// Splats whose coverage decision lies within 2e-3 px of a pixel boundary
// (coverageIsBorderline) may legitimately differ between GPU and CPU rounding of sqrt and
// division; they are excluded from the exact list comparison and counted (expected ~0).
#include "core/context.h"
#include "core/profiler.h"
#include "core/resources.h"
#include "render/gpu_scene.h"
#include "render/render_splat.h"
#include "render/splat_tile_renderer.h"
#include "scene/gaussian_scene.h"
#include "splat/reference_renderer.h"
#include "splat/splat_math.h"
#include "splat/tile_coverage.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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

uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// ---------------------------------------------------------------------------------------
// 1. Coverage geometry, pixel by pixel
// ---------------------------------------------------------------------------------------
void testCoverageGeometry() {
    std::printf("Tight coverage vs per-pixel alpha (CPU)\n");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const uint32_t W = 400, H = 300;
    const uint32_t tx16 = (W + 15) / 16, ty16 = (H + 15) / 16;
    uint64_t splatsTested = 0, rectTiles = 0, tightTiles[2] = {0, 0}, droppedPixelsWithAlpha = 0,
             alphaPixelsOutside = 0, countMismatch = 0, notSubset = 0, tight8vs16 = 0;
    for (int n = 0; n < 4000; ++n) {
        ProjectedSplat s;
        s.center = {u(rng) * 460.0f - 30.0f, u(rng) * 360.0f - 30.0f};
        // Random 2D covariance: eigenvalues from 0.3 (the dilation floor) to ~10^4, any angle,
        // so the conic ranges from round to extreme needles.
        const float l1 = 0.3f + std::pow(10.0f, u(rng) * 4.0f), l2 = 0.3f + std::pow(10.0f, u(rng) * 2.0f) * u(rng);
        const float th = u(rng) * 6.2831853f, c = std::cos(th), sn = std::sin(th);
        const glm::vec3 cov(c * c * l1 + sn * sn * l2, c * sn * (l1 - l2), sn * sn * l1 + c * c * l2);
        glm::vec3 conic;
        float radius = 0.0f;
        if (!conicAndRadius(cov, conic, radius)) continue;
        const float o = n % 5 == 0 ? 1.0f / 255.0f + u(rng) * 0.01f : (n % 7 == 0 ? 0.99f + 0.01f * u(rng) : u(rng));
        s.conicOpacity = glm::vec4(conic, o);
        s.radius = radius;
        const TileRect rect = computeTileRect(s.center, s.radius, tx16, ty16);
        if (rect.empty()) continue;
        ++splatsTested;
        rectTiles += uint64_t(rect.maxX - rect.minX) * uint64_t(rect.maxY - rect.minY);
        for (int binIdx = 0; binIdx < 2; ++binIdx) {
            const int S = binIdx == 0 ? 16 : 8;
            const int gx = int((W + S - 1) / S), gy = int((H + S - 1) / S);
            const Coverage cv = makeCoverage(TileBounds::Tight, rect, S, gx, gy, s.center, conic, o);
            std::vector<uint8_t> kept(size_t(gx) * gy, 0);
            const auto [r0, r1] = coverageRows(cv, S);
            uint32_t enumerated = 0;
            for (int r = r0; r < r1; ++r) {
                const auto [x0, x1] = coverageRowSpan(cv, r, S);
                for (int x = x0; x < x1; ++x) {
                    kept[size_t(r) * gx + x] = 1;
                    ++enumerated;
                }
            }
            tightTiles[binIdx] += enumerated;
            countMismatch += enumerated != coverageCount(cv, S);
            // Every pixel of the INRIA rectangle: alpha > 0 => its bin tile is kept.
            for (int py = rect.minY * 16; py < std::min(rect.maxY * 16, int(H)); ++py)
                for (int px = rect.minX * 16; px < std::min(rect.maxX * 16, int(W)); ++px) {
                    const bool alpha = splatAlphaAt(s, float(px), float(py)) > 0.0f;
                    const bool k = kept[size_t(py / S) * gx + px / S] != 0;
                    if (alpha && !k) ++droppedPixelsWithAlpha;
                }
            // Kept tiles must lie inside the rectangle (in bin units).
            const BinRect b = rectToBins(rect, S, gx, gy);
            for (int y = 0; y < gy; ++y)
                for (int x = 0; x < gx; ++x)
                    if (kept[size_t(y) * gx + x] && (x < b.x0 || x >= b.x1 || y < b.y0 || y >= b.y1)) ++notSubset;
            if (S == 8) {
                // An 8-px tile kept => its 16-px parent kept by the 16-px tight coverage.
                const Coverage c16 = makeCoverage(TileBounds::Tight, rect, 16, int(tx16), int(ty16), s.center, conic, o);
                std::vector<uint8_t> k16(size_t(tx16) * ty16, 0);
                const auto [q0, q1] = coverageRows(c16, 16);
                for (int r = q0; r < q1; ++r) {
                    const auto [x0, x1] = coverageRowSpan(c16, r, 16);
                    for (int x = x0; x < x1; ++x) k16[size_t(r) * tx16 + x] = 1;
                }
                for (int y = 0; y < gy; ++y)
                    for (int x = 0; x < gx; ++x)
                        if (kept[size_t(y) * gx + x] && !k16[size_t(y / 2) * tx16 + x / 2]) ++tight8vs16;
            }
        }
        // cutoff-only: every pixel of the screen with alpha > 0 is covered (no 3-sigma clip).
        {
            const Coverage cv = makeCoverage(TileBounds::CutoffOnly, rect, 16, int(tx16), int(ty16), s.center, conic, o);
            std::vector<uint8_t> kept(size_t(tx16) * ty16, 0);
            const auto [r0, r1] = coverageRows(cv, 16);
            for (int r = r0; r < r1; ++r) {
                const auto [x0, x1] = coverageRowSpan(cv, r, 16);
                for (int x = x0; x < x1; ++x) kept[size_t(r) * tx16 + x] = 1;
            }
            for (uint32_t py = 0; py < H; py += 3)
                for (uint32_t px = 0; px < W; px += 3)
                    if (splatAlphaAt(s, float(px), float(py)) > 0.0f && !kept[size_t(py / 16) * tx16 + px / 16])
                        ++alphaPixelsOutside;
        }
    }
    std::printf("  %llu splats: rect %llu tiles, tight %llu (16 px) / %llu (8 px, = %.2f x 16-px tiles)\n",
                static_cast<unsigned long long>(splatsTested), static_cast<unsigned long long>(rectTiles),
                static_cast<unsigned long long>(tightTiles[0]), static_cast<unsigned long long>(tightTiles[1]),
                double(tightTiles[1]) / 4.0);
    std::printf("  pixels with alpha > 0 in a dropped tile: %llu (must be 0); cutoff-only misses: %llu\n",
                static_cast<unsigned long long>(droppedPixelsWithAlpha), static_cast<unsigned long long>(alphaPixelsOutside));
    CHECK(splatsTested > 2000, "too few splats tested (%llu)", static_cast<unsigned long long>(splatsTested));
    CHECK(droppedPixelsWithAlpha == 0, "%llu pixels with alpha >= 1/255 lie in dropped tiles",
          static_cast<unsigned long long>(droppedPixelsWithAlpha));
    CHECK(alphaPixelsOutside == 0, "cutoff-only: %llu pixels with alpha > 0 outside its tiles",
          static_cast<unsigned long long>(alphaPixelsOutside));
    CHECK(countMismatch == 0, "coverageCount != enumerated tiles for %llu splats", static_cast<unsigned long long>(countMismatch));
    CHECK(notSubset == 0, "%llu tight tiles outside INRIA's rectangle", static_cast<unsigned long long>(notSubset));
    CHECK(tight8vs16 == 0, "%llu 8-px tiles kept whose 16-px parent was dropped", static_cast<unsigned long long>(tight8vs16));
    CHECK(tightTiles[0] < rectTiles, "tight bounds should drop tiles");
}

// ---------------------------------------------------------------------------------------
// Synthetic scene (as test_tile_raster, plus needles and sub-1/255 opacities).
// ---------------------------------------------------------------------------------------
GaussianScene makeScene(uint32_t count, uint32_t seed) {
    GaussianScene sc;
    sc.count = count;
    sc.shDegree = 1;
    const uint32_t K = sc.shCoeffsPerChannel();
    sc.posAlpha.resize(count);
    sc.scales.resize(count);
    sc.rotations.resize(count);
    sc.sh.resize(size_t(count) * K * 3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (uint32_t i = 0; i < count; ++i) {
        glm::vec3 p(u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f);
        float alpha = u(rng);
        if (i % 7 == 0) alpha *= 0.01f;
        if (i % 3 == 0) alpha = 0.95f + 0.05f * u(rng);
        if (i % 4099 == 0) p.z = -1.0f;
        sc.posAlpha[i] = glm::vec4(p, alpha);
        const float s = std::exp(u(rng) * 4.0f - 5.5f);
        if (i % 5 == 0)  // needles
            sc.scales[i] = glm::vec4(s * 3.0f, s * 0.01f, s * 0.02f, 0.0f);
        else
            sc.scales[i] = glm::vec4(s, s * (0.2f + u(rng)), s * (0.2f + u(rng)), 0.0f);
        glm::vec4 q(u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f);
        sc.rotations[i] = glm::normalize(q);
        for (uint32_t k = 0; k < K * 3; ++k) sc.sh[size_t(i) * K * 3 + k] = (u(rng) - 0.5f) * (k < 3 ? 2.0f : 0.5f);
    }
    sc.stats.robustMin = glm::vec3(-8.0f);
    sc.stats.robustMax = glm::vec3(8.0f);
    return sc;
}

struct Camera {
    uint32_t W = 640, H = 360;
    glm::mat4 view, proj;
    SplatCamera cam;
};
Camera makeCamera() {
    Camera c;
    c.view = glm::lookAt(glm::vec3(0.3f, 0.5f, 6.0f), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    c.proj = glm::perspective(glm::radians(60.0f), float(c.W) / float(c.H), 0.01f, 100.0f);
    c.proj[1][1] *= -1.0f;
    c.cam = splatCameraFromViewer(c.view, c.proj, glm::mat4(1.0f), c.W, c.H);
    return c;
}

double psnrFloat(const std::vector<float>& rgba, const std::vector<float>& rgb, uint32_t pixels) {
    double se = 0.0;
    for (uint32_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c) {
            const double a = std::clamp(double(rgba[size_t(p) * 4 + c]), 0.0, 1.0);
            const double b = std::clamp(double(rgb[size_t(p) * 3 + c]), 0.0, 1.0);
            se += (a - b) * (a - b);
        }
    const double mse = se / (double(pixels) * 3);
    return mse > 0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
}

// ---------------------------------------------------------------------------------------
// 2. CPU reference: tight == rect, bit for bit
// ---------------------------------------------------------------------------------------
void testReferenceExactness(const GaussianScene& sc, const Camera& c) {
    std::printf("CPU reference renderer: tight bounds vs rect (bit-exact)\n");
    ReferenceOptions ro;
    const ReferenceImage rect = renderReference(sc, c.cam, ro);
    ro.bounds = TileBounds::Tight;
    const ReferenceImage tight = renderReference(sc, c.cam, ro);
    const bool same = rect.rgb == tight.rgb;
    std::printf("  tile instances %llu -> %llu (%.1f%%), evaluated pairs %llu -> %llu, blended %llu / %llu, "
                "terminated %llu / %llu, image %s\n",
                static_cast<unsigned long long>(rect.stats.tileInstances),
                static_cast<unsigned long long>(tight.stats.tileInstances),
                100.0 * double(tight.stats.tileInstances) / double(rect.stats.tileInstances),
                static_cast<unsigned long long>(rect.stats.evaluatedFragments),
                static_cast<unsigned long long>(tight.stats.evaluatedFragments),
                static_cast<unsigned long long>(rect.stats.blendedFragments),
                static_cast<unsigned long long>(tight.stats.blendedFragments),
                static_cast<unsigned long long>(rect.stats.terminatedPixels),
                static_cast<unsigned long long>(tight.stats.terminatedPixels), same ? "IDENTICAL" : "DIFFERENT");
    CHECK(same, "tight-bounds reference image differs from the rect image");
    CHECK(rect.stats.blendedFragments == tight.stats.blendedFragments, "blended pairs differ");
    CHECK(rect.stats.terminatedPixels == tight.stats.terminatedPixels, "terminated pixels differ");
    CHECK(tight.stats.tileInstances < rect.stats.tileInstances, "tight bounds should drop tile instances");
}

// ---------------------------------------------------------------------------------------
// 3. GPU lists and images
// ---------------------------------------------------------------------------------------
struct Result {
    std::vector<uint32_t> keys, values, ranges, tileCounts;
    TileControl control{};
    std::vector<float> image;
    std::vector<RenderSplatFields> splats;  // render-visible values per gaussian
    std::vector<TileRect> rects;
    uint32_t tilesX = 0, tilesY = 0;
};

Result renderTiles(VulkanContext& ctx, const GpuScene& gpu, const Camera& c, TileBounds bounds, uint32_t tileSize,
                   TileSortMode sort, TileDuplicateMode dup, RenderSplatMode rs, uint32_t sortWindow = 0) {
    GpuProfiler profiler;
    profiler.init(ctx, 1);
    CpuProfiler cpuProfiler;
    Image color = createImage2D(ctx, c.W, c.H, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                "test: color");
    Image depth{};
    SplatTileRenderer renderer;
    SplatSettings settings;
    settings.format = SplatTargetFormat::Rgba32f;
    settings.bounds = bounds;
    TileSettings ts;
    ts.sortMode = sort;
    ts.duplicateMode = dup;
    ts.renderSplat = rs;
    ts.tileSize = tileSize;
    ts.sortWindow = sortWindow;
    renderer.init(ctx, gpu, 1, settings, ts);
    renderer.onRenderTargetsChanged(ctx, color, depth);
    ImageState colorState{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    for (uint64_t frame = 0; frame < 2; ++frame) {
        profiler.beginFrame(0, frame);
        ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const FrameInput in{cmd,    profiler, cpuProfiler, color, depth, colorState,
                                {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE},
                                c.view, c.proj,   glm::mat4(1.0f), frame, 0};
            colorState = renderer.render(in).color;
        });
    }
    Result r;
    renderer.readBack(r.keys, r.values, r.ranges, r.control);
    renderer.readBackImage(r.image);
    r.tilesX = renderer.tilesX();
    r.tilesY = renderer.tilesY();
    const uint32_t n = gpu.count;
    r.splats.resize(n);
    r.rects.resize(n);
    if (rs == RenderSplatMode::Full) {
        std::vector<ProjectedSplat> sp;
        renderer.readBackSplats(sp, r.tileCounts);
        for (uint32_t i = 0; i < n; ++i) {
            r.splats[i] = {sp[i].center, sp[i].conicOpacity, sp[i].color, sp[i].radius};
            r.rects[i] = computeTileRect(sp[i].center, sp[i].radius, c.cam.tilesX, c.cam.tilesY);
        }
    } else {
        std::vector<uint32_t> words, rects, depths;
        renderer.readBackPackedSplats(words, rects, depths, r.tileCounts);
        const bool p24 = rs == RenderSplatMode::Packed24;
        const uint32_t stride = p24 ? 6 : 8;
        for (uint32_t i = 0; i < n; ++i) {
            r.splats[i] = decodeRenderSplat(p24, &words[size_t(i) * stride]);
            const uint32_t a = rects[2 * i], b = rects[2 * i + 1];
            r.rects[i] = {int(a & 0xFFFF), int(a >> 16), int(b & 0xFFFF), int(b >> 16)};
        }
    }
    renderer.destroy(ctx);
    destroyImage(ctx, color);
    profiler.destroy();
    return r;
}

// Checks counts and the list against a CPU construction from the GPU's own splats.
void checkLists(const char* name, const Result& r, TileBounds bounds, uint32_t S, TileSortMode sort,
                const std::vector<uint32_t>& depthBits) {
    const int gx = int(r.tilesX), gy = int(r.tilesY);
    std::vector<uint8_t> borderline(r.tileCounts.size(), 0);
    uint32_t countMismatch = 0, nBorderline = 0;
    std::vector<uint32_t> visible;
    for (uint32_t i = 0; i < r.tileCounts.size(); ++i) {
        if (r.tileCounts[i] == 0) continue;
        visible.push_back(i);
        const RenderSplatFields& f = r.splats[i];
        const Coverage cv = makeCoverage(bounds, r.rects[i], int(S), gx, gy, f.center, glm::vec3(f.conicOpacity),
                                         f.conicOpacity.w);
        borderline[i] = coverageIsBorderline(cv, int(S));
        nBorderline += borderline[i];
        if (!borderline[i] && coverageCount(cv, int(S)) != r.tileCounts[i]) ++countMismatch;
    }
    // CPU list: visible splats in (depth, index) order, their tiles row by row, stable sort by tile.
    std::stable_sort(visible.begin(), visible.end(),
                     [&](uint32_t a, uint32_t b) { return depthBits[a] < depthBits[b]; });
    std::vector<std::pair<uint32_t, uint32_t>> cpu;  // (tile, gaussian)
    for (uint32_t g : visible) {
        const RenderSplatFields& f = r.splats[g];
        const Coverage cv = makeCoverage(bounds, r.rects[g], int(S), gx, gy, f.center, glm::vec3(f.conicOpacity),
                                         f.conicOpacity.w);
        const auto [r0, r1] = coverageRows(cv, int(S));
        for (int y = r0; y < r1; ++y) {
            const auto [x0, x1] = coverageRowSpan(cv, y, int(S));
            for (int x = x0; x < x1; ++x) cpu.push_back({uint32_t(y * gx + x), g});
        }
    }
    std::stable_sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    // Compare per tile, ignoring entries of borderline splats on both sides.
    auto filtered = [&](const std::vector<std::pair<uint32_t, uint32_t>>& v) {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        for (const auto& e : v)
            if (!borderline[e.second]) out.push_back(e);
        return out;
    };
    std::vector<std::pair<uint32_t, uint32_t>> gpu;
    for (size_t k = 0; k < r.keys.size(); ++k) gpu.push_back({r.keys[k], r.values[k]});
    const auto a = filtered(gpu), b = filtered(cpu);
    uint32_t bad = 0;
    if (a.size() == b.size())
        for (size_t k = 0; k < a.size(); ++k) bad += a[k] != b[k];
    std::printf("  %-34s entries %u (cpu %zu), borderline splats %u, count mismatches %u, coverageMismatch %u, "
                "list %s\n",
                name, r.control.entries, cpu.size(), nBorderline, countMismatch, r.control.coverageMismatch,
                (a.size() == b.size() && bad == 0) ? "ok" : "MISMATCH");
    CHECK(countMismatch == 0, "%s: %u per-gaussian counts differ from the CPU coverage", name, countMismatch);
    CHECK(a.size() == b.size() && bad == 0, "%s: list differs (%zu vs %zu, %u mismatches)", name, a.size(), b.size(), bad);
    CHECK(r.control.coverageMismatch == 0, "%s: GPU count/write disagreement in %u splats", name,
          r.control.coverageMismatch);
    CHECK(r.control.overflow == 0, "%s: overflow", name);
    CHECK(nBorderline * 20u <= visible.size(), "%s: too many borderline splats (%u)", name, nBorderline);
    (void)sort;
}

void testGpu(VulkanContext& ctx, const GaussianScene& sc, const Camera& c) {
    std::printf("splat-tiles tight / cutoff-only bounds, tile sizes 8/16 (GPU vs CPU)\n");
    GpuScene gpu;
    gpu.upload(ctx, sc);
    const auto binning = TileSortMode::Binning, global = TileSortMode::Global;
    const auto perEntry = TileDuplicateMode::PerEntry, perSplat = TileDuplicateMode::PerSplat;
    const auto full = RenderSplatMode::Full;

    const Result rect16 = renderTiles(ctx, gpu, c, TileBounds::Rect, 16, binning, perEntry, full);
    // Depth keys of the visible splats (for the CPU list order), from a CPU projection.
    std::vector<uint32_t> depthBits(sc.count);
    for (uint32_t i = 0; i < sc.count; ++i) depthBits[i] = floatBits(viewDepth(c.cam.viewInria, glm::vec3(sc.posAlpha[i])));

    struct Case {
        const char* name;
        TileBounds bounds;
        uint32_t S;
        TileSortMode sort;
        TileDuplicateMode dup;
        RenderSplatMode rs;
    };
    const Case cases[] = {
        {"tight 16 binning per-entry full", TileBounds::Tight, 16, binning, perEntry, full},
        {"tight 16 binning per-splat full", TileBounds::Tight, 16, binning, perSplat, full},
        {"tight 16 global per-entry full", TileBounds::Tight, 16, global, perEntry, full},
        {"tight 16 global per-splat full", TileBounds::Tight, 16, global, perSplat, full},
        {"tight 16 binning per-entry p32", TileBounds::Tight, 16, binning, perEntry, RenderSplatMode::Packed32},
        {"tight 16 binning per-entry p24", TileBounds::Tight, 16, binning, perEntry, RenderSplatMode::Packed24},
        {"tight 8 binning per-entry full", TileBounds::Tight, 8, binning, perEntry, full},
        {"tight 8 global per-splat p32", TileBounds::Tight, 8, global, perSplat, RenderSplatMode::Packed32},
        {"rect 8 binning per-entry full", TileBounds::Rect, 8, binning, perEntry, full},
        {"cutoff-only 16 binning per-entry full", TileBounds::CutoffOnly, 16, binning, perEntry, full},
        {"cutoff-only 8 binning per-splat full", TileBounds::CutoffOnly, 8, binning, perSplat, full},
    };
    std::vector<float> rectImageP32, rectImageP24;
    for (const Case& k : cases) {
        const Result r = renderTiles(ctx, gpu, c, k.bounds, k.S, k.sort, k.dup, k.rs);
        checkLists(k.name, r, k.bounds, k.S, k.sort, depthBits);
        if (k.bounds == TileBounds::CutoffOnly) continue;
        // The image must be bit-identical to the rect image of the same render splat mode.
        const std::vector<float>* ref = &rect16.image;
        if (k.rs == RenderSplatMode::Packed32) {
            if (rectImageP32.empty())
                rectImageP32 = renderTiles(ctx, gpu, c, TileBounds::Rect, 16, binning, perEntry, k.rs).image;
            ref = &rectImageP32;
        } else if (k.rs == RenderSplatMode::Packed24) {
            if (rectImageP24.empty())
                rectImageP24 = renderTiles(ctx, gpu, c, TileBounds::Rect, 16, binning, perEntry, k.rs).image;
            ref = &rectImageP24;
        }
        const bool same = r.image == *ref;
        std::printf("      image vs rect 16 (%s): %s, entries %u -> %u\n", renderSplatModeName(k.rs),
                    same ? "bit-identical" : "DIFFERENT", rect16.control.entries, r.control.entries);
        CHECK(same, "%s: image differs from the rect image", k.name);
        if (k.bounds == TileBounds::Tight)
            CHECK(r.control.entries < rect16.control.entries * (k.S == 8 ? 4u : 1u), "%s: no tiles dropped", k.name);
    }

    // cutoff-only vs the CPU reference with the same bounds.
    {
        ReferenceOptions ro;
        ro.bounds = TileBounds::CutoffOnly;
        const ReferenceImage ref = renderReference(sc, c.cam, ro);
        ReferenceOptions rr;
        const ReferenceImage refRect = renderReference(sc, c.cam, rr);
        const Result r = renderTiles(ctx, gpu, c, TileBounds::CutoffOnly, 16, binning, perEntry, full);
        const double p = psnrFloat(r.image, ref.rgb, c.W * c.H);
        const double pRect = psnrFloat(r.image, refRect.rgb, c.W * c.H);
        std::printf("  cutoff-only GPU vs CPU cutoff-only reference: %.2f dB (vs INRIA rect reference: %.2f dB)\n", p,
                    pRect);
        CHECK(p > 80.0, "cutoff-only GPU vs CPU reference %.2f dB", p);
    }
    // 4. StopThePop window vs the CPU reference with the same window.
    for (uint32_t K : {4u, 16u}) {
        ReferenceOptions ro;
        ro.sortWindow = K;
        const ReferenceImage ref = renderReference(sc, c.cam, ro);
        const Result r = renderTiles(ctx, gpu, c, TileBounds::Tight, 16, binning, perEntry, full, K);
        const double p = psnrFloat(r.image, ref.rgb, c.W * c.H);
        const double pInria = psnrFloat(r.image, renderReference(sc, c.cam, ReferenceOptions{}).rgb, c.W * c.H);
        std::printf("  StopThePop window %u: GPU vs CPU same window %.2f dB (vs INRIA order %.2f dB)\n", K, p, pInria);
        CHECK(p > 60.0, "StopThePop K=%u: GPU vs CPU %.2f dB", K, p);
    }
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
    testCoverageGeometry();
    const GaussianScene sc = makeScene(150'000, 77);
    const Camera cam = makeCamera();
    testReferenceExactness(sc, cam);

    VulkanContext ctx;
    try {
        ctx.init(nullptr, validation);
    } catch (const std::exception& e) {
        std::printf("cannot create a headless Vulkan 1.3 device: %s\n", e.what());
        return 2;
    }
    std::printf("Device: %s, validation %s\n", ctx.deviceName.c_str(), ctx.validationEnabled ? "on" : "off");
    try {
        testGpu(ctx, sc, cam);
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
