// GPU tests for Phase 4 (CTest `tile_raster`), headless Vulkan compute:
//
//   1. GpuScan (exclusive prefix sum) vs a CPU scan, bit for bit: sizes around the
//      workgroup (256) and block (2048) sizes and large ones, the kNonZero (flags) and
//      kGather (indirect values) modes, the GPU-side count (smaller than the uploaded
//      data, clamped to the capacity) and the total written to a GPU buffer.
//   2. GpuRadixSort's Phase 4 modes vs std::stable_sort: short keys (keyBits < 32, with
//      the pass count rounded up to even) and 64-bit two-word keys (tile << 32 | depth).
//   3. The splat-tiles renderer on a synthetic scene, both tile-sort modes:
//      - the per-gaussian touched-tile counts equal the tile rectangles of the GPU splats;
//      - the sorted (tile, splat) list equals a CPU implementation of INRIA's pipeline
//        (duplicate in (depth, index) order, stable sort by tile), entry for entry;
//      - the tile ranges equal the CPU's [start, end) per tile;
//      - binning and global modes produce identical lists and identical images;
//      - the image matches the CPU reference renderer (front to back, early termination);
//      - overflow: with a tiny list budget the list is clamped, the overflow counter is
//        exact, the kept entries are the CPU's (truncated) list, and nothing is written
//        out of bounds (validation stays clean).
//
// Exit code: 0 = all checks passed, 1 = a check failed, 2 = no usable GPU / setup error.
#include "core/context.h"
#include "core/profiler.h"
#include "core/resources.h"
#include "render/gpu_radix_sort.h"
#include "render/gpu_scan.h"
#include "render/gpu_scene.h"
#include "render/splat_tile_renderer.h"
#include "scene/gaussian_scene.h"
#include "splat/depth_sort.h"
#include "splat/reference_renderer.h"
#include "splat/splat_math.h"

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

constexpr uint32_t kSentinel = 0xDEADBEEFu;

uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// Upload helper: host data -> device-local buffer (blocking).
void upload(VulkanContext& ctx, const Buffer& dst, const void* data, VkDeviceSize bytes, VkDeviceSize offset = 0) {
    if (bytes == 0) return;
    Buffer staging = createBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload, "test: staging");
    std::memcpy(staging.mapped, data, size_t(bytes));
    VK_CHECK(vmaFlushAllocation(ctx.allocator, staging.allocation, 0, VK_WHOLE_SIZE));
    ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{0, offset, bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    });
    destroyBuffer(ctx, staging);
}

std::vector<uint32_t> download(VulkanContext& ctx, const Buffer& src, VkDeviceSize bytes) {
    Buffer rb = createBuffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback, "test: readback");
    ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, src.buffer, rb.buffer, 1, &region);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(ctx.allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* p = static_cast<const uint32_t*>(rb.mapped);
    std::vector<uint32_t> out(p, p + bytes / 4);
    destroyBuffer(ctx, rb);
    return out;
}

constexpr VkBufferUsageFlags kTestUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

// ---------------------------------------------------------------------------------------
// 1. Prefix sum
// ---------------------------------------------------------------------------------------
void testScan(VulkanContext& ctx) {
    std::printf("GpuScan vs CPU exclusive scan (bit-exact)\n");
    constexpr uint32_t kCapacity = 1'000'003;
    Buffer input = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: scan input");
    Buffer index = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: scan index");
    Buffer output = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: scan output");
    Buffer control = createBuffer(ctx, 16, kTestUsage, MemoryUsage::GpuOnly, "test: scan count/total");
    std::mt19937 rng(42);

    for (uint32_t flags : {0u, uint32_t(GpuScan::kNonZero), uint32_t(GpuScan::kGather)}) {
        GpuScan scan;
        GpuScan::Bindings b;
        b.input = input.buffer;
        b.index = index.buffer;
        b.output = output.buffer;
        b.countBuffer = control.buffer;
        b.countOffset = 4;  // count at word 1
        b.totalBuffer = control.buffer;
        b.totalOffset = 8;  // total at word 2
        b.flags = flags;
        scan.init(ctx, kCapacity, b);
        const char* mode = flags == 0 ? "plain" : flags == GpuScan::kNonZero ? "non-zero flags" : "gather";
        for (uint32_t n : {0u, 1u, 2u, 255u, 256u, 257u, 2047u, 2048u, 2049u, 4097u, 65537u, 300'001u, kCapacity,
                           kCapacity + 77u /* clamped */}) {
            const uint32_t used = std::min(n, kCapacity);
            std::vector<uint32_t> in(kCapacity), idx(kCapacity);
            for (auto& v : in) v = (rng() % 4 == 0) ? 0u : rng() % 50u;  // many zeros (flags mode)
            for (auto& v : idx) v = rng() % kCapacity;
            upload(ctx, input, in.data(), VkDeviceSize(kCapacity) * 4);
            upload(ctx, index, idx.data(), VkDeviceSize(kCapacity) * 4);
            std::vector<uint32_t> sentinel(kCapacity, kSentinel);
            upload(ctx, output, sentinel.data(), VkDeviceSize(kCapacity) * 4);
            const uint32_t ctl[4] = {kSentinel, n, kSentinel, kSentinel};
            upload(ctx, control, ctl, sizeof(ctl));
            ctx.immediateSubmit([&](VkCommandBuffer cmd) { scan.record(cmd); });
            const std::vector<uint32_t> out = download(ctx, output, VkDeviceSize(kCapacity) * 4);
            const std::vector<uint32_t> c = download(ctx, control, 16);

            uint32_t running = 0, mismatches = 0, tail = 0;
            for (uint32_t j = 0; j < used; ++j) {
                if (out[j] != running) ++mismatches;
                uint32_t v = flags == GpuScan::kGather ? in[idx[j]] : in[j];
                if (flags == GpuScan::kNonZero) v = v != 0;
                running += v;
            }
            for (uint32_t j = used; j < kCapacity; ++j) tail += out[j] != kSentinel;
            const bool ok = mismatches == 0 && tail == 0 && c[2] == running;
            std::printf("  %-15s n=%-8u %s\n", mode, n, ok ? "ok" : "MISMATCH");
            CHECK(mismatches == 0, "%s n=%u: %u mismatches", mode, n, mismatches);
            CHECK(tail == 0, "%s n=%u: %u entries past the count modified", mode, n, tail);
            CHECK(c[2] == running, "%s n=%u: total %u, expected %u", mode, n, c[2], running);
            CHECK(c[0] == kSentinel && c[3] == kSentinel, "%s: control words around count/total modified", mode);
        }
        scan.destroy(ctx);
    }
    for (Buffer* b : {&input, &index, &output, &control}) destroyBuffer(ctx, *b);
}

// ---------------------------------------------------------------------------------------
// 2. Radix sort: short keys and 64-bit keys
// ---------------------------------------------------------------------------------------
void testSortModes(VulkanContext& ctx, GpuRadixSort::Variant variant) {
    const char* vname = variant == GpuRadixSort::Variant::Classic ? "classic" : "reorder";
    std::printf("GpuRadixSort (%s) short keys and 64-bit keys vs std::stable_sort\n", vname);
    constexpr uint32_t kCapacity = 600'011;
    Buffer keys = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: keys");
    Buffer keysHi = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: keys hi");
    Buffer values = createBuffer(ctx, kCapacity * 4, kTestUsage, MemoryUsage::GpuOnly, "test: values");
    Buffer count = createBuffer(ctx, 16, kTestUsage, MemoryUsage::GpuOnly, "test: count");
    std::mt19937 rng(7);

    for (bool key64 : {false, true}) {
        GpuRadixSort sort;
        sort.init(ctx, kCapacity, keys.buffer, values.buffer, count.buffer, 0, key64 ? keysHi.buffer : VK_NULL_HANDLE);
        sort.setVariant(variant);
        const std::vector<uint32_t> bitsList = key64 ? std::vector<uint32_t>{40, 45, 46, 64}
                                                     : std::vector<uint32_t>{8, 13, 15, 20, 32};
        for (uint32_t bits : bitsList) {
            for (uint32_t n : {0u, 1u, 2049u, 100'003u, kCapacity}) {
                std::vector<uint32_t> lo(n), hi(n), v(n);
                for (uint32_t i = 0; i < n; ++i) {
                    v[i] = i;
                    if (key64) {
                        const uint32_t hiBits = bits - 32;
                        hi[i] = hiBits >= 32 ? uint32_t(rng()) : uint32_t(rng()) & ((1u << hiBits) - 1u);
                        // Few distinct low words: many full-key ties, so stability is tested.
                        lo[i] = (rng() % 3 == 0) ? floatBits(1.0f + float(rng() % 16)) : uint32_t(rng());
                    } else {
                        lo[i] = bits >= 32 ? uint32_t(rng()) : uint32_t(rng()) & ((1u << bits) - 1u);
                    }
                }
                upload(ctx, keys, lo.data(), VkDeviceSize(n) * 4);
                if (key64) upload(ctx, keysHi, hi.data(), VkDeviceSize(n) * 4);
                upload(ctx, values, v.data(), VkDeviceSize(n) * 4);
                const uint32_t c[4] = {n, 0, 0, 0};
                upload(ctx, count, c, sizeof(c));
                GpuRadixSort::RecordOptions o;
                o.keyBits = bits;
                ctx.immediateSubmit([&](VkCommandBuffer cmd) { sort.record(cmd, nullptr, o); });

                std::vector<uint32_t> order(n);
                std::iota(order.begin(), order.end(), 0u);
                std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                    const uint64_t ka = (uint64_t(key64 ? hi[a] : 0) << 32) | lo[a];
                    const uint64_t kb = (uint64_t(key64 ? hi[b] : 0) << 32) | lo[b];
                    return ka < kb;
                });
                uint32_t bad = 0;
                if (n > 0) {
                    const std::vector<uint32_t> gk = download(ctx, keys, VkDeviceSize(n) * 4);
                    const std::vector<uint32_t> gv = download(ctx, values, VkDeviceSize(n) * 4);
                    std::vector<uint32_t> gh;
                    if (key64) gh = download(ctx, keysHi, VkDeviceSize(n) * 4);
                    for (uint32_t i = 0; i < n; ++i) {
                        const uint32_t e = order[i];
                        if (gv[i] != e || gk[i] != lo[e] || (key64 && gh[i] != hi[e])) ++bad;
                    }
                }
                std::printf("  %s keyBits=%-2u (%u passes) n=%-7u %s\n", key64 ? "64-bit" : "32-bit", bits,
                            GpuRadixSort::passCount(bits), n, bad ? "MISMATCH" : "ok");
                CHECK(bad == 0, "%s %s keyBits=%u n=%u: %u mismatches", vname, key64 ? "64-bit" : "32-bit", bits, n,
                      bad);
            }
        }
        sort.destroy(ctx);
    }
    for (Buffer* b : {&keys, &keysHi, &values, &count}) destroyBuffer(ctx, *b);
}

// ---------------------------------------------------------------------------------------
// 3. The renderer
// ---------------------------------------------------------------------------------------
GaussianScene makeSyntheticScene(uint32_t count, uint32_t seed) {
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
        // A box around the camera (at z = +6 looking at -z): some gaussians are behind
        // the camera, inside the near plane, off screen, large or almost transparent.
        glm::vec3 p(u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f);
        float alpha = u(rng);
        if (i % 7 == 0) alpha *= 0.01f;  // some below 1/255
        if (i % 3 == 0) alpha = 0.95f + 0.04f * u(rng);  // nearly opaque: early termination happens
        if (i % 4099 == 0) p.z = -1.0f;  // EXACTLY equal depths: ties
        sc.posAlpha[i] = glm::vec4(p, alpha);
        const float s = std::exp(u(rng) * 4.0f - 5.5f);  // 0.004 .. 0.22
        sc.scales[i] = glm::vec4(s, s * (0.2f + u(rng)), s * (0.2f + u(rng)), 0.0f);
        glm::vec4 q(u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f);
        sc.rotations[i] = glm::normalize(q);
        for (uint32_t k = 0; k < K * 3; ++k) sc.sh[size_t(i) * K * 3 + k] = (u(rng) - 0.5f) * (k < 3 ? 2.0f : 0.5f);
    }
    sc.stats.robustMin = glm::vec3(-8.0f);
    sc.stats.robustMax = glm::vec3(8.0f);
    return sc;
}

struct RenderResult {
    std::vector<uint32_t> keys, values, ranges;
    TileControl control{};
    std::vector<float> image;  // rgba32f
    std::vector<ProjectedSplat> splats;
    std::vector<uint32_t> tileCounts;
};

RenderResult renderTiles(VulkanContext& ctx, const GaussianScene& sc, const GpuScene& gpu, TileSortMode mode,
                         TileDuplicateMode dup, uint32_t budgetMB, uint32_t W, uint32_t H, const glm::mat4& view,
                         const glm::mat4& proj) {
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
    TileSettings ts;
    ts.sortMode = mode;
    ts.duplicateMode = dup;
    ts.budgetMB = budgetMB;
    ts.stats = false;
    renderer.init(ctx, gpu, 1, settings, ts);
    renderer.onRenderTargetsChanged(ctx, color, depth);
    const glm::mat4 model(1.0f);
    ImageState colorState{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    for (uint64_t frame = 0; frame < 2; ++frame) {  // the second frame reuses every buffer
        profiler.beginFrame(0, frame);
        ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const FrameInput in{cmd,   profiler, cpuProfiler, color, depth, colorState,
                                {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE},
                                view,  proj,     model,       frame, 0};
            colorState = renderer.render(in).color;
        });
    }
    RenderResult r;
    renderer.readBack(r.keys, r.values, r.ranges, r.control);
    renderer.readBackImage(r.image);
    renderer.readBackSplats(r.splats, r.tileCounts);
    renderer.destroy(ctx);
    destroyImage(ctx, color);
    profiler.destroy();
    (void)sc;
    return r;
}

// CPU version of the list construction from the GPU's own splats (so that the test checks
// the duplication / sort / ranges logic exactly, independent of float differences in the
// projection): visible splats in (depth, index) order, duplicated per tile of the tile
// rectangle (row by row), truncated to `capacity` (overflow), stable-sorted by tile.
struct CpuList {
    std::vector<uint32_t> keys, values;  // sorted by tile
    uint64_t duplicates = 0;
    std::vector<uint32_t> start, end;    // per tile
};
CpuList cpuTileList(const std::vector<ProjectedSplat>& splats, const std::vector<uint32_t>& tileCounts, uint32_t tilesX,
                    uint32_t tilesY, uint32_t capacity, TileSortMode mode) {
    // Visible = a non-zero touched-tile count (the GPU only writes the Splat of visible
    // gaussians; test a) checks the counts against the splats' rectangles).
    std::vector<uint32_t> visible;
    for (uint32_t i = 0; i < splats.size(); ++i)
        if (tileCounts[i] != 0) visible.push_back(i);
    // Binning duplicates in depth order (ties by index); global in index order. The final
    // per-tile order is (depth, index) either way, but the overflow truncation differs.
    if (mode == TileSortMode::Binning)
        std::stable_sort(visible.begin(), visible.end(), [&](uint32_t a, uint32_t b) {
            return floatBits(splats[a].depth) < floatBits(splats[b].depth);
        });
    std::vector<uint32_t> tiles, vals;
    CpuList out;
    for (uint32_t g : visible) {
        const TileRect r = computeTileRect(splats[g].center, splats[g].radius, tilesX, tilesY);
        for (int y = r.minY; y < r.maxY; ++y)
            for (int x = r.minX; x < r.maxX; ++x) {
                ++out.duplicates;
                if (tiles.size() < capacity) {
                    tiles.push_back(uint32_t(y) * tilesX + uint32_t(x));
                    vals.push_back(g);
                }
            }
    }
    std::vector<uint32_t> order(tiles.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (tiles[a] != tiles[b]) return tiles[a] < tiles[b];
        if (mode == TileSortMode::Global) return floatBits(splats[vals[a]].depth) < floatBits(splats[vals[b]].depth);
        return false;  // binning: already in depth order, stability keeps it
    });
    for (uint32_t o : order) {
        out.keys.push_back(tiles[o]);
        out.values.push_back(vals[o]);
    }
    out.start.assign(size_t(tilesX) * tilesY, 0);
    out.end.assign(size_t(tilesX) * tilesY, 0);
    for (uint32_t k = 0; k < out.keys.size(); ++k) {
        const uint32_t t = out.keys[k];
        if (k == 0 || out.keys[k - 1] != t) out.start[t] = k;
        out.end[t] = k + 1;
    }
    return out;
}

void checkList(const char* name, const RenderResult& r, const CpuList& cpu, uint32_t capacity) {
    const uint64_t expectEntries = std::min<uint64_t>(cpu.duplicates, capacity);
    CHECK(r.control.duplicates == cpu.duplicates, "%s: duplicates gpu %u cpu %llu", name, r.control.duplicates,
          static_cast<unsigned long long>(cpu.duplicates));
    CHECK(r.control.entries == expectEntries, "%s: entries gpu %u expected %llu", name, r.control.entries,
          static_cast<unsigned long long>(expectEntries));
    CHECK(r.control.overflow == cpu.duplicates - expectEntries, "%s: overflow gpu %u expected %llu", name,
          r.control.overflow, static_cast<unsigned long long>(cpu.duplicates - expectEntries));
    uint32_t bad = 0;
    if (r.keys.size() == cpu.keys.size())
        for (size_t k = 0; k < r.keys.size(); ++k) bad += r.keys[k] != cpu.keys[k] || r.values[k] != cpu.values[k];
    CHECK(r.keys.size() == cpu.keys.size() && bad == 0, "%s: list differs (%zu vs %zu entries, %u mismatches)", name,
          r.keys.size(), cpu.keys.size(), bad);
    uint32_t badRanges = 0;
    for (size_t t = 0; t < cpu.start.size(); ++t)
        badRanges += r.ranges[2 * t] != cpu.start[t] || r.ranges[2 * t + 1] != cpu.end[t];
    CHECK(badRanges == 0, "%s: %u tile ranges differ", name, badRanges);
    std::printf("  %-30s duplicates %u, entries %u, overflow %u; list %s, ranges %s\n", name, r.control.duplicates,
                r.control.entries, r.control.overflow, bad ? "MISMATCH" : "ok", badRanges ? "MISMATCH" : "ok");
}

void testRenderer(VulkanContext& ctx) {
    std::printf("splat-tiles renderer vs CPU (lists, ranges, image)\n");
    const uint32_t W = 640, H = 360;
    const GaussianScene sc = makeSyntheticScene(250'000, 99);
    GpuScene gpu;
    gpu.upload(ctx, sc);
    const glm::mat4 view = glm::lookAt(glm::vec3(0.3f, 0.5f, 6.0f), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(W) / float(H), 0.01f, 100.0f);
    proj[1][1] *= -1.0f;
    const SplatCamera cam = splatCameraFromViewer(view, proj, glm::mat4(1.0f), W, H);
    const uint32_t tilesX = cam.tilesX, tilesY = cam.tilesY;

    const auto perEntry = TileDuplicateMode::PerEntry, perSplat = TileDuplicateMode::PerSplat;
    const RenderResult bin = renderTiles(ctx, sc, gpu, TileSortMode::Binning, perEntry, 256, W, H, view, proj);
    const RenderResult glob = renderTiles(ctx, sc, gpu, TileSortMode::Global, perEntry, 256, W, H, view, proj);
    const RenderResult binLoop = renderTiles(ctx, sc, gpu, TileSortMode::Binning, perSplat, 256, W, H, view, proj);
    const RenderResult globLoop = renderTiles(ctx, sc, gpu, TileSortMode::Global, perSplat, 256, W, H, view, proj);

    // a) Touched-tile counts = the tile rectangles of the GPU splats (only visible splats
    //    are written by the GPU); the visible set equals the CPU's near/opacity/preprocess
    //    culling.
    uint32_t badCounts = 0, visible = 0, visibleMismatch = 0;
    std::vector<glm::vec3> coeffs(sc.shCoeffsPerChannel());
    for (uint32_t i = 0; i < sc.count; ++i) {
        const glm::vec4 pa = sc.posAlpha[i];
        bool cpuVisible = false;
        if (viewDepth(cam.viewInria, glm::vec3(pa)) > kNearCullDepth && pa.w >= kMinAlpha) {
            for (uint32_t k = 0; k < coeffs.size(); ++k) coeffs[k] = sc.shCoeff(i, k);
            cpuVisible = preprocessGaussian(pa, glm::vec3(sc.scales[i]), sc.rotations[i], coeffs.data(), sc.shDegree,
                                            cam)
                             .cullReason == uint32_t(CullReason::Visible);
        }
        const bool gpuVisible = bin.tileCounts[i] != 0;
        visibleMismatch += cpuVisible != gpuVisible;
        if (!gpuVisible) continue;
        ++visible;
        const ProjectedSplat& s = bin.splats[i];
        const TileRect r = computeTileRect(s.center, s.radius, tilesX, tilesY);
        badCounts += s.cullReason != uint32_t(CullReason::Visible) ||
                     bin.tileCounts[i] != uint32_t((r.maxX - r.minX) * (r.maxY - r.minY));
    }
    CHECK(visibleMismatch == 0, "%u gaussians visible on only one of CPU/GPU", visibleMismatch);
    std::printf("  visible %u (gpu counter %u), tile counts mismatches %u\n", visible, bin.control.cullCounts[0],
                badCounts);
    CHECK(badCounts == 0, "%u touched-tile counts differ from the GPU splats' tile rectangles", badCounts);
    CHECK(bin.control.cullCounts[0] == visible, "visible counter %u, visible splats %u", bin.control.cullCounts[0],
          visible);
    CHECK(bin.control.visible == visible, "binning: scan total of the flags %u, expected %u", bin.control.visible,
          visible);

    // b) Lists and ranges, both modes, vs the CPU construction.
    const uint32_t capacity = 1u << 30;
    checkList("binning", bin, cpuTileList(bin.splats, bin.tileCounts, tilesX, tilesY, capacity, TileSortMode::Binning), capacity);
    checkList("global", glob, cpuTileList(glob.splats, glob.tileCounts, tilesX, tilesY, capacity, TileSortMode::Global), capacity);
    CHECK(bin.keys == glob.keys && bin.values == glob.values, "binning and global lists differ");
    CHECK(bin.image == glob.image, "binning and global images differ");
    CHECK(binLoop.keys == bin.keys && binLoop.values == bin.values && binLoop.ranges == bin.ranges,
          "binning: per-splat and per-entry duplication differ");
    CHECK(globLoop.keys == bin.keys && globLoop.values == bin.values && globLoop.ranges == bin.ranges,
          "global: per-splat and per-entry duplication differ");
    const bool same = binLoop.keys == bin.keys && globLoop.keys == bin.keys && glob.keys == bin.keys;
    std::printf("  per-splat vs per-entry duplication, binning vs global: %s\n", same ? "identical" : "DIFFERENT");
    // There must be exact depth ties among the listed splats (the scene has them), so the
    // (depth, index) tie order is actually exercised.
    {
        uint32_t ties = 0;
        for (size_t k = 1; k < bin.keys.size(); ++k)
            ties += bin.keys[k] == bin.keys[k - 1] &&
                    floatBits(bin.splats[bin.values[k]].depth) == floatBits(bin.splats[bin.values[k - 1]].depth);
        std::printf("  adjacent entries with equal tile and equal depth: %u\n", ties);
        CHECK(ties > 0, "the synthetic scene was meant to contain exact depth ties within tiles");
    }

    // c) Image vs the CPU reference renderer (INRIA forward pass with early termination).
    ReferenceOptions ro;
    const ReferenceImage ref = renderReference(sc, cam, ro);
    double se = 0.0, maxErr = 0.0;
    uint64_t over = 0;
    for (uint32_t p = 0; p < W * H; ++p)
        for (int c = 0; c < 3; ++c) {
            const double a = std::clamp(double(bin.image[size_t(p) * 4 + c]), 0.0, 1.0);
            const double b = std::clamp(double(ref.rgb[size_t(p) * 3 + c]), 0.0, 1.0);
            se += (a - b) * (a - b);
            maxErr = std::max(maxErr, std::abs(a - b));
            over += std::abs(a - b) > 2.0 / 255.0;
        }
    const double mse = se / (double(W) * H * 3);
    const double psnr = mse > 0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
    std::printf("  image vs CPU reference (float, before quantization): PSNR %.2f dB, max %.2f/255, %llu of %u "
                "channel values > 2/255; reference: %llu pairs evaluated, %llu terminated pixels\n",
                psnr, maxErr * 255.0, static_cast<unsigned long long>(over), W * H * 3,
                static_cast<unsigned long long>(ref.stats.evaluatedFragments),
                static_cast<unsigned long long>(ref.stats.terminatedPixels));
    CHECK(psnr > 60.0, "image PSNR vs reference %.2f dB", psnr);
    CHECK(ref.stats.terminatedPixels > 0, "the test scene should make some pixels terminate early");

    // d) Overflow: 1 MB budget -> capacity << duplicates. Clamped, counted, consistent.
    for (TileSortMode mode : {TileSortMode::Binning, TileSortMode::Global})
    for (TileDuplicateMode dup : {perEntry, perSplat}) {
        const RenderResult small = renderTiles(ctx, sc, gpu, mode, dup, 1, W, H, view, proj);
        const uint32_t cap = small.control.entries;
        const CpuList cpu = cpuTileList(small.splats, small.tileCounts, tilesX, tilesY, cap, mode);
        CHECK(small.control.overflow > 0, "%s: the 1 MB budget was meant to overflow", tileSortModeName(mode));
        const std::string name = std::string(tileSortModeName(mode)) + "/" + tileDuplicateModeName(dup) + ", 1 MB";
        checkList(name.c_str(), small, cpu, cap);
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
    VulkanContext ctx;
    try {
        ctx.init(nullptr, validation);
    } catch (const std::exception& e) {
        std::printf("cannot create a headless Vulkan 1.3 device: %s\n", e.what());
        return 2;
    }
    std::printf("Device: %s, subgroup size %u, validation %s\n", ctx.deviceName.c_str(), ctx.subgroup.size,
                ctx.validationEnabled ? "on" : "off");
    try {
        testScan(ctx);
        for (auto v : {GpuRadixSort::Variant::Classic, GpuRadixSort::Variant::Reorder}) testSortModes(ctx, v);
        testRenderer(ctx);
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
