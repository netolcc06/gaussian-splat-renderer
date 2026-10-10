#include "webgpu/wgpu_tile_selftest.h"

#include "scene/gaussian_scene.h"
#include "scene/scene_compression.h"
#include "splat/depth_sort.h"
#include "splat/reference_renderer.h"
#include "splat/tile_coverage.h"
#include "webgpu/coverage_spans.h"
#include "webgpu/shader_constants.h"
#include "webgpu/wgpu_compute.h"
#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_radix_sort.h"
#include "webgpu/wgpu_resources.h"
#include "webgpu/wgpu_scan.h"
#include "webgpu/wgpu_scene.h"
#include "webgpu/wgpu_shaders.h"
#include "webgpu/wgpu_tile_renderer.h"

#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <set>

namespace gsr {

namespace {

using json = nlohmann::json;
using web::SpanRecord;
using web::WebTileControl;
using web::kCoverageSpanWords;

constexpr uint32_t kSentinel = 0xDEADBEEFu;

uint32_t floatBits(float f) { return std::bit_cast<uint32_t>(f); }

// --- Asynchronous readbacks: copies into MAP_READ buffers, one map each, then a callback --------
struct Job {
    struct Item {
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
        uint32_t rowPitch = 0;  // textures: bytes per padded row
        std::vector<uint8_t> data;
        bool ok = false;
        std::string error;
    };
    WGPUDevice device = nullptr;
    std::vector<Item> items;
    uint32_t pending = 0;
    bool done = false;
    std::function<void(Job&)> onComplete;
    std::vector<std::shared_ptr<void>> keepAlive;  // objects whose bind groups the submit uses

    size_t copyBuffer(WGPUCommandEncoder encoder, WGPUBuffer src, uint64_t bytes, uint64_t offset = 0) {
        Item it;
        it.size = std::max<uint64_t>((bytes + 3) & ~uint64_t(3), 4);
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        desc.size = it.size;
        it.buffer = wgpuDeviceCreateBuffer(device, &desc);
        if (bytes > 0) wgpuCommandEncoderCopyBufferToBuffer(encoder, src, offset, it.buffer, 0, (bytes + 3) & ~uint64_t(3));
        items.push_back(std::move(it));
        return items.size() - 1;
    }
    size_t copyTexture(WGPUCommandEncoder encoder, const WgpuTexture2D& tex, uint32_t bytesPerPixel) {
        Item it;
        it.rowPitch = alignedBytesPerRow(tex.width * bytesPerPixel);
        it.size = uint64_t(it.rowPitch) * tex.height;
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        desc.size = it.size;
        it.buffer = wgpuDeviceCreateBuffer(device, &desc);
        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = tex.texture;
        WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
        dst.buffer = it.buffer;
        dst.layout.bytesPerRow = it.rowPitch;
        dst.layout.rowsPerImage = tex.height;
        const WGPUExtent3D extent = {tex.width, tex.height, 1};
        wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
        items.push_back(std::move(it));
        return items.size() - 1;
    }

    struct MapCtx {
        Job* job;
        size_t index;
    };
    static void onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*) {
        std::unique_ptr<MapCtx> ctx(static_cast<MapCtx*>(userdata1));
        Job& job = *ctx->job;
        Item& it = job.items[ctx->index];
        if (status == WGPUMapAsyncStatus_Success) {
            const auto* p = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(it.buffer, 0, size_t(it.size)));
            it.data.assign(p, p + it.size);
            it.ok = true;
            wgpuBufferUnmap(it.buffer);
        } else {
            it.error = "map failed: " + toString(message);
        }
        wgpuBufferRelease(it.buffer);
        it.buffer = nullptr;
        if (--job.pending == 0) job.finish();
    }
    void submitAndMap(WGPUQueue queue, WGPUCommandEncoder encoder) {
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(queue, 1, &commands);
        wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);
        pending = uint32_t(items.size());
        if (pending == 0) return finish();
        for (size_t i = 0; i < items.size(); ++i) {
            WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            cb.mode = WGPUCallbackMode_AllowSpontaneous;
            cb.callback = &Job::onMapped;
            cb.userdata1 = new MapCtx{this, i};
            wgpuBufferMapAsync(items[i].buffer, WGPUMapMode_Read, 0, size_t(items[i].size), cb);
        }
    }
    void finish() {
        keepAlive.clear();
        if (onComplete) onComplete(*this);
        done = true;
    }
    bool allOk() const {
        return std::all_of(items.begin(), items.end(), [](const Item& it) { return it.ok; });
    }
    template <typename T>
    std::vector<T> as(size_t i, size_t count = SIZE_MAX) const {
        const std::vector<uint8_t>& d = items[i].data;
        const size_t n = std::min(count, d.size() / sizeof(T));
        std::vector<T> v(n);
        if (n) std::memcpy(v.data(), d.data(), n * sizeof(T));
        return v;
    }
    // Texture rows without the padding.
    template <typename T>
    std::vector<T> texture(size_t i, uint32_t width, uint32_t height, uint32_t components) const {
        const Item& it = items[i];
        std::vector<T> v(size_t(width) * height * components);
        const size_t row = size_t(width) * components * sizeof(T);
        for (uint32_t y = 0; y < height; ++y)
            std::memcpy(reinterpret_cast<uint8_t*>(v.data()) + y * row, it.data.data() + size_t(y) * it.rowPitch, row);
        return v;
    }
};

WGPUBuffer makeBuffer(WGPUDevice device, uint64_t bytes, WGPUBufferUsage usage, const char* label) {
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.usage = usage;
    desc.size = std::max<uint64_t>((bytes + 3) & ~uint64_t(3), 16);
    return wgpuDeviceCreateBuffer(device, &desc);
}

// A shared_ptr deleter for raw WebGPU buffers kept alive by a job.
std::shared_ptr<void> holdBuffer(WGPUBuffer b) {
    return std::shared_ptr<void>(b, [](void* p) { wgpuBufferRelease(static_cast<WGPUBuffer>(p)); });
}

WGPUComputePassEncoder beginPass(WGPUCommandEncoder encoder, const char* label) {
    return beginComputePass(encoder, label, nullptr);
}
void endPass(WGPUComputePassEncoder pass) {
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

// The synthetic scene of tests/test_tile_raster.cpp (makeSyntheticScene): a box around the camera,
// gaussians behind it, inside the near plane, off screen, large, almost transparent, nearly opaque
// (early termination), and exact depth ties.
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
        glm::vec3 p(u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f);
        float alpha = u(rng);
        if (i % 7 == 0) alpha *= 0.01f;
        if (i % 3 == 0) alpha = 0.95f + 0.04f * u(rng);
        if (i % 4099 == 0) p.z = -1.0f;
        sc.posAlpha[i] = glm::vec4(p, alpha);
        const float s = std::exp(u(rng) * 4.0f - 5.5f);
        sc.scales[i] = glm::vec4(s, s * (0.2f + u(rng)), s * (0.2f + u(rng)), 0.0f);
        glm::vec4 q(u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f);
        sc.rotations[i] = glm::normalize(q);
        for (uint32_t k = 0; k < K * 3; ++k) sc.sh[size_t(i) * K * 3 + k] = (u(rng) - 0.5f) * (k < 3 ? 2.0f : 0.5f);
    }
    return sc;
}

// The coverage of a GPU splat, the CPU way (tile_coverage.h).
Coverage cpuCoverage(const ProjectedSplat& s, uint32_t tilesX, uint32_t tilesY) {
    return splatCoverage(TileBounds::Tight, s, tilesX, tilesY, int(web::kTileSize), int(tilesX), int(tilesY));
}

// Coverage checks shared by the synthetic "coverage" test and the real-scene analysis.
json coverageChecks(const std::vector<ProjectedSplat>& splats, const std::vector<uint32_t>& tileCounts,
                    const std::vector<uint32_t>& spans, uint32_t tilesX, uint32_t tilesY, bool& pass) {
    const uint32_t n = uint32_t(tileCounts.size());
    uint64_t visible = 0, counted = 0, equal = 0, borderlineDiff = 0, hardDiff = 0, countMismatch = 0,
             supersetViolations = 0, supersetBorderline = 0, cpuInconsistent = 0;
    uint64_t storedTotal = 0, exactTotal = 0;
    std::array<uint64_t, 14> rowHist{};  // 0..8, 9-12, 13-16, 17-32, 33-64, >64
    const uint32_t kList[] = {1, 2, 4, 8, 16, 32};
    std::array<uint64_t, 6> tailExtra{};  // list entries added by a tail at row K, per K
    json firstHard;
    std::vector<uint8_t> mark;
    for (uint32_t i = 0; i < n; ++i) {
        const ProjectedSplat& s = splats[i];
        if (!(s.radius > 0.0f)) continue;  // culled: never written by the GPU
        ++visible;
        const uint32_t* w = spans.data() + size_t(i) * kCoverageSpanWords;
        const uint32_t gpuCount = web::spanRecordCount(w);
        counted += tileCounts[i] > 0;
        storedTotal += tileCounts[i];
        if (gpuCount != tileCounts[i]) ++countMismatch;
        const uint32_t nRows = w[1] & 0x7FFFu;
        rowHist[nRows <= 8 ? nRows : nRows <= 12 ? 9 : nRows <= 16 ? 10 : nRows <= 32 ? 11 : nRows <= 64 ? 12 : 13]++;

        const Coverage cv = cpuCoverage(s, tilesX, tilesY);
        SpanRecord cpu;
        exactTotal += coverageCount(cv, int(web::kTileSize));
        const uint32_t cpuCount = web::encodeCoverageSpans(cv, int(web::kTileSize), cpu);
        // The encoder's count must be what the record enumerates (a field overflow would break it).
        uint32_t enumerated = 0;
        web::forEachSpanTile(cpu.data(), [&](uint32_t, uint32_t) { ++enumerated; });
        if (cpuCount != web::spanRecordCount(cpu.data()) || cpuCount != enumerated) ++cpuInconsistent;
        const bool same = std::equal(cpu.begin(), cpu.end(), w);
        const bool borderline = coverageIsBorderline(cv, int(web::kTileSize));
        if (same) {
            ++equal;
        } else if (borderline) {
            ++borderlineDiff;
        } else {
            ++hardDiff;
            if (firstHard.is_null()) {
                firstHard = {{"splat", i}, {"cpu", std::vector<uint32_t>(cpu.begin(), cpu.end())},
                             {"gpu", std::vector<uint32_t>(w, w + kCoverageSpanWords)},
                             {"center", {s.center.x, s.center.y}},
                             {"conicOpacity", {s.conicOpacity.x, s.conicOpacity.y, s.conicOpacity.z, s.conicOpacity.w}},
                             {"radius", s.radius}};
            }
        }
        // Every exact tile must be in the stored set (the tail is a superset by construction).
        const auto [r0, r1] = coverageRows(cv, int(web::kTileSize));
        std::set<uint64_t> gpuTiles;
        web::forEachSpanTile(w, [&](uint32_t x, uint32_t y) { gpuTiles.insert(uint64_t(y) << 32 | x); });
        bool missing = false;
        std::vector<std::pair<int, int>> rowSpans;
        for (int r = r0; r < r1; ++r) {
            const auto [x0, x1] = coverageRowSpan(cv, r, int(web::kTileSize));
            rowSpans.push_back({x0, x1});
            for (int x = x0; x < x1; ++x) missing = missing || !gpuTiles.count(uint64_t(uint32_t(r)) << 32 | uint32_t(x));
        }
        if (missing) (borderline ? supersetBorderline : supersetViolations)++;
        // Tail overhead for other K (exact spans; the tail = the union of rows >= K).
        for (size_t k = 0; k < std::size(kList); ++k) {
            const uint32_t K = kList[k];
            if (rowSpans.size() <= K) continue;
            int lo = INT32_MAX, hi = INT32_MIN;
            uint64_t exactTail = 0;
            for (size_t r = K; r < rowSpans.size(); ++r) {
                if (rowSpans[r].second > rowSpans[r].first) {
                    lo = std::min(lo, rowSpans[r].first);
                    hi = std::max(hi, rowSpans[r].second);
                    exactTail += uint64_t(rowSpans[r].second - rowSpans[r].first);
                }
            }
            if (hi > lo) tailExtra[k] += uint64_t(hi - lo) * (rowSpans.size() - K) - exactTail;
        }
    }
    const bool ok = hardDiff == 0 && countMismatch == 0 && supersetViolations == 0 && cpuInconsistent == 0;
    pass = pass && ok;
    json overhead = json::object();
    for (size_t k = 0; k < std::size(kList); ++k)
        overhead["K=" + std::to_string(kList[k])] = exactTotal ? 100.0 * double(tailExtra[k]) / double(exactTotal) : 0.0;
    json j = {{"pass", ok},
              {"visible", visible},
              {"withTiles", counted},
              {"recordsEqualCpu", equal},
              {"borderlineDifferences", borderlineDiff},
              {"hardDifferences", hardDiff},
              {"countMismatches", countMismatch},
              {"cpuRecordsInconsistent", cpuInconsistent},
              {"exactTilesMissing", supersetViolations},
              {"exactTilesMissingBorderline", supersetBorderline},
              {"entriesStored", storedTotal},
              {"entriesExactTight", exactTotal},
              {"storedOverheadPct", exactTotal ? 100.0 * (double(storedTotal) - double(exactTotal)) / double(exactTotal) : 0.0},
              {"tailOverheadPctByK", overhead},
              {"rowsHistogram", {{"0", rowHist[0]}, {"1", rowHist[1]}, {"2", rowHist[2]}, {"3", rowHist[3]},
                                 {"4", rowHist[4]}, {"5", rowHist[5]}, {"6", rowHist[6]}, {"7", rowHist[7]},
                                 {"8", rowHist[8]}, {"9-12", rowHist[9]}, {"13-16", rowHist[10]},
                                 {"17-32", rowHist[11]}, {"33-64", rowHist[12]}, {">64", rowHist[13]}}}};
    if (!firstHard.is_null()) j["firstHardDifference"] = firstHard;
    return j;
}

// The CPU construction of the sorted list from the GPU's own splats and spans (INRIA's pipeline in
// binning form): visible splats with tiles in (depth bits, index) order, duplicated in that order
// through the stored records, at most `capacity` entries, then a stable sort by tile.
struct CpuList {
    std::vector<uint32_t> order;  // depth-sorted visible indices (with tiles)
    std::vector<uint32_t> keys, values, ranges;
    uint64_t duplicates = 0;
};

CpuList cpuTileList(const std::vector<ProjectedSplat>& splats, const std::vector<uint32_t>& tileCounts,
                    const std::vector<uint32_t>& spans, uint32_t tilesX, uint32_t tilesY, uint32_t capacity) {
    CpuList out;
    std::vector<uint32_t> keys, values, tmpK, tmpV;
    for (uint32_t i = 0; i < tileCounts.size(); ++i) {
        if (tileCounts[i] == 0) continue;
        keys.push_back(floatBits(splats[i].depth));
        values.push_back(i);
    }
    DepthSorter::radixSortPairs(keys, values, tmpK, tmpV);
    out.order = values;
    std::vector<uint32_t> listKeys, listValues;
    for (uint32_t g : out.order) {
        web::forEachSpanTile(spans.data() + size_t(g) * kCoverageSpanWords, [&](uint32_t x, uint32_t y) {
            ++out.duplicates;
            if (listKeys.size() < capacity) {
                listKeys.push_back(y * tilesX + x);
                listValues.push_back(g);
            }
        });
    }
    DepthSorter::radixSortPairs(listKeys, listValues, tmpK, tmpV);  // stable: (depth, index) order per tile
    out.keys = std::move(listKeys);
    out.values = std::move(listValues);
    const uint32_t tileCount = tilesX * tilesY;
    out.ranges.assign(size_t(tileCount) * 2, 0);
    for (size_t k = 0; k < out.keys.size(); ++k) {
        const uint32_t t = out.keys[k];
        if (t >= tileCount) continue;
        if (k == 0 || out.keys[k - 1] != t) out.ranges[2 * t] = uint32_t(k);
        if (k + 1 == out.keys.size() || out.keys[k + 1] != t) out.ranges[2 * t + 1] = uint32_t(k + 1);
    }
    return out;
}

double psnrClamped(const std::vector<float>& rgba, const std::vector<float>& rgb, double& maxErr, uint64_t& over) {
    double se = 0.0;
    maxErr = 0.0;
    over = 0;
    const size_t pixels = rgb.size() / 3;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c) {
            const double a = std::clamp(double(rgba[p * 4 + c]), 0.0, 1.0);
            const double b = std::clamp(double(rgb[p * 3 + c]), 0.0, 1.0);
            se += (a - b) * (a - b);
            maxErr = std::max(maxErr, std::abs(a - b));
            over += std::abs(a - b) > 2.0 / 255.0;
        }
    const double mse = se / double(pixels * 3);
    return mse > 0 ? 10.0 * std::log10(1.0 / mse) : 999.0;
}

}  // namespace

struct WgpuTileSelftest::Impl {
    WGPUDevice device;
    WGPUQueue queue;
    WgslShaderCache& shaders;
    WGPUBindGroupLayout sceneLayout;
    bool subgroups;
    std::function<SceneAccess()> sceneAccess;
    WGPULimits limits = WGPU_LIMITS_INIT;

    std::map<std::string, std::unique_ptr<Job>> jobs;
    std::map<std::string, json> results;
    std::unique_ptr<WgpuScanKernels> scanKernels[2];
    std::unique_ptr<WgpuSortKernels> sortKernels[2];

    Impl(WGPUDevice d, WGPUQueue q, WgslShaderCache& s, WGPUBindGroupLayout l, bool sg, std::function<SceneAccess()> a)
        : device(d), queue(q), shaders(s), sceneLayout(l), subgroups(sg), sceneAccess(std::move(a)) {
        wgpuDeviceGetLimits(device, &limits);
    }

    std::vector<bool> kernelModes() const { return subgroups ? std::vector<bool>{false, true} : std::vector<bool>{false}; }
    const WgpuScanKernels& scan(bool sg) {
        auto& k = scanKernels[sg];
        if (!k) {
            k = std::make_unique<WgpuScanKernels>();
            k->create(device, shaders, sg);
        }
        return *k;
    }
    const WgpuSortKernels& sort(bool sg) {
        auto& k = sortKernels[sg];
        if (!k) {
            k = std::make_unique<WgpuSortKernels>();
            k->create(device, shaders, sg);
        }
        return *k;
    }
    Job& newJob(const std::string& name) {
        auto job = std::make_unique<Job>();
        job->device = device;
        Job& j = *job;
        jobs[name] = std::move(job);
        return j;
    }

    void startScan();
    void startSort();
    void startTiles();  // coverage, tiles, image32
    void startScene();
};

// --- scan --------------------------------------------------------------------------------------
void WgpuTileSelftest::Impl::startScan() {
    struct Case {
        std::string name;
        uint32_t size, count, maxCount, flags, maxGroupsX;
    };
    const uint32_t G = WgpuScan::kGather, Z = WgpuScan::kNonZero;
    std::vector<Case> cases = {
        {"1", 1, 1, 1, 0, 65535},
        {"255", 255, 255, 255, Z, 65535},
        {"256", 256, 256, 256, G, 65535},
        {"257", 257, 257, 257, 0, 65535},
        {"2047 nonzero", 2047, 2047, 2047, Z, 65535},
        {"2048 gather", 2048, 2048, 2048, G, 65535},
        {"2049", 2049, 2049, 2049, 0, 65535},
        {"4097 gather+nonzero", 4097, 4097, 4097, G | Z, 65535},
        {"100000", 100000, 100000, 100000, 0, 65535},
        {"1000003 gather", 1000003, 1000003, 1000003, G, 65535},
        {"count 700 of 6144", 6144, 700, 6144, Z, 65535},
        {"count 5000 clamped to 3000", 3000, 5000, 3000, 0, 65535},
        {"empty", 100, 0, 100, 0, 65535},
        {"100000, 2D grid (3 groups per row)", 100000, 100000, 100000, G, 3},
    };
    Job& job = newJob("scan");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    struct Expect {
        std::string name;
        bool sg;
        std::vector<uint32_t> expected;  // size entries (sentinel past the count)
        uint32_t total;
        size_t outItem, ctlItem;
    };
    auto expects = std::make_shared<std::vector<Expect>>();
    std::mt19937 rng(1234);
    for (const Case& c : cases) {
        std::vector<uint32_t> input(c.size), index(c.size);
        for (uint32_t& v : input) v = rng() % 3 == 0 ? 0u : rng() % 10;
        std::iota(index.begin(), index.end(), 0u);
        std::shuffle(index.begin(), index.end(), rng);
        const uint32_t n = std::min(c.count, c.maxCount);
        std::vector<uint32_t> expected(c.size, kSentinel);
        uint32_t run = 0;
        for (uint32_t j = 0; j < n; ++j) {
            expected[j] = run;
            uint32_t v = (c.flags & G) ? input[index[j]] : input[j];
            if (c.flags & Z) v = v != 0;
            run += v;
        }
        for (bool sg : kernelModes()) {
            const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            WGPUBuffer in = makeBuffer(device, uint64_t(c.size) * 4, st, "scan test input");
            WGPUBuffer idx = makeBuffer(device, uint64_t(c.size) * 4, st, "scan test index");
            WGPUBuffer out = makeBuffer(device, uint64_t(c.size) * 4, st, "scan test output");
            WGPUBuffer ctl = makeBuffer(device, 64, st, "scan test control");
            wgpuQueueWriteBuffer(queue, in, 0, input.data(), size_t(c.size) * 4);
            wgpuQueueWriteBuffer(queue, idx, 0, index.data(), size_t(c.size) * 4);
            const std::vector<uint32_t> fill(c.size, kSentinel);
            wgpuQueueWriteBuffer(queue, out, 0, fill.data(), size_t(c.size) * 4);
            const uint32_t control[4] = {c.count, kSentinel, kSentinel, kSentinel};
            wgpuQueueWriteBuffer(queue, ctl, 0, control, sizeof(control));
            auto scanObj = std::make_shared<WgpuScan>();
            WgpuScan::Bindings b;
            b.input = in;
            b.index = (c.flags & G) ? idx : nullptr;
            b.output = out;
            b.control = ctl;
            b.countOffset = 0;
            b.totalOffset = 2;
            b.flags = c.flags;
            scanObj->create(device, queue, scan(sg), c.maxCount, b, c.maxGroupsX);
            WGPUComputePassEncoder pass = beginPass(encoder, "scan test");
            scanObj->encode(pass);
            endPass(pass);
            Expect e{c.name, sg, expected, run, 0, 0};
            e.outItem = job.copyBuffer(encoder, out, uint64_t(c.size) * 4);
            e.ctlItem = job.copyBuffer(encoder, ctl, 16);
            expects->push_back(std::move(e));
            job.keepAlive.push_back(scanObj);
            for (WGPUBuffer x : {in, idx, out, ctl}) job.keepAlive.push_back(holdBuffer(x));
        }
    }
    job.onComplete = [this, expects](Job& j) {
        json cs = json::array();
        bool pass = j.allOk();
        for (const Expect& e : *expects) {
            const auto out = j.as<uint32_t>(e.outItem, e.expected.size());
            const auto ctl = j.as<uint32_t>(e.ctlItem);
            uint64_t bad = 0;
            json first;
            for (size_t k = 0; k < e.expected.size() && k < out.size(); ++k) {
                if (out[k] != e.expected[k]) {
                    if (!bad++) first = {{"index", k}, {"got", out[k]}, {"expected", e.expected[k]}};
                }
            }
            const bool ok = bad == 0 && out.size() == e.expected.size() && ctl.size() >= 3 && ctl[2] == e.total;
            pass = pass && ok;
            json c = {{"name", e.name}, {"kernels", e.sg ? "subgroups" : "portable"}, {"pass", ok},
                      {"mismatches", bad}, {"total", ctl.size() >= 3 ? ctl[2] : 0}, {"expectedTotal", e.total}};
            if (!first.is_null()) c["firstMismatch"] = first;
            cs.push_back(c);
        }
        results["scan"] = {{"pass", pass}, {"cases", cs}};
    };
    job.submitAndMap(queue, encoder);
}

// --- sort --------------------------------------------------------------------------------------
void WgpuTileSelftest::Impl::startSort() {
    struct Case {
        std::string name;
        uint32_t size, count, maxCount, keyBits, maxGroupsX;
        int pattern;  // 0 random, 1 equal, 2 sorted, 3 reverse, 4 few distinct, 5 13-bit random, 6 depth-like floats
    };
    std::vector<Case> cases = {
        {"random 32-bit, 1000003", 1000003, 1000003, 1000003, 32, 65535, 0},
        {"all keys equal, 70000", 70000, 70000, 70000, 32, 65535, 1},
        {"sorted, 50000", 50000, 50000, 50000, 32, 65535, 2},
        {"reverse, 50001", 50001, 50001, 50001, 32, 65535, 3},
        {"8 distinct keys (stability), 300000", 300000, 300000, 300000, 32, 65535, 4},
        {"13-bit keys, 4 passes, 2000000", 2000000, 2000000, 2000000, 13, 65535, 5},
        {"depth-like float bits, 1244819", 1244819, 1244819, 1244819, 32, 65535, 6},
        {"1 key", 1, 1, 1, 32, 65535, 0},
        {"1024 keys", 1024, 1024, 1024, 32, 65535, 0},
        {"1025 keys", 1025, 1025, 1025, 32, 65535, 4},
        {"count 3000 of 10000", 10000, 3000, 10000, 32, 65535, 0},
        {"count 9000 clamped to 5000", 5000, 9000, 5000, 32, 65535, 4},
        {"0 keys", 64, 0, 64, 32, 65535, 0},
        {"100000, 2D grid (7 groups per row)", 100000, 100000, 100000, 32, 7, 4},
    };
    Job& job = newJob("sort");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    struct Expect {
        std::string name;
        bool sg;
        uint32_t passes;
        std::vector<uint32_t> keys, values;  // full buffer contents expected
        size_t keyItem, valueItem;
    };
    auto expects = std::make_shared<std::vector<Expect>>();
    std::mt19937 rng(777);
    for (const Case& c : cases) {
        std::vector<uint32_t> keys(c.size), values(c.size);
        for (uint32_t i = 0; i < c.size; ++i) {
            uint32_t k = 0;
            switch (c.pattern) {
            case 0: k = rng(); break;
            case 1: k = 0x3F800000u; break;
            case 2: k = i * 3u; break;
            case 3: k = (c.size - i) * 5u; break;
            case 4: k = (rng() & 7u) * 0x01010101u; break;
            case 5: k = rng() % 8161u; break;
            case 6: k = floatBits(0.2f + 30.0f * float(rng() % 100000) / 100000.0f); break;
            }
            keys[i] = k;
            values[i] = i;
        }
        const uint32_t n = std::min(c.count, c.maxCount);
        std::vector<uint32_t> ek(keys.begin(), keys.begin() + n), ev(values.begin(), values.begin() + n), tk, tv;
        DepthSorter::radixSortPairs(ek, ev, tk, tv);
        ek.insert(ek.end(), keys.begin() + n, keys.end());
        ev.insert(ev.end(), values.begin() + n, values.end());
        for (bool sg : kernelModes()) {
            const WGPUBufferUsage st = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            WGPUBuffer kb = makeBuffer(device, uint64_t(c.size) * 4, st, "sort test keys");
            WGPUBuffer vb = makeBuffer(device, uint64_t(c.size) * 4, st, "sort test values");
            WGPUBuffer cb = makeBuffer(device, 16, st, "sort test count");
            wgpuQueueWriteBuffer(queue, kb, 0, keys.data(), size_t(c.size) * 4);
            wgpuQueueWriteBuffer(queue, vb, 0, values.data(), size_t(c.size) * 4);
            const uint32_t cnt[4] = {kSentinel, c.count, 0, 0};
            wgpuQueueWriteBuffer(queue, cb, 0, cnt, sizeof(cnt));
            auto sorter = std::make_shared<WgpuRadixSort>();
            sorter->create(device, queue, sort(sg), c.maxCount, kb, vb, cb, 1, c.maxGroupsX);
            WGPUComputePassEncoder pass = beginPass(encoder, "sort test");
            sorter->encode(pass, c.keyBits);
            endPass(pass);
            Expect e{c.name, sg, WgpuRadixSort::passCount(c.keyBits), ek, ev, 0, 0};
            e.keyItem = job.copyBuffer(encoder, kb, uint64_t(c.size) * 4);
            e.valueItem = job.copyBuffer(encoder, vb, uint64_t(c.size) * 4);
            expects->push_back(std::move(e));
            job.keepAlive.push_back(sorter);
            for (WGPUBuffer x : {kb, vb, cb}) job.keepAlive.push_back(holdBuffer(x));
        }
    }
    job.onComplete = [this, expects](Job& j) {
        json cs = json::array();
        bool pass = j.allOk();
        for (const Expect& e : *expects) {
            const auto k = j.as<uint32_t>(e.keyItem, e.keys.size());
            const auto v = j.as<uint32_t>(e.valueItem, e.values.size());
            const bool ok = k == e.keys && v == e.values;
            uint64_t bad = 0;
            for (size_t i = 0; i < std::min(k.size(), e.keys.size()); ++i) bad += k[i] != e.keys[i] || v[i] != e.values[i];
            pass = pass && ok;
            cs.push_back({{"name", e.name}, {"kernels", e.sg ? "subgroups" : "portable"}, {"passes", e.passes},
                          {"pass", ok}, {"mismatches", bad}});
        }
        results["sort"] = {{"pass", pass}, {"cases", cs}};
    };
    job.submitAndMap(queue, encoder);
}

// --- coverage, tiles, image32: one synthetic frame through several renderer configurations ------
void WgpuTileSelftest::Impl::startTiles() {
    constexpr uint32_t W = 640, H = 360, N = 250000;
    auto scene = std::make_shared<GaussianScene>(makeSyntheticScene(N, 99));
    CompressionOptions co;
    co.threads = 1;
    const EncodedScene encoded = encodeScene(*scene, co);
    auto gpuScene = std::make_shared<WgpuScene>();
    std::string error;
    if (!gpuScene->createFromEncoded(device, queue, sceneLayout, encoded, &error)) {
        const json r = {{"pass", false}, {"error", "scene upload: " + error}};
        results["coverage"] = results["tiles"] = results["image32"] = r;
        return;
    }
    const glm::mat4 view = glm::lookAt(glm::vec3(0.3f, 0.5f, 6.0f), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(W) / float(H), 0.01f, 100.0f);
    proj[1][1] *= -1.0f;
    const SplatCamera cam = splatCameraFromViewer(view, proj, glm::mat4(1.0f), W, H);

    auto target = std::make_shared<WgpuTexture2D>();
    target->create(device, W, H, WGPUTextureFormat_RGBA32Float, WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc,
                   "selftest rgba32f target");

    struct Config {
        std::string name;
        WgpuTileSettings settings;
    };
    std::vector<Config> configs;
    WgpuTileSettings base;
    base.adaptiveList = false;
    base.fixedCapacity = 4u << 20;
    configs.push_back({"portable", base});
    {
        WgpuTileSettings s = base;
        s.boundsMode = 0;
        s.fixedCapacity = 12u << 20;
        configs.push_back({"rect bounds", s});
    }
    {
        WgpuTileSettings s = base;
        s.maxGroupsX = 7;
        configs.push_back({"2D grids (7 groups per row)", s});
    }
    {
        WgpuTileSettings s = base;
        s.fixedCapacity = 65536;
        configs.push_back({"overflow (65536 entries)", s});
    }
    if (subgroups) {
        WgpuTileSettings s = base;
        s.subgroups = true;
        configs.push_back({"subgroups", s});
    }

    Job& job = newJob("tiles");
    job.keepAlive.push_back(gpuScene);
    job.keepAlive.push_back(target);
    struct Out {
        std::string name;
        uint32_t capacity = 0;
        size_t control, keys, values, ranges, image;
        size_t splats = 0, tileCounts = 0, spans = 0, depthValues = 0;
    };
    auto outs = std::make_shared<std::vector<Out>>();
    uint32_t tilesX = 0, tilesY = 0;
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        auto renderer = std::make_shared<WgpuTileRenderer>();
        if (!renderer->create(device, queue, shaders, sceneLayout, limits, configs[ci].settings)) {
            const json r = {{"pass", false}, {"error", "renderer creation failed: " + configs[ci].name}};
            results["coverage"] = results["tiles"] = results["image32"] = r;
            return;
        }
        // One encoder per configuration: each renderer's queue.writeBuffer calls (parameters,
        // control reset) must land before ITS commands, so every configuration is its own submit.
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        renderer->encode(encoder, *gpuScene, *target, cam, 1);
        const WgpuTileRenderer::DebugBuffers b = renderer->debugBuffers();
        tilesX = b.tilesX;
        tilesY = b.tilesY;
        Out o;
        o.name = configs[ci].name;
        o.capacity = b.capacity;
        o.control = job.copyBuffer(encoder, b.control, sizeof(WebTileControl));
        o.keys = job.copyBuffer(encoder, b.listKeys, uint64_t(b.capacity) * 4);
        o.values = job.copyBuffer(encoder, b.listValues, uint64_t(b.capacity) * 4);
        o.ranges = job.copyBuffer(encoder, b.ranges, uint64_t(b.tilesX) * b.tilesY * 8);
        o.image = job.copyTexture(encoder, *target, 16);
        if (ci == 0) {
            o.splats = job.copyBuffer(encoder, b.splats, uint64_t(N) * sizeof(ProjectedSplat));
            o.tileCounts = job.copyBuffer(encoder, b.tileCounts, uint64_t(N) * 4);
            o.spans = job.copyBuffer(encoder, b.spans, uint64_t(N) * kCoverageSpanWords * 4);
            o.depthValues = job.copyBuffer(encoder, b.depthValues, uint64_t(N) * 4);
        }
        outs->push_back(o);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(queue, 1, &commands);
        wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);
        renderer->afterSubmit();
        job.keepAlive.push_back(renderer);
    }
    // The maps go with an empty final submit (all copies above are already queued).
    WGPUCommandEncoder last = wgpuDeviceCreateCommandEncoder(device, nullptr);
    job.onComplete = [this, outs, scene, cam, tilesX, tilesY](Job& j) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!j.allOk()) {
            const json r = {{"pass", false}, {"error", "readback failed"}};
            results["coverage"] = results["tiles"] = results["image32"] = r;
            return;
        }
        const Out& a = (*outs)[0];
        const auto splats = j.as<ProjectedSplat>(a.splats);
        const auto tileCounts = j.as<uint32_t>(a.tileCounts);
        const auto spans = j.as<uint32_t>(a.spans);
        const auto depthValues = j.as<uint32_t>(a.depthValues);
        auto controlOf = [&](const Out& o) {
            WebTileControl c{};
            std::memcpy(&c, j.items[o.control].data.data(), sizeof(c));
            return c;
        };
        const WebTileControl ca = controlOf(a);

        // ---- coverage ----
        {
            bool pass = ca.coverageMismatch == 0;
            json cov = coverageChecks(splats, tileCounts, spans, tilesX, tilesY, pass);
            // Visibility vs the CPU preprocess (near, opacity, singular, off screen).
            uint32_t visMismatch = 0, gpuVisible = 0;
            std::vector<glm::vec3> coeffs(scene->shCoeffsPerChannel());
            for (uint32_t i = 0; i < scene->count; ++i) {
                const glm::vec4 pa = scene->posAlpha[i];
                bool cpuVisible = false;
                if (viewDepth(cam.viewInria, glm::vec3(pa)) > kNearCullDepth && pa.w >= kMinAlpha) {
                    for (uint32_t k = 0; k < coeffs.size(); ++k) coeffs[k] = scene->shCoeff(i, k);
                    cpuVisible = preprocessGaussian(pa, glm::vec3(scene->scales[i]), scene->rotations[i], coeffs.data(),
                                                    scene->shDegree, cam)
                                     .cullReason == uint32_t(CullReason::Visible);
                }
                const bool gv = splats[i].radius > 0.0f;
                gpuVisible += gv;
                visMismatch += cpuVisible != gv;
            }
            const bool visOk = visMismatch == 0 && ca.cullCounts[0] == gpuVisible;
            pass = pass && visOk;
            cov["visibleCpuGpuMismatches"] = visMismatch;
            cov["visibleCounter"] = ca.cullCounts[0];
            cov["coverageMismatchCounter"] = ca.coverageMismatch;
            cov["pass"] = pass;
            results["coverage"] = cov;
        }

        // ---- tiles ----
        {
            json t = json::object();
            bool pass = true;
            const CpuList cpu = cpuTileList(splats, tileCounts, spans, tilesX, tilesY, a.capacity);
            const std::vector<uint32_t> gpuOrder(depthValues.begin(), depthValues.begin() + std::min<size_t>(ca.visible, depthValues.size()));
            const bool orderOk = gpuOrder == cpu.order;
            uint64_t orderDiff = 0;
            for (size_t k = 0; k < std::min(gpuOrder.size(), cpu.order.size()); ++k) orderDiff += gpuOrder[k] != cpu.order[k];
            t["depthOrder"] = {{"pass", orderOk}, {"visible", ca.visible}, {"cpuVisible", cpu.order.size()},
                               {"mismatches", orderDiff}};
            pass = pass && orderOk;
            auto checkList = [&](const Out& o, const CpuList& expect, const char* label) {
                const WebTileControl c = controlOf(o);
                const auto keys = j.as<uint32_t>(o.keys, c.entries);
                const auto values = j.as<uint32_t>(o.values, c.entries);
                const auto ranges = j.as<uint32_t>(o.ranges);
                uint64_t bad = 0, badRanges = 0;
                for (size_t k = 0; k < std::min(keys.size(), expect.keys.size()); ++k)
                    bad += keys[k] != expect.keys[k] || values[k] != expect.values[k];
                bad += keys.size() != expect.keys.size();
                for (size_t k = 0; k < std::min(ranges.size(), expect.ranges.size()); ++k) badRanges += ranges[k] != expect.ranges[k];
                const bool ok = bad == 0 && badRanges == 0 && c.duplicates == expect.duplicates &&
                                c.entries == std::min<uint64_t>(expect.duplicates, o.capacity) &&
                                c.overflow == expect.duplicates - c.entries && c.coverageMismatch == 0;
                pass = pass && ok;
                t[label] = {{"pass", ok}, {"duplicates", c.duplicates}, {"entries", c.entries},
                            {"overflow", c.overflow}, {"capacity", o.capacity}, {"listMismatches", bad},
                            {"rangeMismatches", badRanges}, {"listGroups", {c.listGroups[0], c.listGroups[1]}}};
            };
            checkList(a, cpu, "portable vs CPU");
            for (const Out& o : *outs) {
                if (o.name.rfind("overflow", 0) == 0) {
                    checkList(o, cpuTileList(splats, tileCounts, spans, tilesX, tilesY, o.capacity), "overflow vs CPU");
                }
            }
            // Configurations that must give the same list (2D grids, subgroups) and image.
            const auto imageA = j.as<uint8_t>(a.image);
            for (const Out& o : *outs) {
                if (&o == &a || o.name.rfind("overflow", 0) == 0) continue;
                const bool rect = o.name == "rect bounds";
                WebTileControl c = controlOf(o);
                // The list dispatch arguments differ by design in the 2D-grid configuration.
                WebTileControl cmpA = ca, cmpO = c;
                std::fill(std::begin(cmpA.listGroups), std::end(cmpA.listGroups), 0u);
                std::fill(std::begin(cmpO.listGroups), std::end(cmpO.listGroups), 0u);
                const bool sameImage = j.items[o.image].data == j.items[a.image].data;
                bool sameList = true;
                if (!rect) {
                    sameList = j.items[o.keys].data == j.items[a.keys].data &&
                               j.items[o.values].data == j.items[a.values].data &&
                               j.items[o.ranges].data == j.items[a.ranges].data &&
                               std::memcmp(&cmpO, &cmpA, sizeof(c)) == 0;
                }
                const bool ok = sameImage && sameList && c.overflow == 0;
                pass = pass && ok;
                json r = {{"pass", ok}, {"identicalImage", sameImage}, {"duplicates", c.duplicates}, {"overflow", c.overflow},
                          {"listGroups", {c.listGroups[0], c.listGroups[1]}}};
                if (!rect) r["identicalListRangesControl"] = sameList;
                else r["entriesRectOverTight"] = ca.duplicates ? double(c.duplicates) / double(ca.duplicates) : 0.0;
                t[o.name + " vs portable"] = r;
            }
            t["pass"] = pass;
            results["tiles"] = t;
        }

        // ---- image32 vs the CPU reference renderer ----
        {
            ReferenceOptions ro;
            ro.threads = 1;
            ro.bounds = TileBounds::Tight;
            const auto tr = std::chrono::steady_clock::now();
            const ReferenceImage ref = renderReference(*scene, cam, ro);
            const double refMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
            const auto img = j.texture<float>(a.image, cam.width, cam.height, 4);
            double maxErr = 0.0;
            uint64_t over = 0;
            const double psnr = psnrClamped(img, ref.rgb, maxErr, over);
            const bool ok = psnr > 60.0 && ref.stats.terminatedPixels > 0;
            results["image32"] = {{"pass", ok}, {"psnr", psnr}, {"maxAbs255", maxErr * 255.0}, {"channelsOver2", over},
                                  {"width", cam.width}, {"height", cam.height},
                                  {"referenceTerminatedPixels", ref.stats.terminatedPixels},
                                  {"referenceMs", refMs}};
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        results["tiles"]["cpuCheckMs"] = ms;
    };
    job.submitAndMap(queue, last);
}

// --- the app's current scene (analysis + coverage checks on real data) --------------------------
void WgpuTileSelftest::Impl::startScene() {
    const SceneAccess access = sceneAccess ? sceneAccess() : SceneAccess{};
    if (!access.renderer || !access.renderer->valid() || !access.renderer->lastStats().valid) {
        results["scene"] = {{"pass", false}, {"error", "no splat-tiles frame rendered yet"}};
        return;
    }
    const WgpuTileRenderer::DebugBuffers b = access.renderer->debugBuffers();
    Job& job = newJob("scene");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    const size_t iSplats = job.copyBuffer(encoder, b.splats, uint64_t(b.count) * sizeof(ProjectedSplat));
    const size_t iCounts = job.copyBuffer(encoder, b.tileCounts, uint64_t(b.count) * 4);
    const size_t iSpans = job.copyBuffer(encoder, b.spans, uint64_t(b.count) * kCoverageSpanWords * 4);
    const size_t iControl = job.copyBuffer(encoder, b.control, sizeof(WebTileControl));
    const SplatCamera cam = access.camera;
    const uint32_t tilesX = b.tilesX, tilesY = b.tilesY;
    job.onComplete = [this, iSplats, iCounts, iSpans, iControl, cam, tilesX, tilesY](Job& j) {
        if (!j.allOk()) {
            results["scene"] = {{"pass", false}, {"error", "readback failed"}};
            return;
        }
        const auto splats = j.as<ProjectedSplat>(iSplats);
        const auto counts = j.as<uint32_t>(iCounts);
        const auto spans = j.as<uint32_t>(iSpans);
        WebTileControl c{};
        std::memcpy(&c, j.items[iControl].data.data(), sizeof(c));
        bool pass = c.coverageMismatch == 0;
        json r = coverageChecks(splats, counts, spans, tilesX, tilesY, pass);
        r["pass"] = pass;
        r["control"] = {{"gaussians", c.gaussianCount}, {"visible", c.visible}, {"duplicates", c.duplicates},
                        {"entries", c.entries}, {"overflow", c.overflow}, {"mismatch", c.coverageMismatch}};
        r["tiles"] = {tilesX, tilesY};
        (void)cam;
        results["scene"] = r;
    };
    job.submitAndMap(queue, encoder);
}

WgpuTileSelftest::WgpuTileSelftest(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders,
                                   WGPUBindGroupLayout sceneLayout, bool subgroupsAvailable,
                                   std::function<SceneAccess()> sceneAccess)
    : impl_(std::make_unique<Impl>(device, queue, shaders, sceneLayout, subgroupsAvailable, std::move(sceneAccess))) {}

WgpuTileSelftest::~WgpuTileSelftest() = default;

bool WgpuTileSelftest::known(const std::string& name) {
    return name == "scan" || name == "sort" || name == "coverage" || name == "tiles" || name == "image32" ||
           name == "scene";
}

std::string WgpuTileSelftest::run(const std::string& name) {
    Impl& s = *impl_;
    if (auto it = s.results.find(name); it != s.results.end()) {
        json r = it->second;
        r["subgroupsTested"] = s.subgroups;
        return r.dump();
    }
    // "scene" can be re-run (a new analysis per call once the previous one finished).
    const std::string jobName = (name == "coverage" || name == "image32") ? "tiles" : name;
    if (!s.jobs.count(jobName)) {
        if (jobName == "scan") s.startScan();
        else if (jobName == "sort") s.startSort();
        else if (jobName == "tiles") s.startTiles();
        else if (jobName == "scene") s.startScene();
        else return json{{"pass", false}, {"error", "unknown self-test " + name}}.dump();
        if (auto it = s.results.find(name); it != s.results.end()) return it->second.dump();  // failed at start
    }
    return R"({"pending":true})";
}

}  // namespace gsr
