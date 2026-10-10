// GPU tests for Phase 3 (CTest `gpu_sort`), headless Vulkan compute:
//
//   1. GpuRadixSort vs the CPU radix sort (DepthSorter::radixSortPairs) BIT FOR BIT, keys
//      and values, on random and adversarial inputs: sizes 0, 1, around the workgroup
//      (256) and block (2048) sizes, 1M+3, 2M+3; all-equal keys, sorted, reversed, few
//      distinct values (stability), single varying byte, float depths, full 32-bit
//      range. Entries past the count are filled with a sentinel and must stay untouched.
//      The count is read from a GPU buffer (also tested: clamping to the capacity).
//   2. The splat-gpusort renderer on a synthetic scene: the compacted visible set equals
//      the CPU's (near cull + opacity cull + INRIA preprocess, which includes the
//      off-screen / frustum test), the cull counters match per reason, the GPU keys equal
//      the CPU depth bits, and the sorted list equals the CPU DepthSorter order on the
//      visible set (exactly, up to the order inside runs of EQUAL keys, see below).
//
// `test_gpu_sort --benchmark` instead prints sort throughput (Mkeys/s) vs size.
// Exit code: 0 = all checks passed, 1 = a check failed, 2 = no usable GPU / setup error.
#include "core/context.h"
#include "core/profiler.h"
#include "core/resources.h"
#include "render/gpu_radix_sort.h"
#include "render/gpu_scene.h"
#include "render/splat_gpusort_renderer.h"
#include "scene/gaussian_scene.h"
#include "splat/depth_sort.h"
#include "splat/splat_math.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
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

// ---------------------------------------------------------------------------------------
// Sort harness: upload keys/values + count, sort on the GPU, read everything back.
// ---------------------------------------------------------------------------------------
class SortHarness {
public:
    SortHarness(VulkanContext& ctx, uint32_t capacity) : m_ctx(ctx), m_capacity(capacity) {
        const VkDeviceSize bytes = VkDeviceSize(std::max(capacity, 1u)) * 4;
        const VkBufferUsageFlags usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        m_keys = createBuffer(ctx, bytes, usage, MemoryUsage::GpuOnly, "test: keys");
        m_values = createBuffer(ctx, bytes, usage, MemoryUsage::GpuOnly, "test: values");
        m_count = createBuffer(ctx, 16, usage, MemoryUsage::GpuOnly, "test: count");
        m_upload = createBuffer(ctx, 2 * bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload, "test: upload");
        m_readback = createBuffer(ctx, 2 * bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback,
                                  "test: readback");
        // Count at byte offset 8 (not 0) to exercise the offset path.
        m_sort.init(ctx, capacity, m_keys.buffer, m_values.buffer, m_count.buffer, 8);
    }
    ~SortHarness() {
        m_sort.destroy(m_ctx);
        for (Buffer* b : {&m_keys, &m_values, &m_count, &m_upload, &m_readback}) destroyBuffer(m_ctx, *b);
    }

    // Uploads the whole capacity (keys/values padded with kSentinel), sets the GPU count
    // to `gpuCount`, sorts and reads the whole capacity back.
    void run(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& values, uint32_t gpuCount,
             std::vector<uint32_t>& outKeys, std::vector<uint32_t>& outValues, GpuProfiler* profiler = nullptr,
             int repeats = 1) {
        const size_t cap = std::max(m_capacity, 1u);
        auto* up = static_cast<uint32_t*>(m_upload.mapped);
        std::fill(up, up + 2 * cap, kSentinel);
        std::copy(keys.begin(), keys.end(), up);
        std::copy(values.begin(), values.end(), up + cap);
        VK_CHECK(vmaFlushAllocation(m_ctx.allocator, m_upload.allocation, 0, VK_WHOLE_SIZE));
        const VkDeviceSize bytes = VkDeviceSize(cap) * 4;
        m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const VkBufferCopy k{0, 0, bytes}, v{bytes, 0, bytes};
            vkCmdCopyBuffer(cmd, m_upload.buffer, m_keys.buffer, 1, &k);
            vkCmdCopyBuffer(cmd, m_upload.buffer, m_values.buffer, 1, &v);
            const uint32_t countWords[4] = {kSentinel, kSentinel, gpuCount, kSentinel};
            vkCmdUpdateBuffer(cmd, m_count.buffer, 0, sizeof(countWords), countWords);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
            m_sort.record(cmd, nullptr);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            const VkBufferCopy rk{0, 0, bytes}, rv{0, bytes, bytes};
            vkCmdCopyBuffer(cmd, m_keys.buffer, m_readback.buffer, 1, &rk);
            vkCmdCopyBuffer(cmd, m_values.buffer, m_readback.buffer, 1, &rv);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
        VK_CHECK(vmaInvalidateAllocation(m_ctx.allocator, m_readback.allocation, 0, VK_WHOLE_SIZE));
        const auto* rb = static_cast<const uint32_t*>(m_readback.mapped);
        outKeys.assign(rb, rb + cap);
        outValues.assign(rb + cap, rb + 2 * cap);

        // Timing: sort the (already sorted) data again `repeats` times. Sorted input is
        // not a special case for an LSD radix sort (same work), but to time realistic
        // inputs the caller re-uploads; here we re-upload before each timed sort.
        for (int r = 0; r < repeats && profiler; ++r) {
            profiler->beginFrame(0, uint64_t(r));
            m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
                const VkBufferCopy k{0, 0, bytes}, v{bytes, 0, bytes};
                vkCmdCopyBuffer(cmd, m_upload.buffer, m_keys.buffer, 1, &k);
                vkCmdCopyBuffer(cmd, m_upload.buffer, m_values.buffer, 1, &v);
                memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
                auto total = profiler->scope(cmd, "sort (total)");
                m_sort.record(cmd, profiler);
            });
        }
        if (profiler) profiler->collectAll();
    }

    uint32_t capacity() const { return m_capacity; }
    void setVariant(GpuRadixSort::Variant v) { m_sort.setVariant(v); }

private:
    VulkanContext& m_ctx;
    uint32_t m_capacity;
    Buffer m_keys, m_values, m_count, m_upload, m_readback;
    GpuRadixSort m_sort;
};

// One sort case: GPU vs CPU, bit for bit, plus the sentinel tail.
void checkSort(SortHarness& h, const char* name, const std::vector<uint32_t>& keys,
               const std::vector<uint32_t>& values, uint32_t gpuCount) {
    std::vector<uint32_t> gk, gv;
    h.run(keys, values, gpuCount, gk, gv);
    const uint32_t n = std::min<uint32_t>(gpuCount, h.capacity());
    std::vector<uint32_t> ck(keys.begin(), keys.begin() + n), cv(values.begin(), values.begin() + n), t1, t2;
    DepthSorter::radixSortPairs(ck, cv, t1, t2);

    uint32_t keyMismatch = 0, valueMismatch = 0, firstBad = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
        if (gk[i] != ck[i]) ++keyMismatch;
        if (gv[i] != cv[i]) ++valueMismatch;
        if ((gk[i] != ck[i] || gv[i] != cv[i]) && firstBad == UINT32_MAX) firstBad = i;
    }
    uint32_t tailTouched = 0;
    for (size_t i = n; i < gk.size(); ++i) {
        const uint32_t expectKey = i < keys.size() ? keys[i] : kSentinel;
        const uint32_t expectValue = i < values.size() ? values[i] : kSentinel;
        if (gk[i] != expectKey || gv[i] != expectValue) ++tailTouched;
    }
    const bool ok = keyMismatch == 0 && valueMismatch == 0 && tailTouched == 0;
    std::printf("  %-34s n=%-8u %s\n", name, n, ok ? "ok" : "MISMATCH");
    CHECK(keyMismatch == 0 && valueMismatch == 0, "%s: %u key / %u value mismatches, first at %u (gpu %08x/%u cpu %08x/%u)",
          name, keyMismatch, valueMismatch, firstBad, firstBad < n ? gk[firstBad] : 0, firstBad < n ? gv[firstBad] : 0,
          firstBad < n ? ck[firstBad] : 0, firstBad < n ? cv[firstBad] : 0);
    CHECK(tailTouched == 0, "%s: %u entries past the count were modified", name, tailTouched);
}

std::vector<uint32_t> iota(uint32_t n) {
    std::vector<uint32_t> v(n);
    std::iota(v.begin(), v.end(), 0u);
    return v;
}

const char* variantName(GpuRadixSort::Variant v) {
    return v == GpuRadixSort::Variant::Classic ? "classic" : "reorder";
}

void testSort(VulkanContext& ctx, GpuRadixSort::Variant variant) {
    std::printf("GPU radix sort (%s scatter) vs CPU radix sort (bit-exact keys + values)\n", variantName(variant));
    constexpr uint32_t kCapacity = 2'000'003;
    SortHarness h(ctx, kCapacity + 5);
    h.setVariant(variant);
    std::mt19937 rng(1234);
    auto randomKeys = [&](uint32_t n) {
        std::vector<uint32_t> k(n);
        for (auto& x : k) x = rng();
        return k;
    };

    // Sizes around the workgroup (256) and block (2048) boundaries, and large ones.
    for (uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 2047u, 2048u, 2049u, 4095u, 4097u, 65537u,
                       1'000'003u, kCapacity}) {
        const std::string name = "random 32-bit keys";
        checkSort(h, name.c_str(), randomKeys(n), iota(n), n);
    }

    const uint32_t n = 300'001;
    checkSort(h, "all equal (1.0f)", std::vector<uint32_t>(n, floatBits(1.0f)), iota(n), n);
    checkSort(h, "all zero", std::vector<uint32_t>(n, 0u), iota(n), n);
    checkSort(h, "all 0xFFFFFFFF", std::vector<uint32_t>(n, 0xFFFFFFFFu), iota(n), n);
    {
        std::vector<uint32_t> k = randomKeys(n);
        std::sort(k.begin(), k.end());
        checkSort(h, "already sorted", k, iota(n), n);
        std::reverse(k.begin(), k.end());
        checkSort(h, "reversed", k, iota(n), n);
    }
    {
        std::vector<uint32_t> k(n);
        for (auto& x : k) x = (rng() % 4) * 0x01010101u;  // 4 values, every byte differs
        checkSort(h, "4 distinct keys (stability)", k, iota(n), n);
        for (auto& x : k) x = rng() % 1000;
        checkSort(h, "keys < 1000 (many ties)", k, iota(n), n);
        for (auto& x : k) x = (rng() & 0xFFu) << 24;
        checkSort(h, "only the top byte varies", k, iota(n), n);
        for (auto& x : k) x = rng() & 0xFFu;
        checkSort(h, "only the low byte varies", k, iota(n), n);
    }
    {
        // Depth-like keys: positive floats over 0.2 .. 1000 (what the renderer sorts),
        // values a random permutation (they must travel with their keys).
        std::uniform_real_distribution<float> logDepth(std::log(0.2f), std::log(1000.0f));
        std::vector<uint32_t> k(n), v = iota(n);
        for (auto& x : k) x = floatBits(std::exp(logDepth(rng)));
        std::shuffle(v.begin(), v.end(), rng);
        checkSort(h, "float depths, permuted values", k, v, n);
    }
    // Count larger than the capacity: clamped on the GPU.
    {
        const uint32_t cap = h.capacity();
        checkSort(h, "count > capacity (clamped)", randomKeys(cap), iota(cap), cap + 1000);
    }
    // Count smaller than the uploaded data: only the prefix is sorted, the rest untouched.
    {
        std::vector<uint32_t> k = randomKeys(5000);
        checkSort(h, "count 3000 of 5000 uploaded", k, iota(5000), 3000);
    }
}

// ---------------------------------------------------------------------------------------
// Compaction + culling + sort inside the renderer, vs the CPU.
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
        // the camera, inside the near plane, off screen, very large or almost transparent.
        glm::vec3 p(u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f, u(rng) * 16.0f - 8.0f);
        float alpha = u(rng);
        if (i % 7 == 0) alpha *= 0.01f;  // ~1/3 of these are below 1/255
        if (i % 1000 == 0) p.z = 5.9f;   // within 0.2 of the camera plane
        if (i % 4099 == 0) {             // EXACTLY equal depths: ties for the sort
            p.z = -1.0f;
        }
        sc.posAlpha[i] = glm::vec4(p, alpha);
        const float s = std::exp(u(rng) * 4.0f - 6.0f);  // 0.0025 .. 0.135
        sc.scales[i] = glm::vec4(s, s * (0.2f + u(rng)), s * (0.2f + u(rng)), 0.0f);
        glm::vec4 q(u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f, u(rng) - 0.5f);
        sc.rotations[i] = glm::normalize(q);
        for (uint32_t k = 0; k < K * 3; ++k) sc.sh[size_t(i) * K * 3 + k] = (u(rng) - 0.5f) * (k < 3 ? 2.0f : 0.5f);
    }
    sc.stats.robustMin = glm::vec3(-8.0f);
    sc.stats.robustMax = glm::vec3(8.0f);
    return sc;
}

void testCompaction(VulkanContext& ctx, GpuRadixSort::Variant variant) {
    std::printf("Culling + stream compaction + sort (splat-gpusort, %s scatter) vs CPU\n", variantName(variant));
    const uint32_t W = 640, H = 360;
    const GaussianScene sc = makeSyntheticScene(250'000, 99);
    GpuScene gpu;
    gpu.upload(ctx, sc);

    GpuProfiler profiler;
    profiler.init(ctx, 1);
    CpuProfiler cpuProfiler;
    Image color = createImage2D(ctx, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                "test: color");
    Image depth{};

    SplatGpuSortRenderer renderer;
    GpuSortSettings sortSettings;
    sortSettings.radixVariant = variant;
    renderer.init(ctx, VK_FORMAT_R8G8B8A8_UNORM, gpu, sc, 1, SplatSettings{}, sortSettings);
    renderer.onRenderTargetsChanged(ctx, color, depth);

    // Camera at (0, 0.5, 6) looking at the origin, 60 deg vertical FOV, our viewer's
    // conventions (Vulkan clip space, y flipped like Camera::projection).
    const glm::mat4 view = glm::lookAt(glm::vec3(0.3f, 0.5f, 6.0f), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(W) / float(H), 0.01f, 100.0f);
    proj[1][1] *= -1.0f;
    const glm::mat4 model(1.0f);

    for (uint64_t frame = 0; frame < 2; ++frame) {  // the second frame reuses the buffers
        profiler.beginFrame(0, frame);
        ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const FrameInput in{cmd,  profiler, cpuProfiler, color, depth,
                                {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE},
                                {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE},
                                view, proj, model, frame, 0};
            renderer.render(in);
        });
    }
    std::vector<uint32_t> gk, gv;
    SplatControl control{};
    renderer.readBack(gk, gv, control);

    // CPU: the same decisions. Near cull exactly as DepthSorter, then the opacity cull,
    // then the INRIA preprocess (singular covariance, off-screen tile rectangle).
    const SplatCamera cam = splatCameraFromViewer(view, proj, model, W, H);
    std::vector<uint32_t> cpuCounts(kCullReasonCount, 0);
    std::vector<uint8_t> cpuVisible(sc.count, 0);
    std::vector<glm::vec3> coeffs(sc.shCoeffsPerChannel());
    for (uint32_t i = 0; i < sc.count; ++i) {
        const glm::vec4 pa = sc.posAlpha[i];
        uint32_t reason;
        const float z = viewDepth(cam.viewInria, glm::vec3(pa));
        if (!(z > kNearCullDepth)) {
            reason = uint32_t(CullReason::NearPlane);
        } else if (pa.w < kMinAlpha) {
            reason = uint32_t(CullReason::LowOpacity);
        } else {
            for (uint32_t k = 0; k < coeffs.size(); ++k) coeffs[k] = sc.shCoeff(i, k);
            reason = preprocessGaussian(pa, glm::vec3(sc.scales[i]), sc.rotations[i], coeffs.data(), sc.shDegree, cam)
                         .cullReason;
        }
        cpuCounts[reason]++;
        cpuVisible[i] = reason == uint32_t(CullReason::Visible);
    }

    // 1) Visible set.
    std::vector<uint8_t> gpuVisible(sc.count, 0);
    uint32_t duplicates = 0, outOfRange = 0;
    for (uint32_t v : gv) {
        if (v >= sc.count) {
            ++outOfRange;
            continue;
        }
        duplicates += gpuVisible[v];
        gpuVisible[v] = 1;
    }
    uint32_t onlyCpu = 0, onlyGpu = 0;
    for (uint32_t i = 0; i < sc.count; ++i) {
        onlyCpu += cpuVisible[i] && !gpuVisible[i];
        onlyGpu += gpuVisible[i] && !cpuVisible[i];
    }
    std::printf("  visible: gpu %u, cpu %u; only cpu %u, only gpu %u, duplicates %u\n", control.draw.instanceCount,
                cpuCounts[0], onlyCpu, onlyGpu, duplicates);
    CHECK(outOfRange == 0 && duplicates == 0, "compact list: %u out of range, %u duplicates", outOfRange, duplicates);
    CHECK(onlyCpu == 0 && onlyGpu == 0, "visible set differs: %u only on the CPU, %u only on the GPU", onlyCpu, onlyGpu);
    CHECK(control.draw.vertexCount == 6 && control.draw.instanceCount == gv.size(), "draw args %u x %u",
          control.draw.vertexCount, control.draw.instanceCount);

    // 2) Cull counters per reason.
    for (uint32_t r = 0; r < kCullReasonCount; ++r) {
        std::printf("  %-16s gpu %7u  cpu %7u\n", cullReasonName(r), control.cullCounts[r], cpuCounts[r]);
        CHECK(control.cullCounts[r] == cpuCounts[r], "cull counter %s", cullReasonName(r));
    }

    // 3) Keys = CPU depth bits, and the order = DepthSorter's on the visible set.
    uint32_t keyMismatch = 0;
    for (size_t j = 0; j < gv.size(); ++j)
        if (gv[j] < sc.count && gk[j] != floatBits(viewDepth(cam.viewInria, glm::vec3(sc.posAlpha[gv[j]]))))
            ++keyMismatch;
    CHECK(keyMismatch == 0, "%u GPU keys differ from the CPU depth bits", keyMismatch);

    std::vector<uint32_t> ck, cv, t1, t2;
    for (uint32_t i = 0; i < sc.count; ++i)
        if (cpuVisible[i]) {
            ck.push_back(floatBits(viewDepth(cam.viewInria, glm::vec3(sc.posAlpha[i]))));
            cv.push_back(i);  // index order, like DepthSorter
        }
    DepthSorter::radixSortPairs(ck, cv, t1, t2);
    uint32_t orderMismatch = 0, tieOnlyMismatch = 0, tiedEntries = 0;
    if (ck.size() == gk.size()) {
        for (size_t j = 0; j < ck.size(); ++j) {
            const bool tied = (j > 0 && ck[j - 1] == ck[j]) || (j + 1 < ck.size() && ck[j + 1] == ck[j]);
            tiedEntries += tied;
            if (gk[j] != ck[j]) ++orderMismatch;  // keys must match exactly, position by position
            else if (gv[j] != cv[j]) (tied ? tieOnlyMismatch : orderMismatch)++;
        }
    }
    std::printf("  sorted order: %u mismatches outside equal-key runs; %u of %u entries have a tied key, %u of them "
                "in a different order (compaction order, see phase3.md)\n",
                orderMismatch, tiedEntries, uint32_t(ck.size()), tieOnlyMismatch);
    CHECK(ck.size() == gk.size() && orderMismatch == 0, "sorted order differs from DepthSorter (%u)", orderMismatch);
    CHECK(tiedEntries > 0, "the synthetic scene was meant to contain tied depths");

    // Within each run of equal keys the GPU order is the compaction order: every visible
    // index of the run must still be present exactly once (a permutation of the run).
    uint32_t badRuns = 0;
    for (size_t j = 0; j < ck.size();) {
        size_t e = j;
        while (e < ck.size() && ck[e] == ck[j]) ++e;
        std::vector<uint32_t> a(gv.begin() + j, gv.begin() + e), b(cv.begin() + j, cv.begin() + e);
        std::sort(a.begin(), a.end());
        if (a != b) ++badRuns;
        j = e;
    }
    CHECK(badRuns == 0, "%u equal-key runs are not permutations of the CPU run", badRuns);

    renderer.destroy(ctx);
    destroyImage(ctx, color);
    profiler.destroy();
    gpu.destroy(ctx);
}

// ---------------------------------------------------------------------------------------
// Throughput (--benchmark)
// ---------------------------------------------------------------------------------------
void benchmarkSort(VulkanContext& ctx) {
    std::printf("GPU radix sort throughput, %s, 32-bit keys + 32-bit values, 4 passes x 8 bits\n",
                ctx.deviceName.c_str());
    std::printf("| scatter | keys | distribution | sort ms | Mkeys/s | setup | histogram (4x) | scan (4x) | scatter (4x) |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");
    const uint32_t maxN = 1u << 24;
    SortHarness h(ctx, maxN);
    std::mt19937 rng(7);
    for (auto variant : {GpuRadixSort::Variant::Classic, GpuRadixSort::Variant::Reorder})
    for (int dist = 0; dist < 2; ++dist) {
        h.setVariant(variant);
        for (uint32_t n = 1u << 14; n <= maxN; n <<= 1) {
            std::vector<uint32_t> keys(n);
            std::uniform_real_distribution<float> logDepth(std::log(0.2f), std::log(100.0f));
            for (auto& k : keys) k = dist == 0 ? rng() : floatBits(std::exp(logDepth(rng)));
            GpuProfiler profiler;
            profiler.init(ctx, 1);
            std::vector<uint32_t> gk, gv;
            h.run(keys, iota(n), n, gk, gv, &profiler, 3);  // warm-up
            profiler.resetAccumulators(0);
            h.run(keys, iota(n), n, gk, gv, &profiler, 20);
            std::map<std::string, double> avg;
            for (const auto& s : profiler.stats())
                if (s.count) avg[s.name] = s.sumMs / double(s.count);
            const double ms = avg["sort (total)"];
            std::printf("| %s | %u | %s | %.3f | %.0f | %.3f | %.3f | %.3f | %.3f |\n", variantName(variant), n,
                        dist == 0 ? "uniform 32-bit" : "depths 0.2..100", ms, double(n) / (ms * 1e3),
                        avg["sort: setup"], avg["sort: histogram"], avg["sort: scan"], avg["sort: scatter"]);
            profiler.destroy();
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool benchmark = false, validation = kDebugBuild;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--benchmark") benchmark = true;
        else if (a == "--validation") validation = true;
        else if (a == "--no-validation") validation = false;
    }
    VulkanContext ctx;
    try {
        ctx.init(nullptr, validation);
    } catch (const std::exception& e) {
        std::printf("cannot create a headless Vulkan 1.3 device: %s\n", e.what());
        return 2;
    }
    std::printf("Device: %s (%s), subgroup size %u (min %u, max %u), validation %s\n", ctx.deviceName.c_str(),
                ctx.driverVersion.c_str(), ctx.subgroup.size, ctx.subgroup.minSize, ctx.subgroup.maxSize,
                ctx.validationEnabled ? "on" : "off");
    try {
        if (benchmark) {
            benchmarkSort(ctx);
        } else {
            for (auto v : {GpuRadixSort::Variant::Classic, GpuRadixSort::Variant::Reorder}) {
                testSort(ctx, v);
                testCompaction(ctx, v);
            }
        }
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        ctx.destroy();
        return 2;
    }
    ctx.destroy();
    const auto& vc = VulkanContext::validationCounters();
    std::printf("Validation messages: %u errors, %u warnings\n", vc.errors.load(), vc.warnings.load());
    CHECK(vc.errors == 0 && vc.warnings == 0, "validation messages");
    if (benchmark) return g_failures ? 1 : 0;
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
