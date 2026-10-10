#include "render/gpu_radix_sort.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace gsr {

namespace {

// Must match SortPushConstants in shaders/radix_sort_common.glsl.
struct SortPushConstants {
    uint32_t shift;
    uint32_t countOffset;  // in uints
    uint32_t maxCount;
};

// Must match struct SortArgs in shaders/radix_sort_common.glsl.
struct SortArgs {
    VkDispatchIndirectCommand dispatch;  // groupsX = numBlocks
    uint32_t count;
    uint32_t numBlocks;
};
static_assert(sizeof(SortArgs) == 20, "layout must match radix_sort_common.glsl");

constexpr uint32_t kWorkgroupSize = 256;  // SORT_WG_SIZE

// Every sort step reads what the previous one wrote (and may overwrite what it read):
// a full compute -> compute memory barrier between consecutive dispatches.
void computeToCompute(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

}  // namespace

uint32_t GpuRadixSort::passCount(uint32_t keyBits) {
    const uint32_t passes = std::max(1u, (keyBits + kRadixBits - 1) / kRadixBits);
    return (passes + 1) & ~1u;  // even: the data must end up back in the caller's buffers
}

void GpuRadixSort::init(VulkanContext& ctx, uint32_t maxCount, VkBuffer keys, VkBuffer values, VkBuffer countBuffer,
                        uint32_t countOffset, VkBuffer keysHi) {
    VkDevice device = ctx.device;
    if (countOffset % 4 != 0) throw std::runtime_error("GpuRadixSort: count offset must be a multiple of 4");

    // The shaders use ballots, subgroup scans and the subgroup id/size built-ins.
    const VkSubgroupFeatureFlags needOps =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if ((ctx.subgroup.operations & needOps) != needOps || !(ctx.subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT))
        throw std::runtime_error("GpuRadixSort: needs basic/ballot/arithmetic subgroup operations in compute");
    // Shared arrays are sized for the smallest subgroup size the device may pick.
    const uint32_t minSubgroup = std::max(ctx.subgroup.minSize, 1u);
    if (minSubgroup < 4 || ctx.subgroup.maxSize > kWorkgroupSize)
        throw std::runtime_error("GpuRadixSort: unsupported subgroup size range");
    const uint32_t maxSubgroups = kWorkgroupSize / minSubgroup;

    m_maxCount = maxCount;
    m_countOffset = countOffset;
    m_keys64 = keysHi != VK_NULL_HANDLE;
    m_maxBlocks = std::max(1u, (maxCount + kBlockSize - 1) / kBlockSize);
    const VkDeviceSize n = std::max<VkDeviceSize>(maxCount, 1);

    // --- Buffers ---------------------------------------------------------------------
    m_keysB = createBuffer(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::GpuOnly, "radix sort: keys B");
    m_valuesB =
        createBuffer(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::GpuOnly, "radix sort: values B");
    if (m_keys64)
        m_keysHiB = createBuffer(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::GpuOnly,
                                 "radix sort: keys B (high words)");
    m_blockHist = createBuffer(ctx, VkDeviceSize(kRadixSize) * m_maxBlocks * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               MemoryUsage::GpuOnly, "radix sort: block histograms");
    m_digitCounts = createBuffer(ctx, kRadixSize * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::GpuOnly,
                                 "radix sort: digit counts");
    m_args = createBuffer(ctx, sizeof(SortArgs), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          MemoryUsage::GpuOnly, "radix sort: args");

    // --- Descriptors: one layout for all shaders (each uses a subset) ------------------
    // Bindings 8/9 (high words in/out) exist only in the 64-bit layout.
    const uint32_t bindings = m_keys64 ? 10 : 8;
    DescriptorLayoutBuilder builder;
    for (uint32_t b = 0; b < bindings; ++b) builder.add(b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setLayout = builder.build(device);
    m_pool.create(device, 2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * bindings}});
    const VkBuffer in[2] = {keys, m_keysB.buffer};
    const VkBuffer inValues[2] = {values, m_valuesB.buffer};
    const VkBuffer inHi[2] = {keysHi, m_keysHiB.buffer};
    for (uint32_t i = 0; i < 2; ++i) {
        m_sets[i] = m_pool.allocate(device, m_setLayout);
        DescriptorWriter writer;
        writer.buffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, in[i])
            .buffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inValues[i])
            .buffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, in[1 - i])
            .buffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inValues[1 - i])
            .buffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_blockHist.buffer)
            .buffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_digitCounts.buffer)
            .buffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_args.buffer)
            .buffer(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, countBuffer);
        if (m_keys64)
            writer.buffer(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inHi[i])
                .buffer(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inHi[1 - i]);
        writer.update(device, m_sets[i]);
    }

    // --- Pipelines -----------------------------------------------------------------------
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SortPushConstants)};
    m_layout = createPipelineLayout(device, {m_setLayout}, {push});
    const VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};  // constant_id 0 = kMaxSubgroups
    const VkSpecializationInfo spec{1, &entry, sizeof(uint32_t), &maxSubgroups};
    // Full subgroups: linearLane() must enumerate 0..255 exactly once.
    const VkPipelineShaderStageCreateFlags full = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    // The 64-bit builds of histogram/scatter (-DGSR_KEY64) also read/move the high words.
    const char* histogram = m_keys64 ? "radix_histogram_key64.comp.spv" : "radix_histogram.comp.spv";
    const char* scatter = m_keys64 ? "radix_scatter_key64.comp.spv" : "radix_scatter.comp.spv";
    const char* reorder = m_keys64 ? "radix_scatter_reorder_key64.comp.spv" : "radix_scatter_reorder.comp.spv";
    m_setup = createComputePipeline(device, m_layout, "radix_setup.comp.spv", &spec);
    m_histogram = createComputePipeline(device, m_layout, histogram, &spec, full);
    m_scan = createComputePipeline(device, m_layout, "radix_scan.comp.spv", &spec, full);
    m_scatter = createComputePipeline(device, m_layout, scatter, &spec, full);
    m_scatterReorder = createComputePipeline(device, m_layout, reorder, &spec, full);
}

VkDeviceSize GpuRadixSort::scratchBytes() const {
    return m_keysB.size + m_valuesB.size + m_keysHiB.size + m_blockHist.size + m_digitCounts.size + m_args.size;
}

void GpuRadixSort::record(VkCommandBuffer cmd, GpuProfiler* profiler, const RecordOptions& options) const {
    const std::string label = options.label ? options.label : "sort";
    const std::string setupName = label + ": setup", histogramName = label + ": histogram",
                      scanName = label + ": scan", scatterName = label + ": scatter";
    auto scope = [&](std::optional<GpuProfiler::Scope>& s, const std::string& name) {
        if (profiler) s.emplace(*profiler, cmd, name.c_str());
    };
    const uint32_t maxBits = m_keys64 ? 64u : 32u;
    if (options.keyBits == 0 || options.keyBits > maxBits)
        throw std::runtime_error("GpuRadixSort::record: keyBits out of range");
    const uint32_t passes = passCount(options.keyBits);
    SortPushConstants pc{0, m_countOffset / 4, m_maxCount};

    // Inputs (written by compute) visible; previous sort's scratch reads finished.
    computeToCompute(cmd);
    {
        std::optional<GpuProfiler::Scope> s;
        scope(s, setupName);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_setup);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_sets[0], 0, nullptr);
        vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
    }
    // The arguments are read twice: as indirect dispatch parameters (DRAW_INDIRECT stage,
    // INDIRECT_COMMAND_READ) and by the shaders (count, numBlocks).
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);

    for (uint32_t pass = 0; pass < passes; ++pass) {
        pc.shift = pass * kRadixBits;
        std::optional<GpuProfiler::Scope> passScope;
        if (options.perPassScopes)
            scope(passScope, label + ": pass " + std::to_string(pass) + " (bits " + std::to_string(pc.shift) + "-" +
                                 std::to_string(pc.shift + kRadixBits - 1) + ")");
        const VkDescriptorSet set = m_sets[pass % 2];  // even passes A -> B, odd B -> A
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        {
            std::optional<GpuProfiler::Scope> s;
            scope(s, histogramName);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_histogram);
            vkCmdDispatchIndirect(cmd, m_args.buffer, 0);
        }
        computeToCompute(cmd);  // blockHist written -> scanned
        {
            std::optional<GpuProfiler::Scope> s;
            scope(s, scanName);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_scan);
            vkCmdDispatch(cmd, kRadixSize, 1, 1);  // one workgroup per digit
        }
        computeToCompute(cmd);  // blockHist/digitCounts scanned -> read by the scatter
        {
            std::optional<GpuProfiler::Scope> s;
            scope(s, scatterName);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_variant == Variant::Classic ? m_scatter : m_scatterReorder);
            vkCmdDispatchIndirect(cmd, m_args.buffer, 0);
        }
        // Output -> next pass's input (RAW); next pass overwrites this pass's input and
        // blockHist (WAR). The caller's barrier follows the last pass.
        if (pass + 1 < passes) computeToCompute(cmd);
    }
}

void GpuRadixSort::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    for (VkPipeline* p : {&m_setup, &m_histogram, &m_scan, &m_scatter, &m_scatterReorder}) {
        if (*p) vkDestroyPipeline(device, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
    if (m_layout) vkDestroyPipelineLayout(device, m_layout, nullptr);
    m_layout = VK_NULL_HANDLE;
    m_pool.destroy(device);
    if (m_setLayout) vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
    m_setLayout = VK_NULL_HANDLE;
    for (Buffer* b : {&m_keysB, &m_valuesB, &m_keysHiB, &m_blockHist, &m_digitCounts, &m_args}) destroyBuffer(ctx, *b);
}

}  // namespace gsr
