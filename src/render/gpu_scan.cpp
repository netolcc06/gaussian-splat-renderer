#include "render/gpu_scan.h"

#include "core/context.h"
#include "core/descriptors.h"
#include "core/pipelines.h"
#include "core/profiler.h"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace gsr {

namespace {

// Must match ScanPushConstants in shaders/scan_common.glsl.
struct ScanPushConstants {
    uint32_t countOffset;  // in uints
    uint32_t maxCount;
    uint32_t totalOffset;  // in uints
    uint32_t flags;
};

constexpr uint32_t kWorkgroupSize = 256;  // SCAN_WG_SIZE

void computeToCompute(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

}  // namespace

void GpuScan::init(VulkanContext& ctx, uint32_t maxCount, const Bindings& bindings) {
    VkDevice device = ctx.device;
    if (bindings.countOffset % 4 != 0 || bindings.totalOffset % 4 != 0)
        throw std::runtime_error("GpuScan: count/total offsets must be multiples of 4");
    if ((bindings.flags & kGather) && !bindings.index) throw std::runtime_error("GpuScan: gather needs an index buffer");
    const VkSubgroupFeatureFlags needOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if ((ctx.subgroup.operations & needOps) != needOps || !(ctx.subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT))
        throw std::runtime_error("GpuScan: needs basic/arithmetic subgroup operations in compute");
    const uint32_t minSubgroup = std::max(ctx.subgroup.minSize, 1u);
    if (minSubgroup < 4 || ctx.subgroup.maxSize > kWorkgroupSize)
        throw std::runtime_error("GpuScan: unsupported subgroup size range");
    const uint32_t maxSubgroups = kWorkgroupSize / minSubgroup;

    m_maxCount = maxCount;
    m_maxBlocks = std::max(1u, (maxCount + kBlockSize - 1) / kBlockSize);
    m_bindings = bindings;
    m_blockSums = createBuffer(ctx, VkDeviceSize(m_maxBlocks) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               MemoryUsage::GpuOnly, "scan: block sums");

    DescriptorLayoutBuilder builder;
    for (uint32_t b = 0; b <= 5; ++b) builder.add(b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    m_setLayout = builder.build(device);
    m_pool.create(device, 1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}});
    m_set = m_pool.allocate(device, m_setLayout);
    DescriptorWriter()
        .buffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings.input)
        // Without kGather the index buffer is never read, but the binding is statically
        // used by the shader, so it must hold a valid buffer: bind the input again.
        .buffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings.index ? bindings.index : bindings.input)
        .buffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings.output)
        .buffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_blockSums.buffer)
        .buffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings.countBuffer)
        .buffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings.totalBuffer)
        .update(device, m_set);

    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScanPushConstants)};
    m_layout = createPipelineLayout(device, {m_setLayout}, {push});
    const VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};  // constant_id 0 = kMaxSubgroups
    const VkSpecializationInfo spec{1, &entry, sizeof(uint32_t), &maxSubgroups};
    const VkPipelineShaderStageCreateFlags full = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    m_reduce = createComputePipeline(device, m_layout, "scan_reduce.comp.spv", &spec, full);
    m_blocks = createComputePipeline(device, m_layout, "scan_blocks.comp.spv", &spec, full);
    m_downsweep = createComputePipeline(device, m_layout, "scan_downsweep.comp.spv", &spec, full);
}

void GpuScan::record(VkCommandBuffer cmd, GpuProfiler* profiler, const char* scopeName) const {
    std::optional<GpuProfiler::Scope> scope;
    if (profiler && scopeName) scope.emplace(*profiler, cmd, scopeName);
    const ScanPushConstants pc{m_bindings.countOffset / 4, m_maxCount, m_bindings.totalOffset / 4, m_bindings.flags};

    computeToCompute(cmd);  // inputs + count visible; previous use of blockSums finished
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_set, 0, nullptr);
    vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // Enough workgroups for maxCount; those past the GPU-side count return at once.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_reduce);
    vkCmdDispatch(cmd, m_maxBlocks, 1, 1);
    computeToCompute(cmd);  // blockSums written -> scanned
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blocks);
    vkCmdDispatch(cmd, 1, 1, 1);
    computeToCompute(cmd);  // blockSums scanned -> added by the downsweep
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_downsweep);
    vkCmdDispatch(cmd, m_maxBlocks, 1, 1);
}

void GpuScan::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    for (VkPipeline* p : {&m_reduce, &m_blocks, &m_downsweep}) {
        if (*p) vkDestroyPipeline(device, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
    if (m_layout) vkDestroyPipelineLayout(device, m_layout, nullptr);
    m_pool.destroy(device);
    if (m_setLayout) vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
    m_layout = VK_NULL_HANDLE;
    m_setLayout = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    destroyBuffer(ctx, m_blockSums);
}

}  // namespace gsr
