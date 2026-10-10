// GpuScan (Phase 4): exclusive prefix sum of uint32 values in GLSL compute
// (shaders/scan_*.comp, algorithm in shaders/scan_common.glsl).
//
//   output[j] = value(0) + ... + value(j-1),  total = value(0) + ... + value(count-1)
//
// Uses in the tile rasterizer (splat_tile_renderer.cpp):
//   - "tiles touched" per splat -> the offset of each splat's first (tile, splat) entry
//     in the duplicated list, and the total = number of entries (INRIA: cub
//     InclusiveSum over tiles_touched);
//   - visibility flags -> the slot of each visible splat in a compact list, in INDEX
//     order (an order-preserving stream compaction, unlike Phase 3's atomic append).
//
// Like GpuRadixSort, the element count is read on the GPU (countBuffer + offset), and the
// total is written on the GPU (totalBuffer + offset): nothing is read back by the CPU.
#pragma once

#include "core/descriptors.h"
#include "core/resources.h"

namespace gsr {

class VulkanContext;
class GpuProfiler;

class GpuScan {
public:
    static constexpr uint32_t kBlockSize = 2048;  // SCAN_BLOCK_SIZE
    enum Flags : uint32_t {
        kGather = 1,   // value(j) = input[index[j]]
        kNonZero = 2,  // value(j) = (value != 0) ? 1 : 0
    };

    struct Bindings {
        VkBuffer input = VK_NULL_HANDLE;   // uint values (or the gathered array)
        VkBuffer index = VK_NULL_HANDLE;   // kGather only: indices into `input`
        VkBuffer output = VK_NULL_HANDLE;  // >= maxCount uints: the exclusive scan
        VkBuffer countBuffer = VK_NULL_HANDLE;
        uint32_t countOffset = 0;  // bytes, multiple of 4
        VkBuffer totalBuffer = VK_NULL_HANDLE;
        uint32_t totalOffset = 0;  // bytes, multiple of 4
        uint32_t flags = 0;        // Flags
    };

    void init(VulkanContext& ctx, uint32_t maxCount, const Bindings& bindings);
    void destroy(VulkanContext& ctx);

    // Synchronization contract (same as GpuRadixSort): begins with a COMPUTE -> COMPUTE
    // barrier (inputs, count written by compute shaders; previous scan's scratch reads
    // done), has barriers between its three dispatches, and its outputs (output[], the
    // total) were last written by COMPUTE_SHADER when it returns.
    // `scopeName` (optional, with a profiler): one GPU scope around the three dispatches.
    void record(VkCommandBuffer cmd, GpuProfiler* profiler = nullptr, const char* scopeName = nullptr) const;

    uint32_t maxCount() const { return m_maxCount; }

private:
    uint32_t m_maxCount = 0;
    uint32_t m_maxBlocks = 0;
    Bindings m_bindings;
    Buffer m_blockSums;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    DescriptorPool m_pool;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_reduce = VK_NULL_HANDLE;
    VkPipeline m_blocks = VK_NULL_HANDLE;
    VkPipeline m_downsweep = VK_NULL_HANDLE;

public:
    // Phase 6 `--pipeline-stats`.
    std::vector<std::pair<std::string, VkPipeline>> pipelines(const std::string& label) const {
        return {{label + ": scan reduce", m_reduce}, {label + ": scan blocks", m_blocks},
                {label + ": scan downsweep", m_downsweep}};
    }
};

}  // namespace gsr
