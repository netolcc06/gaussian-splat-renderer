// WgpuScan (web phase W2): exclusive prefix sum of u32 values in WGSL compute, the web port of
// src/render/gpu_scan.* (shaders/wgsl/scan.wgsl).
//
//   output[j] = value(0) + ... + value(j-1),  total = value(0) + ... + value(count-1)
//
// The element count is read on the GPU (control[countOffset], clamped to maxCount) and the total is
// written on the GPU (control[totalOffset]): nothing is read back by the CPU. In the splat-tiles
// renderer both live in the control block, which is therefore ONE read_write binding (WebGPU does
// not allow the same buffer as read-only and writable storage in one dispatch).
//
// The pipelines (WgpuScanKernels) are shared by every scan of a renderer; the portable and the
// subgroup build give identical results.
#pragma once

#include "webgpu/wgpu_compute.h"

#include <webgpu/webgpu.h>

#include <cstdint>

namespace gsr {

class WgslShaderCache;

struct WgpuScanKernels {
    bool create(WGPUDevice device, WgslShaderCache& shaders, bool subgroups);
    void release();

    WGPUBindGroupLayout layout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUComputePipeline reduce = nullptr, blocks = nullptr, downsweep = nullptr;
    bool subgroups = false;
};

class WgpuScan {
public:
    enum Flags : uint32_t {
        kGather = 1,   // value(j) = input[index[j]]
        kNonZero = 2,  // value(j) = (value != 0) ? 1 : 0
    };

    struct Bindings {
        WGPUBuffer input = nullptr;   // u32 values (or the gathered array)
        WGPUBuffer index = nullptr;   // kGather only: indices into `input` (else: input is bound again)
        WGPUBuffer output = nullptr;  // >= maxCount u32: the exclusive scan
        WGPUBuffer control = nullptr; // holds the count and receives the total
        uint32_t countOffset = 0;     // in u32
        uint32_t totalOffset = 0;     // in u32
        uint32_t flags = 0;           // Flags
    };

    WgpuScan() = default;
    WgpuScan(const WgpuScan&) = delete;
    WgpuScan& operator=(const WgpuScan&) = delete;
    ~WgpuScan() { release(); }

    void create(WGPUDevice device, WGPUQueue queue, const WgpuScanKernels& kernels, uint32_t maxCount,
                const Bindings& bindings, uint32_t maxGroupsX);
    void release();

    // The three dispatches, into an open compute pass (each dispatch sees the previous one's writes).
    void encode(WGPUComputePassEncoder pass) const;

    uint32_t maxCount() const { return maxCount_; }
    uint64_t scratchBytes() const { return blockSums_.size + params_.size; }

private:
    const WgpuScanKernels* kernels_ = nullptr;
    uint32_t maxCount_ = 0, maxBlocks_ = 0, maxGroupsX_ = 65535;
    WgpuBuffer blockSums_, params_;
    WGPUBindGroup group_ = nullptr;
};

}  // namespace gsr
