// WgpuRadixSort (web phase W2): a stable LSD radix sort of 32-bit (key, value) pairs in WGSL, the web
// port of src/render/gpu_radix_sort.* (shaders/wgsl/radix_sort.wgsl).
//
//   encode():  setup (1 thread: count -> indirect dispatch arguments, 2D-clamped)
//              P x [ histogram -> scan -> scatter ]   (4-bit digits, shift 0, 4, 8, ...)
//
// P = ceil(keyBits / 4) rounded up to EVEN, so the ping-pong A -> B -> A ... ends in the caller's
// buffers (an extra pass on an all-zero digit is the identity: stable). 32-bit depth keys: 8 passes;
// 13-bit tile keys (1080p): 4 passes. Ties keep their input order, so the result equals
// DepthSorter::radixSortPairs bit for bit (in-page self-test "sort").
//
// Why 4-bit digits and 1024-key blocks instead of native's 8 bits / 2048 keys: the portable local
// rank must fit WebGPU's default 16 KB of workgroup memory without subgroups (radix_sort.wgsl).
#pragma once

#include "webgpu/wgpu_compute.h"

#include <webgpu/webgpu.h>

#include <cstdint>

namespace gsr {

class WgslShaderCache;

struct WgpuSortKernels {
    bool create(WGPUDevice device, WgslShaderCache& shaders, bool subgroups);
    void release();

    WGPUBindGroupLayout passLayout = nullptr, setupLayout = nullptr;
    WGPUPipelineLayout passPipelineLayout = nullptr, setupPipelineLayout = nullptr;
    WGPUComputePipeline setup = nullptr, histogram = nullptr, scan = nullptr, scatter = nullptr;
    bool subgroups = false;
};

class WgpuRadixSort {
public:
    static constexpr uint32_t kMaxPasses = 8;  // 32-bit keys
    static uint32_t passCount(uint32_t keyBits);

    WgpuRadixSort() = default;
    WgpuRadixSort(const WgpuRadixSort&) = delete;
    WgpuRadixSort& operator=(const WgpuRadixSort&) = delete;
    ~WgpuRadixSort() { release(); }

    // keys/values: >= maxCount u32 each (Storage usage), sorted in place. countBuffer[countOffset]
    // (u32 index) holds the number of keys; it is clamped to maxCount on the GPU.
    void create(WGPUDevice device, WGPUQueue queue, const WgpuSortKernels& kernels, uint32_t maxCount, WGPUBuffer keys,
                WGPUBuffer values, WGPUBuffer countBuffer, uint32_t countOffset, uint32_t maxGroupsX);
    void release();

    // Setup + the passes for `keyBits` significant low bits, into an open compute pass.
    void encode(WGPUComputePassEncoder pass, uint32_t keyBits) const;

    uint64_t scratchBytes() const { return keysB_.size + valuesB_.size + blockHist_.size + digitCounts_.size; }

private:
    const WgpuSortKernels* kernels_ = nullptr;
    uint32_t maxCount_ = 0;
    WgpuBuffer keysB_, valuesB_, blockHist_, digitCounts_, args_, params_;
    WGPUBindGroup setupGroup_ = nullptr, aToB_ = nullptr, bToA_ = nullptr;
};

}  // namespace gsr
