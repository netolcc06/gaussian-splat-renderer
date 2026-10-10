#include "webgpu/wgpu_scan.h"

#include "webgpu/shader_constants.h"
#include "webgpu/wgpu_shaders.h"

#include <algorithm>

namespace gsr {

namespace {
// Must match ScanParams in shaders/wgsl/scan.wgsl.
struct ScanParams {
    uint32_t countOffset, maxCount, totalOffset, flags;
};
}  // namespace

bool WgpuScanKernels::create(WGPUDevice device, WgslShaderCache& shaders, bool useSubgroups) {
    release();
    subgroups = useSubgroups;
    WgslVariant v{"scan.wgsl", {}};
    if (useSubgroups) v.defines.push_back("GSR_SUBGROUPS");
    WGPUShaderModule module = shaders.get(device, v);
    if (!module) return false;
    layout = makeComputeLayout(device,
                               {{0, BindKind::Uniform},
                                {1, BindKind::ReadOnlyStorage},
                                {2, BindKind::ReadOnlyStorage},
                                {3, BindKind::Storage},
                                {4, BindKind::Storage},
                                {5, BindKind::Storage}},
                               "scan");
    pipelineLayout = makePipelineLayout(device, {layout}, "scan");
    reduce = makeComputePipeline(device, pipelineLayout, module, "scan_reduce", {}, "scan reduce");
    blocks = makeComputePipeline(device, pipelineLayout, module, "scan_blocks", {}, "scan blocks");
    downsweep = makeComputePipeline(device, pipelineLayout, module, "scan_downsweep", {}, "scan downsweep");
    return reduce && blocks && downsweep;
}

void WgpuScanKernels::release() {
    for (WGPUComputePipeline* p : {&reduce, &blocks, &downsweep}) {
        if (*p) wgpuComputePipelineRelease(*p);
        *p = nullptr;
    }
    if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
    if (layout) wgpuBindGroupLayoutRelease(layout);
    pipelineLayout = nullptr;
    layout = nullptr;
}

void WgpuScan::create(WGPUDevice device, WGPUQueue queue, const WgpuScanKernels& kernels, uint32_t maxCount,
                      const Bindings& b, uint32_t maxGroupsX) {
    release();
    kernels_ = &kernels;
    maxCount_ = maxCount;
    maxGroupsX_ = maxGroupsX;
    maxBlocks_ = std::max(1u, (maxCount + web::kScanBlockSize - 1) / web::kScanBlockSize);
    blockSums_.create(device, uint64_t(maxBlocks_) * 4, WGPUBufferUsage_Storage, "scan block sums");
    params_.create(device, sizeof(ScanParams), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "scan params");
    const ScanParams p{b.countOffset, maxCount, b.totalOffset, b.flags};
    wgpuQueueWriteBuffer(queue, params_.buffer, 0, &p, sizeof(p));
    group_ = makeBindGroup(device, kernels.layout,
                           {bindBuffer(0, params_.buffer), bindBuffer(1, b.input),
                            bindBuffer(2, b.index ? b.index : b.input), bindBuffer(3, b.output),
                            bindBuffer(4, blockSums_.buffer), bindBuffer(5, b.control)},
                           "scan");
}

void WgpuScan::release() {
    if (group_) wgpuBindGroupRelease(group_);
    group_ = nullptr;
    blockSums_.release();
    params_.release();
}

void WgpuScan::encode(WGPUComputePassEncoder pass) const {
    wgpuComputePassEncoderSetBindGroup(pass, 0, group_, 0, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, kernels_->reduce);
    dispatch1D(pass, maxBlocks_, maxGroupsX_);  // blocks past the GPU-side count exit at once
    wgpuComputePassEncoderSetPipeline(pass, kernels_->blocks);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, kernels_->downsweep);
    dispatch1D(pass, maxBlocks_, maxGroupsX_);
}

}  // namespace gsr
