#include "webgpu/wgpu_radix_sort.h"

#include "webgpu/shader_constants.h"
#include "webgpu/wgpu_shaders.h"

#include <algorithm>

namespace gsr {

namespace {

// Must match SortPassParams in shaders/wgsl/radix_sort.wgsl; one per 256-byte uniform slot
// (minUniformBufferOffsetAlignment), selected per pass with a dynamic offset.
struct SortPassParams {
    uint32_t shift, countOffset, maxCount, maxGroupsX;
};
constexpr uint64_t kSlot = 256;
constexpr uint64_t kArgsBytes = 32;  // SortArgs

}  // namespace

bool WgpuSortKernels::create(WGPUDevice device, WgslShaderCache& shaders, bool useSubgroups) {
    release();
    subgroups = useSubgroups;
    WgslVariant v{"radix_sort.wgsl", {}};
    if (useSubgroups) v.defines.push_back("GSR_SUBGROUPS");
    WGPUShaderModule module = shaders.get(device, v);
    if (!module) return false;
    passLayout = makeComputeLayout(device,
                                   {{0, BindKind::UniformDynamic},
                                    {1, BindKind::ReadOnlyStorage},
                                    {2, BindKind::ReadOnlyStorage},
                                    {3, BindKind::Storage},
                                    {4, BindKind::Storage},
                                    {5, BindKind::Storage},
                                    {6, BindKind::Storage},
                                    {7, BindKind::ReadOnlyStorage}},
                                   "radix pass");
    setupLayout = makeComputeLayout(
        device, {{0, BindKind::UniformDynamic}, {8, BindKind::ReadOnlyStorage}, {9, BindKind::Storage}}, "radix setup");
    passPipelineLayout = makePipelineLayout(device, {passLayout}, "radix pass");
    setupPipelineLayout = makePipelineLayout(device, {setupLayout}, "radix setup");
    setup = makeComputePipeline(device, setupPipelineLayout, module, "radix_setup", {}, "radix setup");
    histogram = makeComputePipeline(device, passPipelineLayout, module, "radix_histogram", {}, "radix histogram");
    scan = makeComputePipeline(device, passPipelineLayout, module, "radix_scan", {}, "radix scan");
    scatter = makeComputePipeline(device, passPipelineLayout, module, "radix_scatter", {}, "radix scatter");
    return setup && histogram && scan && scatter;
}

void WgpuSortKernels::release() {
    for (WGPUComputePipeline* p : {&setup, &histogram, &scan, &scatter}) {
        if (*p) wgpuComputePipelineRelease(*p);
        *p = nullptr;
    }
    for (WGPUPipelineLayout* p : {&passPipelineLayout, &setupPipelineLayout}) {
        if (*p) wgpuPipelineLayoutRelease(*p);
        *p = nullptr;
    }
    for (WGPUBindGroupLayout* p : {&passLayout, &setupLayout}) {
        if (*p) wgpuBindGroupLayoutRelease(*p);
        *p = nullptr;
    }
}

uint32_t WgpuRadixSort::passCount(uint32_t keyBits) {
    uint32_t p = (std::clamp(keyBits, 1u, 32u) + web::kRadixBits - 1) / web::kRadixBits;
    p += p & 1u;  // even: the data ends in the caller's buffers
    return std::max(p, 2u);
}

void WgpuRadixSort::create(WGPUDevice device, WGPUQueue queue, const WgpuSortKernels& kernels, uint32_t maxCount,
                           WGPUBuffer keys, WGPUBuffer values, WGPUBuffer countBuffer, uint32_t countOffset,
                           uint32_t maxGroupsX) {
    release();
    kernels_ = &kernels;
    maxCount_ = maxCount;
    const uint64_t n = std::max(maxCount, 1u);
    const uint64_t maxBlocks = (n + web::kSortBlockSize - 1) / web::kSortBlockSize;
    keysB_.create(device, n * 4, WGPUBufferUsage_Storage, "sort keys B");
    valuesB_.create(device, n * 4, WGPUBufferUsage_Storage, "sort values B");
    blockHist_.create(device, maxBlocks * web::kRadixSize * 4, WGPUBufferUsage_Storage, "sort block histograms");
    digitCounts_.create(device, web::kRadixSize * 4, WGPUBufferUsage_Storage, "sort digit counts");
    args_.create(device, kArgsBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, "sort args");
    params_.create(device, kSlot * kMaxPasses, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "sort params");
    std::vector<uint8_t> slots(kSlot * kMaxPasses, 0);
    for (uint32_t p = 0; p < kMaxPasses; ++p) {
        const SortPassParams sp{p * web::kRadixBits, countOffset, maxCount, std::max(maxGroupsX, 1u)};
        std::memcpy(slots.data() + p * kSlot, &sp, sizeof(sp));
    }
    wgpuQueueWriteBuffer(queue, params_.buffer, 0, slots.data(), slots.size());

    const uint64_t paramSize = sizeof(SortPassParams);
    setupGroup_ = makeBindGroup(device, kernels.setupLayout,
                                {bindBuffer(0, params_.buffer, paramSize), bindBuffer(8, countBuffer),
                                 bindBuffer(9, args_.buffer)},
                                "radix setup");
    auto passGroup = [&](WGPUBuffer kIn, WGPUBuffer vIn, WGPUBuffer kOut, WGPUBuffer vOut, const char* label) {
        return makeBindGroup(device, kernels.passLayout,
                             {bindBuffer(0, params_.buffer, paramSize), bindBuffer(1, kIn), bindBuffer(2, vIn),
                              bindBuffer(3, kOut), bindBuffer(4, vOut), bindBuffer(5, blockHist_.buffer),
                              bindBuffer(6, digitCounts_.buffer), bindBuffer(7, args_.buffer)},
                             label);
    };
    aToB_ = passGroup(keys, values, keysB_.buffer, valuesB_.buffer, "radix A->B");
    bToA_ = passGroup(keysB_.buffer, valuesB_.buffer, keys, values, "radix B->A");
}

void WgpuRadixSort::release() {
    for (WGPUBindGroup* g : {&setupGroup_, &aToB_, &bToA_}) {
        if (*g) wgpuBindGroupRelease(*g);
        *g = nullptr;
    }
    for (WgpuBuffer* b : {&keysB_, &valuesB_, &blockHist_, &digitCounts_, &args_, &params_}) b->release();
}

void WgpuRadixSort::encode(WGPUComputePassEncoder pass, uint32_t keyBits) const {
    const uint32_t zero = 0;
    wgpuComputePassEncoderSetPipeline(pass, kernels_->setup);
    wgpuComputePassEncoderSetBindGroup(pass, 0, setupGroup_, 1, &zero);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    const uint32_t passes = passCount(keyBits);
    for (uint32_t p = 0; p < passes; ++p) {
        const uint32_t offset = uint32_t(p * kSlot);
        wgpuComputePassEncoderSetBindGroup(pass, 0, (p & 1u) ? bToA_ : aToB_, 1, &offset);
        wgpuComputePassEncoderSetPipeline(pass, kernels_->histogram);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args_.buffer, 0);
        wgpuComputePassEncoderSetPipeline(pass, kernels_->scan);
        wgpuComputePassEncoderDispatchWorkgroups(pass, web::kRadixSize, 1, 1);  // one per digit
        wgpuComputePassEncoderSetPipeline(pass, kernels_->scatter);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args_.buffer, 0);
    }
}

}  // namespace gsr
