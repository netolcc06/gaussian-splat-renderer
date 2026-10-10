#include "webgpu/wgpu_gradient.h"

#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_resources.h"
#include "webgpu/wgpu_shaders.h"

namespace gsr {

namespace {
constexpr uint32_t kGroupSize = 16;  // @workgroup_size(16, 16) in gradient.wgsl
}

bool WgpuGradientPass::create(WGPUDevice device, WgslShaderCache& shaders) {
    WGPUShaderModule module = shaders.get(device, {"gradient.wgsl", {}});
    if (!module) return false;
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = toStringView("gradient");
    desc.layout = nullptr;  // "auto": derived from the shader (one pipeline, so no sharing needed)
    desc.compute.module = module;
    desc.compute.entryPoint = toStringView("main");
    pipeline_ = wgpuDeviceCreateComputePipeline(device, &desc);
    layout_ = wgpuComputePipelineGetBindGroupLayout(pipeline_, 0);
    return pipeline_ != nullptr;
}

void WgpuGradientPass::encode(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& target) {
    if (boundGeneration_ != target.generation) {  // target (re)created: new bind group
        if (bindGroup_) wgpuBindGroupRelease(bindGroup_);
        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = 0;
        entry.textureView = target.view;
        WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg.label = toStringView("gradient");
        bg.layout = layout_;
        bg.entryCount = 1;
        bg.entries = &entry;
        bindGroup_ = wgpuDeviceCreateBindGroup(device, &bg);
        boundGeneration_ = target.generation;
    }

    WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    passDesc.label = toStringView("gradient");
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(pass, pipeline_);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup_, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, (target.width + kGroupSize - 1) / kGroupSize,
                                             (target.height + kGroupSize - 1) / kGroupSize, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

void WgpuGradientPass::release() {
    if (bindGroup_) wgpuBindGroupRelease(bindGroup_);
    if (layout_) wgpuBindGroupLayoutRelease(layout_);
    if (pipeline_) wgpuComputePipelineRelease(pipeline_);
    bindGroup_ = nullptr;
    layout_ = nullptr;
    pipeline_ = nullptr;
    boundGeneration_ = 0;
}

}  // namespace gsr
