// W0 test renderer: one compute pass that writes an exact integer gradient
// (shaders/wgsl/gradient.wgsl) into the rgba16float render target. It proves the compute path
// (pipeline, bind group, storage texture) before the splat renderers exist, like the native
// Phase 0 gradient (shaders/gradient.comp).
#pragma once

#include <webgpu/webgpu.h>

namespace gsr {

class WgslShaderCache;
struct WgpuTexture2D;

class WgpuGradientPass {
public:
    WgpuGradientPass() = default;
    WgpuGradientPass(const WgpuGradientPass&) = delete;
    WgpuGradientPass& operator=(const WgpuGradientPass&) = delete;
    ~WgpuGradientPass() { release(); }

    bool create(WGPUDevice device, WgslShaderCache& shaders);
    // Records the compute pass writing the whole of `target` (rgba16float, STORAGE_BINDING).
    void encode(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& target);
    void release();

private:
    WGPUComputePipeline pipeline_ = nullptr;
    WGPUBindGroupLayout layout_ = nullptr;
    WGPUBindGroup bindGroup_ = nullptr;
    uint64_t boundGeneration_ = 0;  // WgpuTexture2D::generation bindGroup_ was made for
};

}  // namespace gsr
