// Points renderer (web): port of src/render/point_cloud_renderer.* (shaders/wgsl/points.wgsl).
// Every gaussian is a small opaque screen-space disc coloured by its degree-0 SH colour, with a
// depth test and no blending; it renders into the rgba16float target (cleared to the native
// background colour), which the resolve pass then copies to the canvas. The debug view that
// checks scene upload, decoding, scene transform and camera before the splat renderer exists.
#pragma once

#include "webgpu/wgpu_resources.h"

#include <glm/glm.hpp>
#include <webgpu/webgpu.h>

#include <map>
#include <string>

namespace gsr {

class WgslShaderCache;
class WgpuScene;

class WgpuPointsRenderer {
public:
    WgpuPointsRenderer() = default;
    WgpuPointsRenderer(const WgpuPointsRenderer&) = delete;
    WgpuPointsRenderer& operator=(const WgpuPointsRenderer&) = delete;
    ~WgpuPointsRenderer() { release(); }

    bool create(WGPUDevice device, WgslShaderCache& shaders, WGPUBindGroupLayout sceneLayout);
    void release();

    // Draws `scene` into `target` (rgba16float, RenderAttachment usage). mvp = proj * view * model
    // with the native (Vulkan) projection; the shader converts the clip space.
    void encode(WGPUDevice device, WGPUQueue queue, WGPUCommandEncoder encoder, const WgpuScene& scene,
                const WgpuTexture2D& target, const glm::mat4& mvp);

    float opacityThreshold = 0.05f;  // native defaults (PointCloudRenderer)
    float pointSizePx = 2.0f;

private:
    WGPURenderPipeline pipelineFor(WGPUDevice device, const WgpuScene& scene);

    WGPUShaderModule module_ = nullptr;  // owned by the shader cache
    WGPUBindGroupLayout sceneLayout_ = nullptr;  // owned by the caller
    WGPUBindGroupLayout paramsLayout_ = nullptr;
    WGPUPipelineLayout pipelineLayout_ = nullptr;
    WGPUBuffer params_ = nullptr;
    WGPUBindGroup paramsGroup_ = nullptr;
    std::map<std::string, WGPURenderPipeline> pipelines_;  // per scene decoder variant
    WgpuTexture2D depth_;
};

}  // namespace gsr
