#include "webgpu/wgpu_points.h"

#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_scene.h"
#include "webgpu/wgpu_shaders.h"

#include <vector>

namespace gsr {

namespace {

// Must match PointParams in shaders/wgsl/points.wgsl (the native push constants of points.vert,
// minus the unused shCoeffs).
struct PointParams {
    glm::mat4 mvp;
    glm::vec2 viewportSize;
    float pointSizePx;
    float opacityThreshold;
};
static_assert(sizeof(PointParams) == 80, "PointParams must match the WGSL uniform layout");

}  // namespace

bool WgpuPointsRenderer::create(WGPUDevice device, WgslShaderCache& shaders, WGPUBindGroupLayout sceneLayout) {
    module_ = shaders.get(device, {"points.wgsl", {}});
    if (!module_) return false;
    sceneLayout_ = sceneLayout;

    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Vertex;
    entry.buffer.type = WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = sizeof(PointParams);
    WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bglDesc.label = toStringView("points params");
    bglDesc.entryCount = 1;
    bglDesc.entries = &entry;
    paramsLayout_ = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    // Group 0: the scene (shared layout), group 1: this renderer's parameters.
    const WGPUBindGroupLayout layouts[2] = {sceneLayout_, paramsLayout_};
    WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    plDesc.label = toStringView("points");
    plDesc.bindGroupLayoutCount = 2;
    plDesc.bindGroupLayouts = layouts;
    pipelineLayout_ = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUBufferDescriptor bufDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bufDesc.label = toStringView("points params");
    bufDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    bufDesc.size = sizeof(PointParams);
    params_ = wgpuDeviceCreateBuffer(device, &bufDesc);

    WGPUBindGroupEntry bgEntry = WGPU_BIND_GROUP_ENTRY_INIT;
    bgEntry.binding = 0;
    bgEntry.buffer = params_;
    bgEntry.size = sizeof(PointParams);
    WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bgDesc.label = toStringView("points params");
    bgDesc.layout = paramsLayout_;
    bgDesc.entryCount = 1;
    bgDesc.entries = &bgEntry;
    paramsGroup_ = wgpuDeviceCreateBindGroup(device, &bgDesc);
    return paramsLayout_ && pipelineLayout_ && params_ && paramsGroup_;
}

WGPURenderPipeline WgpuPointsRenderer::pipelineFor(WGPUDevice device, const WgpuScene& scene) {
    const std::string key = scene.decoderKey();
    if (auto it = pipelines_.find(key); it != pipelines_.end()) return it->second;

    // The scene format is fixed per pipeline (override constants of scene_decode.wgsl), like the
    // native specialization constants.
    const std::vector<WGPUConstantEntry> constants = scene.decodeConstants();
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = WGPUTextureFormat_RGBA16Float;
    target.writeMask = WGPUColorWriteMask_All;  // opaque, no blending
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module_;
    fragment.entryPoint = toStringView("fs_points");
    fragment.targetCount = 1;
    fragment.targets = &target;

    WGPUDepthStencilState depth = WGPU_DEPTH_STENCIL_STATE_INIT;
    depth.format = WGPUTextureFormat_Depth32Float;
    depth.depthWriteEnabled = WGPUOptionalBool_True;
    depth.depthCompare = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = toStringView("points");
    desc.layout = pipelineLayout_;
    desc.vertex.module = module_;
    desc.vertex.entryPoint = toStringView("vs_points");
    desc.vertex.constantCount = constants.size();
    desc.vertex.constants = constants.data();
    // No vertex buffers: the shader fetches from the scene buffers by instance index.
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.depthStencil = &depth;
    desc.fragment = &fragment;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    pipelines_[key] = pipeline;
    return pipeline;
}

void WgpuPointsRenderer::encode(WGPUDevice device, WGPUQueue queue, WGPUCommandEncoder encoder,
                                const WgpuScene& scene, const WgpuTexture2D& target, const glm::mat4& mvp) {
    if (depth_.width != target.width || depth_.height != target.height) {
        depth_.create(device, target.width, target.height, WGPUTextureFormat_Depth32Float,
                      WGPUTextureUsage_RenderAttachment, "points depth");
    }
    PointParams params{};
    params.mvp = mvp;
    params.viewportSize = glm::vec2(float(target.width), float(target.height));
    params.pointSizePx = pointSizePx;
    params.opacityThreshold = opacityThreshold;
    wgpuQueueWriteBuffer(queue, params_, 0, &params, sizeof(params));  // lands before this frame's submit

    WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color.view = target.view;
    color.loadOp = WGPULoadOp_Clear;
    color.storeOp = WGPUStoreOp_Store;
    color.clearValue = {0.06, 0.06, 0.08, 1.0};  // the native clear colour
    WGPURenderPassDepthStencilAttachment depthAttachment = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    depthAttachment.view = depth_.view;
    depthAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthAttachment.depthStoreOp = WGPUStoreOp_Discard;
    depthAttachment.depthClearValue = 1.0f;

    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = toStringView("points");
    desc.colorAttachmentCount = 1;
    desc.colorAttachments = &color;
    desc.depthStencilAttachment = &depthAttachment;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    wgpuRenderPassEncoderSetPipeline(pass, pipelineFor(device, scene));
    wgpuRenderPassEncoderSetBindGroup(pass, 0, scene.bindGroup(), 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(pass, 1, paramsGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 6, scene.count, 0, 0);  // 6 vertices (one quad) per gaussian
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void WgpuPointsRenderer::release() {
    for (auto& [key, p] : pipelines_)
        if (p) wgpuRenderPipelineRelease(p);
    pipelines_.clear();
    depth_.release();
    if (paramsGroup_) wgpuBindGroupRelease(paramsGroup_);
    if (params_) {
        wgpuBufferDestroy(params_);
        wgpuBufferRelease(params_);
    }
    if (pipelineLayout_) wgpuPipelineLayoutRelease(pipelineLayout_);
    if (paramsLayout_) wgpuBindGroupLayoutRelease(paramsLayout_);
    paramsGroup_ = nullptr;
    params_ = nullptr;
    pipelineLayout_ = nullptr;
    paramsLayout_ = nullptr;
    module_ = nullptr;
}

}  // namespace gsr
