#include "webgpu/wgpu_resolve.h"

#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_shaders.h"

#include <cstdio>
#include <cstring>

namespace gsr {

namespace {

WGPURenderPipeline makePipeline(WGPUDevice device, WGPUPipelineLayout layout, WGPUShaderModule module,
                                WGPUTextureFormat format, const char* label) {
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = toStringView("fs_resolve");
    fragment.targetCount = 1;
    fragment.targets = &target;

    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.layout = layout;
    desc.vertex.module = module;
    desc.vertex.entryPoint = toStringView("vs_fullscreen");
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.fragment = &fragment;
    return wgpuDeviceCreateRenderPipeline(device, &desc);
}

}  // namespace

bool WgpuResolve::create(WGPUDevice device, WgslShaderCache& shaders, WGPUTextureFormat canvasFormat) {
    WGPUShaderModule module = shaders.get(device, {"resolve.wgsl", {}});
    if (!module) return false;

    // Explicit layout: an "auto" layout belongs to a single pipeline, and both pipelines must
    // accept the same bind group. textureLoad needs no sampler, so the texture is declared
    // unfilterable (allowed for rgba16float).
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Fragment;
    entry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    entry.texture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bglDesc.label = toStringView("resolve");
    bglDesc.entryCount = 1;
    bglDesc.entries = &entry;
    layout_ = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    plDesc.label = toStringView("resolve");
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &layout_;
    pipelineLayout_ = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    canvasPipeline_ = makePipeline(device, pipelineLayout_, module, canvasFormat, "resolve to canvas");
    capturePipeline_ = makePipeline(device, pipelineLayout_, module, WGPUTextureFormat_RGBA8Unorm, "resolve to capture");
    return canvasPipeline_ && capturePipeline_;
}

void WgpuResolve::bindSource(WGPUDevice device, const WgpuTexture2D& source) {
    if (boundGeneration_ == source.generation) return;
    if (bindGroup_) wgpuBindGroupRelease(bindGroup_);
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.textureView = source.view;
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.label = toStringView("resolve");
    desc.layout = layout_;
    desc.entryCount = 1;
    desc.entries = &entry;
    bindGroup_ = wgpuDeviceCreateBindGroup(device, &desc);
    boundGeneration_ = source.generation;
}

void WgpuResolve::draw(WGPUCommandEncoder encoder, WGPURenderPipeline pipeline, WGPUTextureView target,
                       const char* label) {
    WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color.view = target;
    color.loadOp = WGPULoadOp_Clear;  // every pixel is overwritten; Clear avoids a load
    color.storeOp = WGPUStoreOp_Store;
    color.clearValue = {0.0, 0.0, 0.0, 1.0};
    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.colorAttachmentCount = 1;
    desc.colorAttachments = &color;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void WgpuResolve::encodeToCanvas(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& source,
                                 WGPUTextureView canvasView) {
    bindSource(device, source);
    draw(encoder, canvasPipeline_, canvasView, "resolve to canvas");
}

bool WgpuResolve::requestCapture() {
    if (status_ == CaptureStatus::Requested || status_ == CaptureStatus::Mapping) return false;
    status_ = CaptureStatus::Requested;
    return true;
}

bool WgpuResolve::encodeCapture(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& source) {
    if (status_ != CaptureStatus::Requested) return false;
    const uint32_t w = source.width, h = source.height;
    if (captureTexture_.width != w || captureTexture_.height != h) {
        captureTexture_.create(device, w, h, WGPUTextureFormat_RGBA8Unorm,
                               WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc, "capture");
    }
    readbackRowPitch_ = alignedBytesPerRow(w * 4);
    const uint64_t size = uint64_t(readbackRowPitch_) * h;
    if (!readback_ || readbackSize_ != size) {
        if (readback_) {
            wgpuBufferDestroy(readback_);
            wgpuBufferRelease(readback_);
        }
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.label = toStringView("capture readback");
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        desc.size = size;
        readback_ = wgpuDeviceCreateBuffer(device, &desc);
        readbackSize_ = size;
    }

    bindSource(device, source);
    draw(encoder, capturePipeline_, captureTexture_.view, "resolve to capture");

    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = captureTexture_.texture;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = readback_;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = readbackRowPitch_;
    dst.layout.rowsPerImage = h;
    const WGPUExtent3D extent = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
    captureWidth_ = w;
    captureHeight_ = h;
    status_ = CaptureStatus::Mapping;
    return true;
}

void WgpuResolve::startMapping() {
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = &WgpuResolve::onMapped;
    cb.userdata1 = this;
    wgpuBufferMapAsync(readback_, WGPUMapMode_Read, 0, size_t(readbackSize_), cb);
}

void WgpuResolve::onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*) {
    auto* self = static_cast<WgpuResolve*>(userdata1);
    if (status != WGPUMapAsyncStatus_Success) {
        std::fprintf(stderr, "capture readback failed: %s\n", toString(message).c_str());
        self->status_ = CaptureStatus::Failed;
        return;
    }
    const auto* mapped = static_cast<const uint8_t*>(
        wgpuBufferGetConstMappedRange(self->readback_, 0, size_t(self->readbackSize_)));
    const uint32_t rowBytes = self->captureWidth_ * 4;
    self->pixels_.resize(size_t(rowBytes) * self->captureHeight_);
    for (uint32_t y = 0; y < self->captureHeight_; ++y) {  // drop the 256-byte row padding
        std::memcpy(self->pixels_.data() + size_t(y) * rowBytes, mapped + size_t(y) * self->readbackRowPitch_, rowBytes);
    }
    wgpuBufferUnmap(self->readback_);
    self->status_ = CaptureStatus::Ready;
}

void WgpuResolve::release() {
    if (readback_) {
        wgpuBufferDestroy(readback_);
        wgpuBufferRelease(readback_);
    }
    readback_ = nullptr;
    captureTexture_.release();
    if (bindGroup_) wgpuBindGroupRelease(bindGroup_);
    if (canvasPipeline_) wgpuRenderPipelineRelease(canvasPipeline_);
    if (capturePipeline_) wgpuRenderPipelineRelease(capturePipeline_);
    if (pipelineLayout_) wgpuPipelineLayoutRelease(pipelineLayout_);
    if (layout_) wgpuBindGroupLayoutRelease(layout_);
    bindGroup_ = nullptr;
    canvasPipeline_ = capturePipeline_ = nullptr;
    pipelineLayout_ = nullptr;
    layout_ = nullptr;
    boundGeneration_ = 0;
}

}  // namespace gsr
