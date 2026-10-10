// Resolve: rgba16float render target -> 8-bit color (shaders/wgsl/resolve.wgsl). Replaces the
// native vkCmdBlitImage. Two pipelines share one module and one explicit bind group layout:
//   - to the canvas texture (the surface's preferred format), every frame;
//   - to an rgba8unorm capture texture, on request, followed by a copy into a MAP_READ buffer
//     and an asynchronous map (the web counterpart of --screenshot; the test reads it back
//     through gsr_capture_*). The capture is the offscreen image, without any UI, and does
//     not depend on how the browser composites the canvas.
#pragma once

#include "webgpu/wgpu_resources.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <vector>

namespace gsr {

class WgslShaderCache;

enum class CaptureStatus : int { None = 0, Requested = 1, Mapping = 2, Ready = 3, Failed = -1 };

class WgpuResolve {
public:
    WgpuResolve() = default;
    WgpuResolve(const WgpuResolve&) = delete;
    WgpuResolve& operator=(const WgpuResolve&) = delete;
    ~WgpuResolve() { release(); }

    bool create(WGPUDevice device, WgslShaderCache& shaders, WGPUTextureFormat canvasFormat);
    void release();

    // Full-screen pass: source -> canvas texture view.
    void encodeToCanvas(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& source,
                        WGPUTextureView canvasView);

    // Capture of the next frame: requestCapture() (false if one is in flight), then each frame
    // encodeCapture() (does nothing unless requested) and, after the submit, startMapping().
    bool requestCapture();
    bool encodeCapture(WGPUDevice device, WGPUCommandEncoder encoder, const WgpuTexture2D& source);
    void startMapping();
    CaptureStatus captureStatus() const { return status_; }
    // Tightly packed RGBA8 rows, top row first (valid when Ready).
    const std::vector<uint8_t>& capturePixels() const { return pixels_; }
    uint32_t captureWidth() const { return captureWidth_; }
    uint32_t captureHeight() const { return captureHeight_; }

private:
    void bindSource(WGPUDevice device, const WgpuTexture2D& source);
    void draw(WGPUCommandEncoder encoder, WGPURenderPipeline pipeline, WGPUTextureView target, const char* label);
    static void onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2);

    WGPUBindGroupLayout layout_ = nullptr;
    WGPUPipelineLayout pipelineLayout_ = nullptr;
    WGPURenderPipeline canvasPipeline_ = nullptr;
    WGPURenderPipeline capturePipeline_ = nullptr;
    WGPUBindGroup bindGroup_ = nullptr;
    uint64_t boundGeneration_ = 0;

    WgpuTexture2D captureTexture_;
    WGPUBuffer readback_ = nullptr;
    uint64_t readbackSize_ = 0;
    uint32_t readbackRowPitch_ = 0;
    CaptureStatus status_ = CaptureStatus::None;
    std::vector<uint8_t> pixels_;
    uint32_t captureWidth_ = 0, captureHeight_ = 0;
};

}  // namespace gsr
