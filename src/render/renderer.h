// Renderer interface: every rendering technique (test scene, baseline splatting,
// tile rasterizer, ...) implements this, so the app can switch between them with
// --renderer or from the UI and compare them on identical inputs.
#pragma once

#include "core/resources.h"

#include <glm/glm.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gsr {

class VulkanContext;
class GpuProfiler;
class CpuProfiler;

// Where an image is after a renderer is done with it, so the app can build the
// next barrier (srcStage/srcAccess) without knowing how the renderer worked.
struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 access;
};

struct FrameInput {
    VkCommandBuffer cmd;
    GpuProfiler& profiler;
    CpuProfiler& cpuProfiler;  // CPU-side stages (e.g. the Phase 2 CPU depth sort)
    // Render targets. Their previous contents are discarded: the renderer transitions
    // them from VK_IMAGE_LAYOUT_UNDEFINED and must write every pixel of `color`.
    const Image& color;
    const Image& depth;
    // Last pipeline stage/access that touched each target (in the previous frame).
    // The renderer's first barrier on each image must use these as its source scope,
    // otherwise this frame could overwrite the image while the previous frame is
    // still reading it (a write-after-read hazard).
    ImageState colorBefore;
    ImageState depthBefore;
    glm::mat4 view;
    glm::mat4 proj;  // Vulkan clip space (y down, depth 0..1)
    // Scene transform (scene data space -> world), see scene/scene_transform.h.
    // Gaussian renderers use proj * view * model; the test renderer ignores it.
    glm::mat4 model;
    uint64_t frameIndex;
    // Frame-in-flight slot (frameIndex % framesInFlight). The CPU waited for the GPU to
    // finish the previous frame that used this slot, so per-slot resources written by
    // the CPU (e.g. host-visible upload buffers) are free to overwrite.
    uint32_t frameSlot;
};

struct RenderOutput {
    ImageState color;  // the app copies `color` to the swapchain / screenshot next
    ImageState depth;
};

class Renderer {
public:
    virtual ~Renderer() = default;
    virtual const char* name() const = 0;
    // Called whenever the render targets are (re)created, e.g. after a resize.
    virtual void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) = 0;
    // Records all GPU work for one frame and returns the final state of both targets.
    virtual RenderOutput render(const FrameInput& in) = 0;
    // Optional renderer-specific ImGui controls.
    virtual void drawUi() {}
    // Phase 6 `--pipeline-stats`: the renderer's pipelines, labelled by shader.
    virtual std::vector<std::pair<std::string, VkPipeline>> pipelines() const { return {}; }
    virtual void destroy(VulkanContext& ctx) = 0;
};

}  // namespace gsr
