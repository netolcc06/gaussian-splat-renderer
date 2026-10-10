// PointCloudRenderer (Phase 1): debug view of a loaded scene.
//
// Draws every gaussian as a small round screen-space point (fixed pixel size),
// coloured by its degree-0 SH colour clamp(0.5 + C0 * f_dc, 0, 1), with depth test
// and no blending. Gaussians with alpha below a threshold (UI slider) are skipped.
// It reads the shared GpuScene buffers directly (posAlpha + sh), so it also checks
// the GPU data layout that Phase 2 builds on.
#pragma once

#include "core/descriptors.h"
#include "render/renderer.h"

namespace gsr {

struct GpuScene;

class PointCloudRenderer final : public Renderer {
public:
    void init(VulkanContext& ctx, VkFormat colorFormat, VkFormat depthFormat, const GpuScene& scene);
    const char* name() const override { return "points"; }
    void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) override;
    RenderOutput render(const FrameInput& in) override;
    void drawUi() override;
    void destroy(VulkanContext& ctx) override;

    float opacityThreshold = 0.05f;
    float pointSizePx = 2.0f;

private:
    const GpuScene* m_scene = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    DescriptorPool m_descriptorPool;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

}  // namespace gsr
