// TestRenderer (Phase 0): proves both GPU paths work end to end.
//   1. compute pass  - gradient.comp writes a background into the color target
//                      through a storage-image descriptor.
//   2. graphics pass - a colored cube (vertex + index buffer, depth test) drawn on
//                      top with dynamic rendering, loadOp = LOAD to keep the gradient.
#pragma once

#include "core/descriptors.h"
#include "render/renderer.h"

namespace gsr {

class TestRenderer final : public Renderer {
public:
    void init(VulkanContext& ctx, VkFormat colorFormat, VkFormat depthFormat);
    const char* name() const override { return "test"; }
    void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) override;
    RenderOutput render(const FrameInput& in) override;
    void destroy(VulkanContext& ctx) override;

private:
    // Compute (gradient)
    VkDescriptorSetLayout m_gradientSetLayout = VK_NULL_HANDLE;
    DescriptorPool m_descriptorPool;
    VkDescriptorSet m_gradientSet = VK_NULL_HANDLE;
    VkPipelineLayout m_gradientLayout = VK_NULL_HANDLE;
    VkPipeline m_gradientPipeline = VK_NULL_HANDLE;

    // Graphics (cube)
    VkPipelineLayout m_cubeLayout = VK_NULL_HANDLE;
    VkPipeline m_cubePipeline = VK_NULL_HANDLE;
    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;
    uint32_t m_indexCount = 0;
};

}  // namespace gsr
