#include "render/point_cloud_renderer.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"
#include "render/gpu_scene.h"

#include <imgui.h>

namespace gsr {

namespace {

// Must match the push_constant block in shaders/points.vert.
struct PointPushConstants {
    glm::mat4 mvp;
    glm::vec2 viewportSize;
    float pointSizePx;
    float opacityThreshold;
    uint32_t shCoeffs;
};
static_assert(sizeof(PointPushConstants) <= 128, "push constants must fit the guaranteed 128 bytes");

}  // namespace

void PointCloudRenderer::init(VulkanContext& ctx, VkFormat colorFormat, VkFormat depthFormat, const GpuScene& scene) {
    VkDevice device = ctx.device;
    m_scene = &scene;

    // Bindings 0-3: the scene buffers (any scene format, shaders/scene_decode.glsl).
    DescriptorLayoutBuilder builder;
    for (uint32_t b = 0; b < 4; ++b) builder.add(b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
    m_setLayout = builder.build(device);
    m_descriptorPool.create(device, 1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}});
    m_set = m_descriptorPool.allocate(device, m_setLayout);
    DescriptorWriter writer;
    scene.bind(writer);
    writer.update(device, m_set);
    SpecializationConstants spec;
    scene.addSpecialization(spec);

    const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PointPushConstants)};
    m_layout = createPipelineLayout(device, {m_setLayout}, {push});

    GraphicsPipelineDesc desc;
    desc.vertexShader = "points.vert.spv";
    desc.fragmentShader = "points.frag.spv";
    desc.vertexSpecialization = spec.info();
    // No vertex buffers: the shader fetches from storage buffers by gl_InstanceIndex.
    desc.cullMode = VK_CULL_MODE_NONE;
    desc.depthTest = true;
    desc.depthWrite = true;
    desc.depthCompare = VK_COMPARE_OP_LESS;
    desc.colorFormats = {colorFormat};
    desc.depthFormat = depthFormat;
    m_pipeline = createGraphicsPipeline(device, m_layout, desc);
}

void PointCloudRenderer::onRenderTargetsChanged(VulkanContext&, const Image&, const Image&) {
    // Render targets are attachments (bound per frame), so there is nothing to update.
}

RenderOutput PointCloudRenderer::render(const FrameInput& in) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;
    auto timer = in.profiler.scope(cmd, "points (graphics)");

    // Previous contents are discarded (UNDEFINED); wait for the previous frame's last use.
    imageBarrier(cmd, in.color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, in.colorBefore.stage, in.colorBefore.access,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(cmd, in.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, in.depthBefore.stage, in.depthBefore.access,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = in.color.view;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.06f, 0.06f, 0.08f, 1.0f}};

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = in.depth.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {{0, 0}, extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &colorAttachment;
    rendering.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(cmd, &rendering);

    const VkViewport viewport{0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &m_set, 0, nullptr);
    PointPushConstants pc{};
    pc.mvp = in.proj * in.view * in.model;
    pc.viewportSize = glm::vec2(float(extent.width), float(extent.height));
    pc.pointSizePx = pointSizePx;
    pc.opacityThreshold = opacityThreshold;
    pc.shCoeffs = m_scene->shCoeffsPerChannel;
    vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 6, m_scene->count, 0, 0);  // 6 vertices (one quad) per gaussian instance

    vkCmdEndRendering(cmd);

    RenderOutput out;
    out.color = {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    out.depth = {VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    return out;
}

void PointCloudRenderer::drawUi() {
    ImGui::SliderFloat("Opacity threshold", &opacityThreshold, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Point size (px)", &pointSizePx, 1.0f, 16.0f, "%.1f");
}

void PointCloudRenderer::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    vkDestroyPipeline(device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(device, m_layout, nullptr);
    m_descriptorPool.destroy(device);
    vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_setLayout = VK_NULL_HANDLE;
}

}  // namespace gsr
