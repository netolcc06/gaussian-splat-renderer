#include "render/splat_draw.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"

#include <imgui.h>

#include <stdexcept>

namespace gsr {

namespace {

VkFormat vkFormatOf(SplatTargetFormat f) {
    switch (f) {
        case SplatTargetFormat::Rgba8: return VK_FORMAT_R8G8B8A8_UNORM;
        case SplatTargetFormat::Rgba16f: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case SplatTargetFormat::Rgba32f: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

}  // namespace

const char* splatTargetFormatName(SplatTargetFormat f) {
    switch (f) {
        case SplatTargetFormat::Rgba8: return "rgba8";
        case SplatTargetFormat::Rgba16f: return "rgba16f";
        case SplatTargetFormat::Rgba32f: return "rgba32f";
    }
    return "?";
}

std::optional<SplatTargetFormat> parseSplatTargetFormat(std::string_view text) {
    if (text == "rgba8") return SplatTargetFormat::Rgba8;
    if (text == "rgba16f") return SplatTargetFormat::Rgba16f;
    if (text == "rgba32f") return SplatTargetFormat::Rgba32f;
    return std::nullopt;
}

std::optional<CoverageBound> parseCoverageBound(std::string_view text) {
    if (text == "tile") return CoverageBound::Tile;
    if (text == "radius") return CoverageBound::Radius;
    return std::nullopt;
}

void SplatDrawStage::init(VulkanContext& ctx, VkFormat /*colorFormat*/, SplatTargetFormat format,
                          VkPipelineLayout layout, const char* vertexShader) {
    m_ctx = &ctx;
    // A float splat target needs blending on that format and a blit from it (the RGBA8
    // case draws into the app's target, which already supports both).
    for (uint32_t f = 0; f < 3; ++f) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, vkFormatOf(SplatTargetFormat(f)), &props);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        m_formatSupported[f] = (props.optimalTilingFeatures & need) == need;
    }
    if (!m_formatSupported[uint32_t(format)])
        throw std::runtime_error(std::string("splat target format ") + splatTargetFormatName(format) +
                                 " does not support blending + blit on this GPU");
    m_format = m_requestedFormat = format;

    for (uint32_t f = 0; f < 3; ++f) {
        if (!m_formatSupported[f]) continue;
        GraphicsPipelineDesc desc;
        desc.vertexShader = vertexShader;
        desc.fragmentShader = "splat.frag.spv";
        desc.cullMode = VK_CULL_MODE_NONE;
        desc.depthTest = false;  // order comes from the sort, not from a depth buffer
        desc.depthWrite = false;
        desc.alphaBlend = true;  // premultiplied over: src + (1 - src.a) * dst
        desc.colorFormats = {vkFormatOf(SplatTargetFormat(f))};
        m_pipelines[f] = createGraphicsPipeline(ctx.device, layout, desc);
    }
}

void SplatDrawStage::destroy(VulkanContext& ctx) {
    destroyImage(ctx, m_accum);
    for (VkPipeline& p : m_pipelines) {
        if (p) vkDestroyPipeline(ctx.device, p, nullptr);
        p = VK_NULL_HANDLE;
    }
}

void SplatDrawStage::createAccumulationTarget(VkExtent2D extent) {
    destroyImage(*m_ctx, m_accum);
    m_accumState = {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    m_extent = extent;
    if (drawsDirectly()) return;
    m_accum = createImage2D(*m_ctx, extent.width, extent.height, vkFormatOf(m_format),
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            "splat: float accumulation target");
}

void SplatDrawStage::onRenderTargetsChanged(VkExtent2D extent) { createAccumulationTarget(extent); }

void SplatDrawStage::applyRequestedFormat(VkExtent2D extent) {
    // A format change from the UI: wait until no submitted frame uses the old target.
    if (m_requestedFormat == m_format) return;
    VK_CHECK(vkDeviceWaitIdle(m_ctx->device));
    m_format = m_requestedFormat;
    createAccumulationTarget(extent);
}

RenderOutput SplatDrawStage::record(const FrameInput& in, const std::function<void(VkCommandBuffer)>& issueDraw) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;
    const bool direct = drawsDirectly();
    const Image& target = direct ? in.color : m_accum;
    const ImageState before = direct ? in.colorBefore : m_accumState;
    {
        auto timer = in.profiler.scope(cmd, "draw splats (graphics)");
        // Previous contents are discarded (UNDEFINED) and cleared; blending reads the target.
        imageBarrier(cmd, target.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, before.stage, before.access,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = target.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // black background, like INRIA

        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &colorAttachment;
        vkCmdBeginRendering(cmd, &rendering);

        const VkViewport viewport{0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[uint32_t(m_format)]);
        issueDraw(cmd);
        vkCmdEndRendering(cmd);
    }

    RenderOutput out;
    out.depth = in.depthBefore;  // the depth target is not used
    if (direct) {
        out.color = {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
        return out;
    }

    // Float target -> the app's RGBA8 colour target. The blit converts float to UNORM
    // (clamp to [0, 1], round to nearest), the only place the float result is quantized.
    {
        auto timer = in.profiler.scope(cmd, "resolve (blit to rgba8)");
        imageBarrier(cmd, m_accum.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT);
        imageBarrier(cmd, in.color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, in.colorBefore.stage, in.colorBefore.access,
                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
        vkCmdBlitImage(cmd, m_accum.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, in.color.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    // Next frame's draw into m_accum must wait for this blit's reads (execution dependency).
    m_accumState = {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE};
    out.color = {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
    return out;
}

void SplatDrawStage::drawFormatUi() {
    const char* formats[] = {"rgba8", "rgba16f", "rgba32f"};
    int format = int(m_requestedFormat);
    if (ImGui::Combo("Splat target", &format, formats, 3) && m_formatSupported[size_t(format)])
        m_requestedFormat = SplatTargetFormat(format);
}

}  // namespace gsr
