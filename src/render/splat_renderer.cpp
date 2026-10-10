#include "render/splat_renderer.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"
#include "render/gpu_scene.h"
#include "scene/gaussian_scene.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gsr {

namespace {

// Must match the push_constant block in shaders/splat_preprocess.comp (std430 offsets:
// 0, 64, 80, 88, 96, 104, 108, 112).
struct PreprocessPushConstants {
    glm::mat4 viewInria;
    glm::vec4 camPos;
    glm::vec2 focal;
    glm::vec2 tanHalfFov;
    glm::uvec2 imageSize;
    uint32_t count;
    uint32_t shCoeffs;
    uint32_t shDegree;
};
static_assert(sizeof(PreprocessPushConstants) == 116, "layout must match splat_preprocess.comp");

// Must match the push_constant block in shaders/splat.vert.
struct DrawPushConstants {
    glm::vec2 viewportSize;
    uint32_t visibleCount;
    uint32_t bound;
};

constexpr uint32_t kPreprocessGroupSize = 256;  // local_size_x in splat_preprocess.comp

}  // namespace

void SplatRenderer::init(VulkanContext& ctx, VkFormat colorFormat, const GpuScene& gpuScene,
                         const GaussianScene& cpuScene, uint32_t framesInFlight, const SplatSettings& settings) {
    VkDevice device = ctx.device;
    m_ctx = &ctx;
    m_gpuScene = &gpuScene;
    m_cpuScene = &cpuScene;
    m_settings = settings;
    m_settings.maxShDegree = std::clamp(m_settings.maxShDegree, 0, 3);

    const uint32_t count = gpuScene.count;

    // --- Buffers ---------------------------------------------------------------------
    m_splats = createBuffer(ctx, VkDeviceSize(count) * sizeof(ProjectedSplat), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            MemoryUsage::GpuOnly, "splat: per-splat data");
    m_indices = createBuffer(ctx, VkDeviceSize(count) * sizeof(uint32_t),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::GpuOnly, "splat: sorted indices");
    m_staging.resize(framesInFlight);
    for (Buffer& b : m_staging)
        b = createBuffer(ctx, VkDeviceSize(count) * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         MemoryUsage::Upload, "splat: sorted indices (staging)");

    // --- Descriptors -------------------------------------------------------------------
    m_preprocessSetLayout = DescriptorLayoutBuilder()
                                .add(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // scene 0-3
                                .add(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
                                .add(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
                                .add(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
                                .add(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // splats (out)
                                .build(device);
    m_drawSetLayout = DescriptorLayoutBuilder()
                          .add(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // splats
                          .add(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // sorted indices
                          .build(device);
    m_descriptorPool.create(device, 2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}});
    m_preprocessSet = m_descriptorPool.allocate(device, m_preprocessSetLayout);
    m_drawSet = m_descriptorPool.allocate(device, m_drawSetLayout);
    {
        DescriptorWriter writer;
        gpuScene.bind(writer);  // bindings 0-3: the scene, any format
        writer.buffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_splats.buffer).update(device, m_preprocessSet);
    }
    DescriptorWriter()
        .buffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_splats.buffer)
        .buffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_indices.buffer)
        .update(device, m_drawSet);

    // --- Pipelines ---------------------------------------------------------------------
    const VkPushConstantRange preprocessPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PreprocessPushConstants)};
    m_preprocessLayout = createPipelineLayout(device, {m_preprocessSetLayout}, {preprocessPush});
    SpecializationConstants spec;
    gpuScene.addSpecialization(spec);  // the scene decoder's format (scene_decode.glsl)
    m_preprocessPipeline = createComputePipeline(device, m_preprocessLayout, "splat_preprocess.comp.spv", spec.info());

    const VkPushConstantRange drawPush{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawPushConstants)};
    m_drawLayout = createPipelineLayout(device, {m_drawSetLayout}, {drawPush});
    m_draw.init(ctx, colorFormat, m_settings.format, m_drawLayout, "splat.vert.spv");
}

int SplatRenderer::effectiveShDegree() const { return std::min(m_gpuScene->shDegree, m_settings.maxShDegree); }

void SplatRenderer::onRenderTargetsChanged(VulkanContext&, const Image& color, const Image&) {
    // Called with the GPU idle: the intermediate target follows the colour target's size.
    m_draw.onRenderTargetsChanged(color.extent);
}

RenderOutput SplatRenderer::render(const FrameInput& in) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;

    m_draw.applyRequestedFormat(extent);  // a format change from the UI (waits for idle)

    const SplatCamera cam = splatCameraFromViewer(in.view, in.proj, in.model, extent.width, extent.height);
    const uint32_t count = m_gpuScene->count;

    // === Stage 1: CPU depth sort + upload into this slot's staging buffer ==============
    m_sorter.sort(m_cpuScene->posAlpha.data(), count, cam.viewInria);
    in.cpuProfiler.add("sort: depth keys (cpu)", m_sorter.depthMs);
    in.cpuProfiler.add("sort: radix (cpu)", m_sorter.sortMs);
    in.cpuProfiler.add("sort: total (cpu)", m_sorter.depthMs + m_sorter.sortMs);
    const uint32_t visible = uint32_t(m_sorter.order().size());
    m_lastVisible = visible;
    const Buffer& staging = m_staging[in.frameSlot];
    {
        auto t = in.cpuProfiler.scope("upload indices: memcpy (cpu)");
        std::memcpy(staging.mapped, m_sorter.order().data(), size_t(visible) * sizeof(uint32_t));
        // No-op on host-coherent memory. The submit that follows makes host writes visible.
        VK_CHECK(vmaFlushAllocation(m_ctx->allocator, staging.allocation, 0, VK_WHOLE_SIZE));
    }

    // Write-after-read: the previous frame's vertex shader read m_indices and m_splats,
    // which this frame's copy and compute pass overwrite. Execution dependency only.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_NONE,
                  VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE);
    {
        auto timer = in.profiler.scope(cmd, "upload indices (copy)");
        if (visible > 0) {
            const VkBufferCopy region{0, 0, VkDeviceSize(visible) * sizeof(uint32_t)};
            vkCmdCopyBuffer(cmd, staging.buffer, m_indices.buffer, 1, &region);
        }
    }

    // === Stage 2: preprocess (compute) ================================================
    {
        auto timer = in.profiler.scope(cmd, "preprocess (compute)");
        PreprocessPushConstants pc{};
        pc.viewInria = cam.viewInria;
        pc.camPos = glm::vec4(cam.position, 1.0f);
        pc.focal = {cam.focalX, cam.focalY};
        pc.tanHalfFov = {cam.tanHalfFovX, cam.tanHalfFovY};
        pc.imageSize = {extent.width, extent.height};
        pc.count = count;
        pc.shCoeffs = m_gpuScene->shCoeffsPerChannel;
        pc.shDegree = uint32_t(effectiveShDegree());
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessLayout, 0, 1, &m_preprocessSet, 0,
                                nullptr);
        vkCmdPushConstants(cmd, m_preprocessLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (count + kPreprocessGroupSize - 1) / kPreprocessGroupSize, 1, 1);
    }

    // Read-after-write: copy (indices) and compute (splats) -> vertex shader reads.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // === Stage 3: draw (instanced quads, back to front, premultiplied over) ===========
    return m_draw.record(in, [&](VkCommandBuffer drawCmd) {
        if (visible == 0) return;
        vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawLayout, 0, 1, &m_drawSet, 0, nullptr);
        DrawPushConstants pc{};
        pc.viewportSize = {float(extent.width), float(extent.height)};
        pc.visibleCount = visible;
        pc.bound = uint32_t(m_settings.bound) | (uint32_t(m_settings.bounds) << 8);  // splat.vert
        vkCmdPushConstants(drawCmd, m_drawLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(drawCmd, 6, visible, 0, 0);  // 6 vertices (one quad) per splat instance
    });
}

void SplatRenderer::drawUi() {
    m_draw.drawFormatUi();
    const char* bounds[] = {"tile (INRIA)", "radius"};
    int bound = int(m_settings.bound);
    if (ImGui::Combo("Quad extent", &bound, bounds, 2)) m_settings.bound = CoverageBound(bound);
    ImGui::SliderInt("Max SH degree", &m_settings.maxShDegree, 0, 3);
    ImGui::Text("Drawn instances: %u of %u (near-culled on CPU)", m_lastVisible, m_gpuScene->count);
    ImGui::Text("SH degree used: %d (scene %d)", effectiveShDegree(), m_gpuScene->shDegree);
}

void SplatRenderer::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    m_draw.destroy(ctx);
    vkDestroyPipelineLayout(device, m_drawLayout, nullptr);
    vkDestroyPipeline(device, m_preprocessPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_preprocessLayout, nullptr);
    m_descriptorPool.destroy(device);
    vkDestroyDescriptorSetLayout(device, m_drawSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_preprocessSetLayout, nullptr);
    destroyBuffer(ctx, m_splats);
    destroyBuffer(ctx, m_indices);
    for (Buffer& b : m_staging) destroyBuffer(ctx, b);
    m_staging.clear();
    m_drawLayout = m_preprocessLayout = VK_NULL_HANDLE;
    m_preprocessPipeline = VK_NULL_HANDLE;
    m_drawSetLayout = m_preprocessSetLayout = VK_NULL_HANDLE;
}

}  // namespace gsr
