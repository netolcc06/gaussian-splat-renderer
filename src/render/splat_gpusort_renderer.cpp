#include "render/splat_gpusort_renderer.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"
#include "render/gpu_scene.h"
#include "scene/gaussian_scene.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace gsr {

namespace {

// Must match the push_constant block in shaders/splat_preprocess.comp built with
// GSR_COMPACT (the Phase 2 block + flags).
struct PreprocessPushConstants {
    glm::mat4 viewInria;
    glm::vec4 camPos;
    glm::vec2 focal;
    glm::vec2 tanHalfFov;
    glm::uvec2 imageSize;
    uint32_t count;
    uint32_t shCoeffs;
    uint32_t shDegree;
    uint32_t flags;
};
static_assert(sizeof(PreprocessPushConstants) == 120, "layout must match splat_preprocess.comp (GSR_COMPACT)");
constexpr uint32_t kFlagAppend = 1;    // kFlagAppend in the shader
constexpr uint32_t kFlagWriteAll = 2;  // kFlagWriteAll in the shader

// Must match the push_constant block in shaders/splat.vert (visibleCount unused here).
struct DrawPushConstants {
    glm::vec2 viewportSize;
    uint32_t visibleCount;
    uint32_t bound;
};

constexpr uint32_t kPreprocessGroupSize = 256;  // local_size_x in splat_preprocess.comp
constexpr VkDeviceSize kCountersOffset = offsetof(SplatControl, cullCounts);

const char* const kCullNames[kCullReasonCount] = {"visible", "near plane", "singular cov.", "off screen",
                                                  "opacity < 1/255"};

}  // namespace

void SplatGpuSortRenderer::init(VulkanContext& ctx, VkFormat colorFormat, const GpuScene& gpuScene,
                                const GaussianScene& cpuScene, uint32_t framesInFlight, const SplatSettings& settings,
                                const GpuSortSettings& sortSettings) {
    VkDevice device = ctx.device;
    m_ctx = &ctx;
    m_gpuScene = &gpuScene;
    m_settings = settings;
    m_settings.maxShDegree = std::clamp(m_settings.maxShDegree, 0, 3);
    m_sortSettings = sortSettings;
    m_sceneCentre = 0.5f * (cpuScene.stats.robustMin + cpuScene.stats.robustMax);

    const VkDeviceSize count = std::max<VkDeviceSize>(gpuScene.count, 1);

    // --- Buffers ---------------------------------------------------------------------
    m_splats = createBuffer(ctx, count * sizeof(ProjectedSplat), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            MemoryUsage::GpuOnly, "splat-gpusort: per-splat data");
    m_sortKeys = createBuffer(ctx, count * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              MemoryUsage::GpuOnly, "splat-gpusort: sort keys");
    m_sortValues = createBuffer(ctx, count * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                MemoryUsage::GpuOnly, "splat-gpusort: sorted indices");
    m_control = createBuffer(ctx, sizeof(SplatControl),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             MemoryUsage::GpuOnly, "splat-gpusort: control (draw args + cull counters)");
    m_statsReadback.resize(framesInFlight);
    for (Buffer& b : m_statsReadback)
        b = createBuffer(ctx, sizeof(SplatControl), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback,
                         "splat-gpusort: stats readback");
    m_slotInfo.assign(framesInFlight, SlotInfo{});

    // --- Descriptors -------------------------------------------------------------------
    DescriptorLayoutBuilder pre;
    for (uint32_t b = 0; b <= 7; ++b) pre.add(b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    m_preprocessSetLayout = pre.build(device);
    m_drawSetLayout = DescriptorLayoutBuilder()
                          .add(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // splats
                          .add(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // sorted indices
                          .add(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // draw args
                          .build(device);
    m_descriptorPool.create(device, 2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11}});
    m_preprocessSet = m_descriptorPool.allocate(device, m_preprocessSetLayout);
    m_drawSet = m_descriptorPool.allocate(device, m_drawSetLayout);
    DescriptorWriter preWriter;
    gpuScene.bind(preWriter);  // bindings 0-3: the scene, any format
    preWriter.buffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_splats.buffer)
        .buffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_sortKeys.buffer)
        .buffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_sortValues.buffer)
        .buffer(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_control.buffer)
        .update(device, m_preprocessSet);
    DescriptorWriter()
        .buffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_splats.buffer)
        .buffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_sortValues.buffer)
        .buffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_control.buffer)
        .update(device, m_drawSet);

    // --- Pipelines ---------------------------------------------------------------------
    const VkPushConstantRange preprocessPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PreprocessPushConstants)};
    m_preprocessLayout = createPipelineLayout(device, {m_preprocessSetLayout}, {preprocessPush});
    // Full subgroups: the ballot-based append assumes every lane of a subgroup exists.
    SpecializationConstants spec;
    gpuScene.addSpecialization(spec);  // the scene decoder's format (scene_decode.glsl)
    m_preprocessPipeline = createComputePipeline(device, m_preprocessLayout, "splat_preprocess_compact.comp.spv",
                                                 spec.info(), VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);

    // The sort reads the visible count straight from the draw arguments.
    m_sort.init(ctx, gpuScene.count, m_sortKeys.buffer, m_sortValues.buffer, m_control.buffer,
                uint32_t(offsetof(SplatControl, draw) + offsetof(VkDrawIndirectCommand, instanceCount)));
    m_sort.setVariant(m_sortSettings.radixVariant);

    const VkPushConstantRange drawPush{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawPushConstants)};
    m_drawLayout = createPipelineLayout(device, {m_drawSetLayout}, {drawPush});
    m_draw.init(ctx, colorFormat, m_settings.format, m_drawLayout, "splat_indirect.vert.spv");
}

int SplatGpuSortRenderer::effectiveShDegree() const { return std::min(m_gpuScene->shDegree, m_settings.maxShDegree); }

void SplatGpuSortRenderer::onRenderTargetsChanged(VulkanContext&, const Image& color, const Image&) {
    m_draw.onRenderTargetsChanged(color.extent);
    m_haveSortedList = false;  // the culling depends on the screen size: sort again
}

bool SplatGpuSortRenderer::shouldSort(const SplatCamera& cam, VkExtent2D extent) const {
    if (!m_haveSortedList || extent.width != m_lastSortExtent.width || extent.height != m_lastSortExtent.height)
        return true;
    const GpuSortSettings& s = m_sortSettings;
    if (s.sortEvery == 1) return true;
    if (s.sortEvery > 1 && m_framesRendered - m_lastSortFrame >= s.sortEvery) return true;
    if (s.sortThresholdDeg > 0.0f) {
        // Rotation angle between the two camera orientations: R = R_last^T R_now,
        // angle = acos((trace R - 1) / 2).
        const glm::mat3 r = glm::transpose(glm::mat3(m_lastSortView)) * glm::mat3(cam.viewInria);
        const float c = std::clamp((r[0][0] + r[1][1] + r[2][2] - 1.0f) * 0.5f, -1.0f, 1.0f);
        const float rotationDeg = glm::degrees(std::acos(c));
        // Translation as the parallax angle it causes at the scene centre.
        const float dist = std::max(glm::length(m_sceneCentre - m_lastSortPos), 1e-6f);
        const float parallaxDeg = glm::degrees(std::atan(glm::length(cam.position - m_lastSortPos) / dist));
        if (rotationDeg > s.sortThresholdDeg || parallaxDeg > s.sortThresholdDeg) return true;
    }
    return false;
}

RenderOutput SplatGpuSortRenderer::render(const FrameInput& in) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;
    m_draw.applyRequestedFormat(extent);  // a format change from the UI (waits for idle)

    // Statistics of the frame that last used this slot: the app waited for it, so the
    // readback is complete (no stall). Invalidate in case the memory is not coherent.
    SlotInfo& slot = m_slotInfo[in.frameSlot];
    if (slot.used) {
        const Buffer& rb = m_statsReadback[in.frameSlot];
        VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
        if (!m_lastStats.valid || slot.frameIndex >= m_lastStats.frameIndex) {
            std::memcpy(&m_lastStats.control, rb.mapped, sizeof(SplatControl));
            m_lastStats.valid = true;
            m_lastStats.frameIndex = slot.frameIndex;
            m_lastStats.sorted = slot.sorted;
        }
    }

    const SplatCamera cam = splatCameraFromViewer(in.view, in.proj, in.model, extent.width, extent.height);
    const uint32_t count = m_gpuScene->count;
    const bool sortThisFrame = shouldSort(cam, extent);

    // Frame boundary. The previous frame's draw read m_splats / m_sortValues / m_control
    // (vertex shader, indirect command) and copied m_control; its compute passes wrote
    // and read every buffer here. This frame's transfers and compute passes overwrite
    // them: wait for all of it (WAR) and make the previous writes available (WAW).
    memoryBarrier(cmd,
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 0. Reset the control block ===================================================
    {
        auto timer = in.profiler.scope(cmd, "reset counters (transfer)");
        if (sortThisFrame) {
            // instanceCount is the append counter of this frame's compact list.
            const VkDrawIndirectCommand args{6, 0, 0, 0};
            vkCmdUpdateBuffer(cmd, m_control.buffer, 0, sizeof(args), &args);
        }
        vkCmdFillBuffer(cmd, m_control.buffer, kCountersOffset, sizeof(SplatControl) - kCountersOffset, 0);
    }
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 1. Preprocess + cull + compact ================================================
    {
        auto timer = in.profiler.scope(cmd, "preprocess + cull + compact (compute)");
        PreprocessPushConstants pc{};
        pc.viewInria = cam.viewInria;
        pc.camPos = glm::vec4(cam.position, 1.0f);
        pc.focal = {cam.focalX, cam.focalY};
        pc.tanHalfFov = {cam.tanHalfFovX, cam.tanHalfFovY};
        pc.imageSize = {extent.width, extent.height};
        pc.count = count;
        pc.shCoeffs = m_gpuScene->shCoeffsPerChannel;
        pc.shDegree = uint32_t(effectiveShDegree());
        pc.flags = sortThisFrame ? kFlagAppend : kFlagWriteAll;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessLayout, 0, 1, &m_preprocessSet, 0,
                                nullptr);
        vkCmdPushConstants(cmd, m_preprocessLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (count + kPreprocessGroupSize - 1) / kPreprocessGroupSize, 1, 1);
    }

    // Cull statistics -> this slot's host buffer (read when the slot comes around again).
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    {
        const VkBufferCopy region{0, 0, sizeof(SplatControl)};
        vkCmdCopyBuffer(cmd, m_control.buffer, m_statsReadback[in.frameSlot].buffer, 1, &region);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    }
    slot = {true, sortThisFrame, in.frameIndex};

    // === 2. Sort the compact list (count from the GPU, indirect dispatches) ===========
    if (sortThisFrame) {
        auto timer = in.profiler.scope(cmd, "sort (total)");
        m_sort.setVariant(m_sortSettings.radixVariant);
        m_sort.record(cmd, &in.profiler);  // begins with its own compute -> compute barrier
        m_haveSortedList = true;
        m_lastSortFrame = m_framesRendered;
        m_lastSortView = cam.viewInria;
        m_lastSortPos = cam.position;
        m_lastSortExtent = extent;
        ++m_sortsPerformed;
    }
    ++m_framesRendered;

    // Everything the draw consumes was last written by a compute shader: the sorted
    // indices (sort), the splats (preprocess) and the instance count (preprocess atomics),
    // which is read both as an indirect argument and by the vertex shader.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);

    // === 3. Indirect draw ================================================================
    return m_draw.record(in, [&](VkCommandBuffer drawCmd) {
        vkCmdBindDescriptorSets(drawCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawLayout, 0, 1, &m_drawSet, 0, nullptr);
        DrawPushConstants pc{};
        pc.viewportSize = {float(extent.width), float(extent.height)};
        pc.visibleCount = 0;  // unused: the vertex shader reads the GPU-written count
        pc.bound = uint32_t(m_settings.bound) | (uint32_t(m_settings.bounds) << 8);  // splat.vert
        vkCmdPushConstants(drawCmd, m_drawLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdDrawIndirect(drawCmd, m_control.buffer, offsetof(SplatControl, draw), 1, sizeof(VkDrawIndirectCommand));
    });
}

void SplatGpuSortRenderer::readBack(std::vector<uint32_t>& keys, std::vector<uint32_t>& values,
                                    SplatControl& control) {
    const VkDeviceSize n = m_sortKeys.size / 4;
    Buffer rb = createBuffer(*m_ctx, (2 * n) * 4 + sizeof(SplatControl), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::Readback, "splat-gpusort: debug readback");
    m_ctx->immediateSubmit([&](VkCommandBuffer cmd) {
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy k{0, 0, n * 4}, v{0, n * 4, n * 4}, c{0, 2 * n * 4, sizeof(SplatControl)};
        vkCmdCopyBuffer(cmd, m_sortKeys.buffer, rb.buffer, 1, &k);
        vkCmdCopyBuffer(cmd, m_sortValues.buffer, rb.buffer, 1, &v);
        vkCmdCopyBuffer(cmd, m_control.buffer, rb.buffer, 1, &c);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* data = static_cast<const uint8_t*>(rb.mapped);
    std::memcpy(&control, data + 2 * n * 4, sizeof(SplatControl));
    const uint32_t visible = std::min<uint32_t>(control.draw.instanceCount, uint32_t(n));
    keys.resize(visible);
    values.resize(visible);
    std::memcpy(keys.data(), data, size_t(visible) * 4);
    std::memcpy(values.data(), data + n * 4, size_t(visible) * 4);
    destroyBuffer(*m_ctx, rb);
}

void SplatGpuSortRenderer::drawUi() {
    m_draw.drawFormatUi();
    const char* bounds[] = {"tile (INRIA)", "radius"};
    int bound = int(m_settings.bound);
    if (ImGui::Combo("Quad extent", &bound, bounds, 2)) m_settings.bound = CoverageBound(bound);
    ImGui::SliderInt("Max SH degree", &m_settings.maxShDegree, 0, 3);
    ImGui::Text("SH degree used: %d (scene %d)", effectiveShDegree(), m_gpuScene->shDegree);

    int every = int(m_sortSettings.sortEvery);
    if (ImGui::SliderInt("Sort every N frames (0 = threshold only)", &every, 0, 60))
        m_sortSettings.sortEvery = uint32_t(std::max(every, 0));
    ImGui::SliderFloat("Re-sort threshold (deg, 0 = off)", &m_sortSettings.sortThresholdDeg, 0.0f, 20.0f, "%.2f");
    const char* variants[] = {"classic (reference)", "reorder (local sort + coalesced writes)"};
    int variant = int(m_sortSettings.radixVariant);
    if (ImGui::Combo("Radix scatter", &variant, variants, 2))
        m_sortSettings.radixVariant = GpuRadixSort::Variant(variant);
    ImGui::Text("Sorted %llu of %llu frames", static_cast<unsigned long long>(m_sortsPerformed),
                static_cast<unsigned long long>(m_framesRendered));

    if (m_lastStats.valid) {
        const SplatControl& c = m_lastStats.control;
        ImGui::Text("Frame %llu (%s): %u instances drawn of %u", static_cast<unsigned long long>(m_lastStats.frameIndex),
                    m_lastStats.sorted ? "sorted" : "reused order", c.draw.instanceCount, m_gpuScene->count);
        if (ImGui::BeginTable("cull", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Cull reason");
            ImGui::TableSetupColumn("Gaussians");
            ImGui::TableSetupColumn("%");
            ImGui::TableHeadersRow();
            for (uint32_t r = 0; r < kCullReasonCount; ++r) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(kCullNames[r]);
                ImGui::TableNextColumn();
                ImGui::Text("%u", c.cullCounts[r]);
                ImGui::TableNextColumn();
                ImGui::Text("%.1f", 100.0 * double(c.cullCounts[r]) / double(std::max(m_gpuScene->count, 1u)));
            }
            ImGui::EndTable();
        }
    }
}

void SplatGpuSortRenderer::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    m_draw.destroy(ctx);
    m_sort.destroy(ctx);
    vkDestroyPipelineLayout(device, m_drawLayout, nullptr);
    vkDestroyPipeline(device, m_preprocessPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_preprocessLayout, nullptr);
    m_descriptorPool.destroy(device);
    vkDestroyDescriptorSetLayout(device, m_drawSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_preprocessSetLayout, nullptr);
    for (Buffer* b : {&m_splats, &m_sortKeys, &m_sortValues, &m_control}) destroyBuffer(ctx, *b);
    for (Buffer& b : m_statsReadback) destroyBuffer(ctx, b);
    m_statsReadback.clear();
    m_drawLayout = m_preprocessLayout = VK_NULL_HANDLE;
    m_preprocessPipeline = VK_NULL_HANDLE;
    m_drawSetLayout = m_preprocessSetLayout = VK_NULL_HANDLE;
}

const char* cullReasonName(uint32_t r) { return r < kCullReasonCount ? kCullNames[r] : "?"; }

}  // namespace gsr
