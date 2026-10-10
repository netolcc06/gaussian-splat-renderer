#include "render/gpu_scene.h"

#include "core/context.h"
#include "core/descriptors.h"
#include "core/pipelines.h"
#include "scene/gaussian_scene.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace gsr {

bool GpuScene::zeroCopy(const GaussianScene& scene, const CompressionOptions& options) {
    return options.format == SceneFormat::F32 && options.layout == SceneLayout::Soa &&
           options.maxShDegree >= scene.shDegree && !options.fp16Positions && options.shThreshold < 0.0f;
}

void GpuScene::upload(VulkanContext& ctx, const GaussianScene& scene, const CompressionOptions& options,
                      EncodedScene* preEncoded) {
    MemoryScope memoryScope(ctx, "scene");
    count = scene.count;

    struct Part {
        const void* data;
        VkDeviceSize size;
        std::string name;
    };
    std::vector<Part> parts;  // one per binding 0..3, size 0 = no buffer
    EncodedScene encoded;     // must outlive the staging copy

    if (zeroCopy(scene, options)) {
        // The Phase 1-4 path: the GaussianScene arrays are already the GPU layout.
        format = SceneFormat::F32;
        layout = SceneLayout::Soa;
        geometry = GeometryEncoding::SoaF32;
        color = ColorEncoding::SoaF32;
        shDegree = scene.shDegree;
        shCoeffsPerChannel = scene.shCoeffsPerChannel();
        geometryStride = colorStride = colorOffset = codebookStride = 0;
        variableSh = VariableSh::Off;
        degreeHistogram = {};
        encodeMs = 0.0;
        codebook = {};
        parts = {{scene.posAlpha.data(), scene.posAlpha.size() * sizeof(glm::vec4), "gaussians: posAlpha"},
                 {scene.scales.data(), scene.scales.size() * sizeof(glm::vec4), "gaussians: scales"},
                 {scene.rotations.data(), scene.rotations.size() * sizeof(glm::vec4), "gaussians: rotations"},
                 {scene.sh.data(), scene.sh.size() * sizeof(float), "gaussians: sh"}};
    } else {
        encoded = preEncoded ? std::move(*preEncoded) : encodeScene(scene, options);
        format = encoded.format;
        layout = encoded.layout;
        geometry = encoded.geometry;
        color = encoded.color;
        shDegree = encoded.shDegree;
        shCoeffsPerChannel = encoded.shCoeffs;
        geometryStride = encoded.geometryStride;
        colorStride = encoded.colorStride;
        colorOffset = encoded.colorOffset;
        codebookStride = encoded.codebookStride;
        variableSh = encoded.variableSh;
        degreeHistogram = encoded.degreeHistogram;
        encodeMs = encoded.encodeMs;
        codebook = encoded.codebook;
        for (size_t b = 0; b < encoded.buffers.size(); ++b)
            parts.push_back({encoded.buffers[b].data(), encoded.buffers[b].size() * 4, encoded.bufferNames[b]});
    }
    parts.resize(4, Part{nullptr, 0, ""});

    const auto start = std::chrono::steady_clock::now();
    // A single staging buffer holding every array back to back: one allocation, one
    // memcpy pass, one submit + fence wait (instead of one blocking upload per array).
    totalBytes = 0;
    for (const Part& p : parts) totalBytes += p.size;
    Buffer staging{};
    try {
        staging = createBuffer(ctx, std::max<VkDeviceSize>(totalBytes, 4), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               MemoryUsage::Upload, "gaussians: staging");
        VkDeviceSize offset = 0;
        for (const Part& p : parts) {
            if (p.size) std::memcpy(static_cast<uint8_t*>(staging.mapped) + offset, p.data, p.size);
            offset += p.size;
        }
        VK_CHECK(vmaFlushAllocation(ctx.allocator, staging.allocation, 0, VK_WHOLE_SIZE));

        for (size_t b = 0; b < 4; ++b)
            if (parts[b].size)
                buffers[b] = createBuffer(ctx, parts[b].size,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          MemoryUsage::GpuOnly, parts[b].name.c_str());

        ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            VkDeviceSize srcOffset = 0;
            for (size_t b = 0; b < 4; ++b) {
                if (parts[b].size) {
                    const VkBufferCopy region{srcOffset, 0, parts[b].size};
                    vkCmdCopyBuffer(cmd, staging.buffer, buffers[b].buffer, 1, &region);
                }
                srcOffset += parts[b].size;
            }
            // A host fence wait only makes the copy visible to the HOST. For later GPU
            // reads on this queue, a barrier is the spec-guaranteed way: its destination
            // scope covers every command submitted afterwards on the same queue.
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        });
    } catch (...) {
        // E.g. VK_ERROR_OUT_OF_DEVICE_MEMORY for a very large scene: free what was
        // allocated so the caller can keep running with its previous scene.
        destroyBuffer(ctx, staging);
        destroy(ctx);
        throw;
    }
    destroyBuffer(ctx, staging);
    uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void GpuScene::destroy(VulkanContext& ctx) {
    for (Buffer& b : buffers) destroyBuffer(ctx, b);
    count = 0;
    totalBytes = 0;
}

void GpuScene::bind(DescriptorWriter& writer, uint32_t first) const {
    const VkBuffer fallback = buffers[0].buffer;
    auto pick = [&](size_t b) { return buffers[b].buffer ? buffers[b].buffer : fallback; };
    VkBuffer b1 = pick(1);
    if (layout == SceneLayout::Aos) b1 = buffers[0].buffer;  // geometry and colour live in the same records
    writer.buffer(first + 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pick(0))
        .buffer(first + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b1)
        .buffer(first + 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pick(2))
        .buffer(first + 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pick(3));
}

void GpuScene::addSpecialization(SpecializationConstants& spec) const {
    spec.set(kSpecGeometryEncoding, uint32_t(geometry))
        .set(kSpecColorEncoding, uint32_t(color))
        .set(kSpecGeometryStride, geometryStride)
        .set(kSpecColorStride, colorStride)
        .set(kSpecColorOffset, colorOffset)
        .set(kSpecStoredCoeffs, shCoeffsPerChannel)
        .set(kSpecCodebookStride, codebookStride)
        .set(kSpecVariableSh, uint32_t(variableSh));
}

}  // namespace gsr
