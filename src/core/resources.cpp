#include "core/resources.h"

#include "core/context.h"

#include <cstring>

namespace gsr {

namespace {
constexpr VkDeviceSize kDedicatedBufferBytes = 16ull * 1024 * 1024;
}  // namespace

Buffer createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    MemoryUsage memory, const char* debugName) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    switch (memory) {
        case MemoryUsage::GpuOnly:
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            // Large buffers (scene arrays, the tile list and its sort scratch) get their own
            // VkDeviceMemory. Otherwise VMA sub-allocates them from 256 MB blocks and keeps an
            // empty block alive after they are freed, so e.g. the adaptive tile list's shrink
            // would free VMA allocations but not process memory (measured, Phase 5).
            if (size >= kDedicatedBufferBytes) allocInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            break;
        case MemoryUsage::Upload:
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                              VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case MemoryUsage::Readback:
            allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                              VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    Buffer buffer;
    VmaAllocationInfo result{};
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &info, &allocInfo, &buffer.buffer, &buffer.allocation, &result));
    buffer.size = size;
    buffer.mapped = result.pMappedData;
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addrInfo.buffer = buffer.buffer;
        buffer.address = vkGetBufferDeviceAddress(ctx.device, &addrInfo);
    }
    if (debugName) ctx.setDebugName(buffer.buffer, VK_OBJECT_TYPE_BUFFER, debugName);

    MemoryRecord record;
    record.owner = ctx.memory.currentOwner(debugName);
    record.name = debugName ? debugName : "(unnamed buffer)";
    record.requested = size;
    record.allocated = result.size;
    record.usage = usage;
    record.memoryKind = memory == MemoryUsage::GpuOnly ? 0u : memory == MemoryUsage::Upload ? 1u : 2u;
    ctx.memory.add(reinterpret_cast<uint64_t>(buffer.buffer), std::move(record));
    return buffer;
}

void destroyBuffer(const VulkanContext& ctx, Buffer& buffer) {
    if (buffer.buffer) {
        ctx.memory.remove(reinterpret_cast<uint64_t>(buffer.buffer));
        vmaDestroyBuffer(ctx.allocator, buffer.buffer, buffer.allocation);
    }
    buffer = Buffer{};
}

Buffer createBufferWithData(VulkanContext& ctx, const void* data, VkDeviceSize size,
                            VkBufferUsageFlags usage, const char* debugName) {
    Buffer staging = createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload, "staging");
    std::memcpy(staging.mapped, data, size);
    // Mapped memory may be non-coherent; flushing is a no-op when it is coherent.
    VK_CHECK(vmaFlushAllocation(ctx.allocator, staging.allocation, 0, VK_WHOLE_SIZE));

    Buffer buffer = createBuffer(ctx, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 MemoryUsage::GpuOnly, debugName);
    ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy region{0, 0, size};
        vkCmdCopyBuffer(cmd, staging.buffer, buffer.buffer, 1, &region);
        // The fence wait inside immediateSubmit only makes the copy available to the
        // HOST; it is not a device-to-device memory dependency. This barrier is: its
        // second synchronization scope covers every command submitted later on this
        // queue, whatever stage or access (vertex fetch, index fetch, shader reads, ...)
        // first uses the buffer. Same pattern as GpuScene::upload.
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    });
    destroyBuffer(ctx, staging);
    return buffer;
}

Image createImage2D(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                    VkImageUsageFlags usage, const char* debugName) {
    Image image;
    image.format = format;
    image.extent = {width, height};
    const bool isDepth = format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ||
                         format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    image.aspect = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // Render targets get their own memory block; this is what drivers prefer.
    allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VmaAllocationInfo result{};
    VK_CHECK(vmaCreateImage(ctx.allocator, &info, &allocInfo, &image.image, &image.allocation, &result));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {image.aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(ctx.device, &viewInfo, nullptr, &image.view));

    if (debugName) ctx.setDebugName(image.image, VK_OBJECT_TYPE_IMAGE, debugName);

    MemoryRecord record;
    record.owner = ctx.memory.currentOwner(debugName);
    record.name = debugName ? debugName : "(unnamed image)";
    record.requested = result.size;  // images: the driver decides the size (tiling, padding)
    record.allocated = result.size;
    record.image = true;
    record.usage = usage;
    ctx.memory.add(reinterpret_cast<uint64_t>(image.image), std::move(record));
    return image;
}

void destroyImage(const VulkanContext& ctx, Image& image) {
    if (image.view) vkDestroyImageView(ctx.device, image.view, nullptr);
    if (image.image) {
        ctx.memory.remove(reinterpret_cast<uint64_t>(image.image));
        vmaDestroyImage(ctx.allocator, image.image, image.allocation);
    }
    image = Image{};
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

}  // namespace gsr
