// Buffers, images and barriers: thin, explicit wrappers around Vulkan + VMA.
#pragma once

#include "core/vk_common.h"

namespace gsr {

class VulkanContext;

// Where a buffer's memory should live. VMA picks the actual memory type.
enum class MemoryUsage {
    GpuOnly,   // device-local; written by the GPU or by a staging upload
    Upload,    // host-visible, persistently mapped, written sequentially by the CPU
    Readback,  // host-visible, persistently mapped, read by the CPU (cached)
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;        // non-null for Upload/Readback buffers
    VkDeviceAddress address = 0;   // non-zero if created with SHADER_DEVICE_ADDRESS usage
};

Buffer createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    MemoryUsage memory, const char* debugName = nullptr);
void destroyBuffer(const VulkanContext& ctx, Buffer& buffer);

// Creates a GpuOnly buffer and fills it through a temporary staging buffer
// (blocking upload via VulkanContext::immediateSubmit). TRANSFER_DST is added to `usage`.
// The upload ends with a COPY -> ALL_COMMANDS/MEMORY_READ barrier, so any later command
// on the queue may read the buffer without further synchronization.
Buffer createBufferWithData(VulkanContext& ctx, const void* data, VkDeviceSize size,
                            VkBufferUsageFlags usage, const char* debugName = nullptr);

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

// A single-mip, single-layer, device-local 2D image plus a view of it.
Image createImage2D(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                    VkImageUsageFlags usage, const char* debugName = nullptr);
void destroyImage(const VulkanContext& ctx, Image& image);

// One synchronization2 image barrier (also performs the layout transition).
// Kept verbose on purpose: every barrier in the renderer names both sides explicitly.
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

// A global memory barrier (no layout change), e.g. a buffer written by compute and
// read by a later draw.
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

}  // namespace gsr
