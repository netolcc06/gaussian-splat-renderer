// Swapchain: the queue of window-sized images we render into and hand to the OS.
//
// Images are only ever written by a blit (copy from our offscreen render target)
// and by the ImGui pass, so they need TRANSFER_DST and COLOR_ATTACHMENT usage.
#pragma once

#include "core/vk_common.h"

#include <vector>

namespace gsr {

class VulkanContext;

class Swapchain {
public:
    // Creates (or re-creates, reusing the old handle) the swapchain for the given
    // framebuffer size. `vsync` picks FIFO; otherwise IMMEDIATE/MAILBOX if available.
    // The caller must make sure the GPU no longer uses the old images (vkDeviceWaitIdle).
    void create(const VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync);
    void destroy(const VulkanContext& ctx);

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent{};
    uint32_t minImageCount = 0;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    // One "render finished" semaphore per swapchain image (not per frame in flight):
    // the presentation engine may still hold the semaphore of an image until that
    // same image is acquired again, so it can only be reused for that image.
    std::vector<VkSemaphore> renderFinished;

private:
    void destroyImageResources(const VulkanContext& ctx);
};

}  // namespace gsr
