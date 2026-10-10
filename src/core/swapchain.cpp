#include "core/swapchain.h"

#include "core/context.h"

#include <algorithm>

namespace gsr {

namespace {

VkSurfaceFormatKHR chooseSurfaceFormat(VkPhysicalDevice gpu, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, formats.data());

    // We want a UNORM (not _SRGB) format: the renderer works with display-referred
    // colors (like the 3DGS reference implementation, which blends in sRGB space),
    // so values must reach the screen unchanged, without an extra linear->sRGB encode.
    for (VkFormat wanted : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM})
        for (const auto& f : formats)
            if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
    return formats.front();
}

VkPresentModeKHR choosePresentMode(VkPhysicalDevice gpu, VkSurfaceKHR surface, bool vsync) {
    if (vsync) return VK_PRESENT_MODE_FIFO_KHR;  // always supported
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, modes.data());
    for (VkPresentModeKHR wanted : {VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR})
        if (std::find(modes.begin(), modes.end(), wanted) != modes.end()) return wanted;
    return VK_PRESENT_MODE_FIFO_KHR;
}

}  // namespace

void Swapchain::create(const VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync) {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice, ctx.surface, &caps));

    const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(ctx.physicalDevice, ctx.surface);
    presentMode = choosePresentMode(ctx.physicalDevice, ctx.surface, vsync);

    // currentExtent == 0xFFFFFFFF means "the window will match whatever you pick".
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    // One more image than the minimum so we never wait on the driver to release one.
    minImageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) minImageCount = std::min(minImageCount, caps.maxImageCount);

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((caps.supportedUsageFlags & usage) != usage)
        throw std::runtime_error("Swapchain does not support COLOR_ATTACHMENT|TRANSFER_DST usage");

    // We copy our render target into the swapchain with vkCmdBlitImage, which needs
    // BLIT_DST support for the chosen format.
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, surfaceFormat.format, &formatProps);
    if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
        throw std::runtime_error("Swapchain format does not support BLIT_DST");

    VkSwapchainKHR oldSwapchain = handle;
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = ctx.surface;
    info.minImageCount = minImageCount;
    info.imageFormat = surfaceFormat.format;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = usage;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  // single queue family
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = presentMode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = oldSwapchain;  // lets the driver recycle resources on resize
    VK_CHECK(vkCreateSwapchainKHR(ctx.device, &info, nullptr, &handle));

    // The old swapchain (if any) is retired by the create call above; destroy it now.
    destroyImageResources(ctx);
    if (oldSwapchain) vkDestroySwapchainKHR(ctx.device, oldSwapchain, nullptr);

    format = surfaceFormat.format;
    colorSpace = surfaceFormat.colorSpace;

    uint32_t imageCount = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(ctx.device, handle, &imageCount, nullptr));
    images.resize(imageCount);
    VK_CHECK(vkGetSwapchainImagesKHR(ctx.device, handle, &imageCount, images.data()));

    views.resize(imageCount);
    renderFinished.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx.device, &viewInfo, nullptr, &views[i]));

        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(ctx.device, &semInfo, nullptr, &renderFinished[i]));
        ctx.setDebugName(images[i], VK_OBJECT_TYPE_IMAGE, "swapchain image");
        ctx.setDebugName(renderFinished[i], VK_OBJECT_TYPE_SEMAPHORE, "render finished");
    }
}

void Swapchain::destroyImageResources(const VulkanContext& ctx) {
    for (VkImageView view : views) vkDestroyImageView(ctx.device, view, nullptr);
    for (VkSemaphore sem : renderFinished) vkDestroySemaphore(ctx.device, sem, nullptr);
    views.clear();
    renderFinished.clear();
    images.clear();
}

void Swapchain::destroy(const VulkanContext& ctx) {
    destroyImageResources(ctx);
    if (handle) vkDestroySwapchainKHR(ctx.device, handle, nullptr);
    handle = VK_NULL_HANDLE;
}

}  // namespace gsr
