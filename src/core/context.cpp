#include "core/context.h"

#include <GLFW/glfw3.h>
#include <vulkan/vk_enum_string_helper.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace gsr {

const char* vkResultName(VkResult result) { return string_VkResult(result); }

void throwVkError(VkResult result, const char* expr, const char* file, int line) {
    throw std::runtime_error(std::string("Vulkan call failed: ") + expr + " -> " +
                             vkResultName(result) + " (" + file + ":" + std::to_string(line) + ")");
}

ValidationCounters& VulkanContext::validationCounters() {
    static ValidationCounters counters;
    return counters;
}

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// Called by the validation layer (and the loader) for every message that matches the
// severity/type mask chosen in createInstance(). We only subscribe to warnings and
// errors, count them, and print them to stderr so automated runs can grep for them.
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*userData*/) {
    const bool isError = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
    auto& counters = VulkanContext::validationCounters();
    (isError ? counters.errors : counters.warnings)++;
    std::fprintf(stderr, "[vulkan][%s] %s\n", isError ? "ERROR" : "WARNING",
                 data->pMessage ? data->pMessage : "(no message)");
    std::fflush(stderr);
    return VK_FALSE;  // never abort the call that triggered the message
}

VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    return info;
}

bool hasLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers)
        if (std::strcmp(layer.layerName, name) == 0) return true;
    return false;
}

bool hasInstanceExtension(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (const auto& ext : exts)
        if (std::strcmp(ext.extensionName, name) == 0) return true;
    return false;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice gpu) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, exts.data());
    return exts;
}

bool hasDeviceExtension(VkPhysicalDevice gpu, const char* name) {
    for (const auto& ext : deviceExtensions(gpu))
        if (std::strcmp(ext.extensionName, name) == 0) return true;
    return false;
}

constexpr uint32_t kVendorNvidia = 0x10DE;

}  // namespace

void VulkanContext::init(GLFWwindow* window, bool enableValidation) {
    headless = window == nullptr;
    createInstance(enableValidation);
    if (!headless) VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    pickPhysicalDevice();
    createDevice();
    createAllocator();

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &m_uploadPool));
}

void VulkanContext::createInstance(bool enableValidation) {
    uint32_t loaderVersion = 0;
    vkEnumerateInstanceVersion(&loaderVersion);
    if (loaderVersion < VK_API_VERSION_1_3)
        throw std::runtime_error("The Vulkan loader does not support Vulkan 1.3");

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Gaussian Splat Renderer";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "gsr";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // GLFW tells us which surface extensions this platform needs (Win32 here).
    // Headless contexts (tests) need none.
    std::vector<const char*> extensions;
    if (!headless) {
        uint32_t glfwCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
        if (!glfwExts) throw std::runtime_error("GLFW: Vulkan surface extensions unavailable");
        extensions.assign(glfwExts, glfwExts + glfwCount);
    }

    std::vector<const char*> layers;
    if (enableValidation) {
        if (!hasLayer(kValidationLayer))
            throw std::runtime_error(
                "Validation requested but VK_LAYER_KHRONOS_validation was not found "
                "(is the Vulkan SDK installed / VK_LAYER_PATH set?)");
        layers.push_back(kValidationLayer);
    }
    // Debug utils gives us object names and the messenger. Use it whenever available.
    debugUtilsEnabled = hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtilsEnabled) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &appInfo;
    info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.data();
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    // Chaining a messenger create-info into the instance create-info also reports
    // problems that happen inside vkCreateInstance/vkDestroyInstance themselves.
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = messengerCreateInfo();
    if (enableValidation && debugUtilsEnabled) info.pNext = &messengerInfo;

    VK_CHECK(vkCreateInstance(&info, nullptr, &instance));
    validationEnabled = enableValidation;

    if (enableValidation && debugUtilsEnabled) {
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        VK_CHECK(createMessenger(instance, &messengerInfo, nullptr, &debugMessenger));
    }
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> gpus(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, gpus.data()));

    // Score every GPU that meets our hard requirements; the highest score wins.
    // Discrete GPUs beat integrated ones, and NVIDIA gets a small bonus because this
    // project targets an RTX laptop with an Intel iGPU next to it.
    int bestScore = -1;
    for (VkPhysicalDevice gpu : gpus) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        if (!headless && !hasDeviceExtension(gpu, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;

        // Required features (all Vulkan 1.2/1.3 core, but still optional to support).
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f12.pNext = &f13;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(gpu, &f2);
        if (!f13.dynamicRendering || !f13.synchronization2 || !f13.shaderDemoteToHelperInvocation ||
            !f13.computeFullSubgroups || !f13.maintenance4 ||
            !f12.timelineSemaphore ||
            !f12.bufferDeviceAddress || !f12.hostQueryReset)
            continue;

        // We need one queue family that can do graphics, compute and present.
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, families.data());
        int family = -1;
        for (uint32_t i = 0; i < familyCount; ++i) {
            VkBool32 present = headless ? VK_TRUE : VK_FALSE;  // headless: nothing to present to
            if (!headless) vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &present);
            const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if ((families[i].queueFlags & need) == need && present &&
                families[i].timestampValidBits > 0) {
                family = static_cast<int>(i);
                break;
            }
        }
        if (family < 0) continue;

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;
        if (props.vendorID == kVendorNvidia) score += 50;
        if (score > bestScore) {
            bestScore = score;
            physicalDevice = gpu;
            queueFamily = static_cast<uint32_t>(family);
        }
    }
    if (!physicalDevice) throw std::runtime_error("No suitable Vulkan 1.3 GPU found");

    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    deviceName = properties.deviceName;

    VkPhysicalDeviceDriverProperties driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties props11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties props13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    driverProps.pNext = &props11;
    props11.pNext = &props13;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &driverProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    driverVersion = std::string(driverProps.driverName) + " " + driverProps.driverInfo;
    subgroup.size = props11.subgroupSize;
    subgroup.minSize = props13.minSubgroupSize;
    subgroup.maxSize = props13.maxSubgroupSize;
    subgroup.stages = props11.subgroupSupportedStages;
    subgroup.operations = props11.subgroupSupportedOperations;
}

void VulkanContext::createDevice() {
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    std::vector<const char*> extensions;
    if (!headless) extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    // Optional: lets VMA report real per-heap budgets (used in benchmark output).
    memoryBudgetEnabled = hasDeviceExtension(physicalDevice, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (memoryBudgetEnabled) extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    // Optional (Phase 6): statistics of compiled pipelines (registers, shared memory, spills).
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR execFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    {
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        query.pNext = &supported12;
        if (hasDeviceExtension(physicalDevice, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
            supported12.pNext = &execFeatures;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &query);
        supported12.pNext = nullptr;
    }
    pipelineExecutableInfo = execFeatures.pipelineExecutableInfo == VK_TRUE;
    if (pipelineExecutableInfo) extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    shaderFloat16 = supported12.shaderFloat16 == VK_TRUE;

    // Features are enabled by chaining the per-version feature structs.
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;   // no VkRenderPass / VkFramebuffer objects
    f13.synchronization2 = VK_TRUE;   // vkCmdPipelineBarrier2, vkQueueSubmit2
    // glslc targeting Vulkan 1.3 compiles GLSL `discard` to OpDemoteToHelperInvocation,
    // which needs this (mandatory in 1.3) feature enabled explicitly.
    f13.shaderDemoteToHelperInvocation = VK_TRUE;
    // Subgroup algorithms (Phase 3 compaction + radix sort) assume every subgroup of a
    // workgroup is full; pipelines request it with REQUIRE_FULL_SUBGROUPS.
    f13.computeFullSubgroups = VK_TRUE;
    // Phase 6: the tile render pass takes its workgroup size from a specialization constant
    // (local_size_x_id: SPIR-V LocalSizeId), which needs maintenance4 (mandatory in 1.3).
    f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f13;
    f12.timelineSemaphore = VK_TRUE;    // frame pacing with a single counter
    f12.bufferDeviceAddress = VK_TRUE;  // GPU pointers to buffers (later phases)
    f12.hostQueryReset = VK_TRUE;       // reset timestamp queries from the CPU
    f12.shaderFloat16 = shaderFloat16 ? VK_TRUE : VK_FALSE;  // Phase 6 fp16 render experiment
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exec{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    exec.pipelineExecutableInfo = VK_TRUE;
    if (pipelineExecutableInfo) f13.pNext = &exec;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f12;

    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.pNext = &features;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queueInfo;
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(physicalDevice, &info, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    setDebugName(queue, VK_OBJECT_TYPE_QUEUE, "main queue");
}

void VulkanContext::createAllocator() {
    VmaAllocatorCreateInfo info{};
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (memoryBudgetEnabled) info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    info.instance = instance;
    info.physicalDevice = physicalDevice;
    info.device = device;
    VK_CHECK(vmaCreateAllocator(&info, &allocator));
}

void VulkanContext::immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = m_uploadPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;

    // A fence is the simplest "wait for this one submission" primitive.
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    VK_CHECK(vkQueueSubmit2(queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, m_uploadPool, 1, &cmd);
}

void VulkanContext::setDebugName(VkObjectType type, uint64_t handle, const char* name) const {
    if (!debugUtilsEnabled || !device) return;
    static PFN_vkSetDebugUtilsObjectNameEXT setName = nullptr;
    if (!setName)
        setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (!setName) return;
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    setName(device, &info);
}

void VulkanContext::destroy() {
    if (device) {
        vkDeviceWaitIdle(device);
        if (m_uploadPool) vkDestroyCommandPool(device, m_uploadPool, nullptr);
        if (allocator) vmaDestroyAllocator(allocator);
        vkDestroyDevice(device, nullptr);
    }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (debugMessenger) {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(instance, debugMessenger, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
    *this = VulkanContext{};
}

}  // namespace gsr
