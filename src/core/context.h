// VulkanContext: the long-lived Vulkan objects every other part of the renderer needs.
//
//   VkInstance  -> VkDebugUtilsMessengerEXT (validation output)
//               -> VkSurfaceKHR (the window we present to)
//               -> VkPhysicalDevice (the GPU we picked)
//               -> VkDevice + one VkQueue (graphics + compute + present)
//               -> VmaAllocator (all buffer/image memory)
//
// Phase 0 uses a single queue for everything. Async compute (a second queue) is a
// later optimization and would be added here.
#pragma once

#include "core/memory_report.h"
#include "core/vk_common.h"

#include <atomic>
#include <functional>
#include <string>

struct GLFWwindow;

namespace gsr {

struct ValidationCounters {
    std::atomic<uint32_t> errors{0};
    std::atomic<uint32_t> warnings{0};
};

class VulkanContext {
public:
    // Creates everything listed above. `enableValidation` turns on the Khronos
    // validation layer and the debug messenger. `window == nullptr` creates a HEADLESS
    // context (no surface, no swapchain extension): compute-only tests use it.
    void init(GLFWwindow* window, bool enableValidation);
    void destroy();

    // Records commands with `record`, submits them and blocks until the GPU is done.
    // Only for setup work (uploads); never use it inside the frame loop.
    void immediateSubmit(const std::function<void(VkCommandBuffer)>& record);

    // Attaches a name to a Vulkan object so validation messages and tools
    // (RenderDoc, Nsight) show it. No-op when debug utils is unavailable.
    void setDebugName(VkObjectType type, uint64_t handle, const char* name) const;
    template <typename T>
    void setDebugName(T handle, VkObjectType type, const char* name) const {
        setDebugName(type, reinterpret_cast<uint64_t>(handle), name);
    }

    // Counts of validation warnings/errors reported so far (process-wide).
    static ValidationCounters& validationCounters();

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    // Subgroup (warp/wave) properties, for compute shaders that use subgroup operations
    // (Phase 3: stream compaction and the radix sort). minSize/maxSize bound the size a
    // pipeline may actually get (it may vary per pipeline, e.g. 8..32 on Intel).
    struct SubgroupInfo {
        uint32_t size = 0, minSize = 0, maxSize = 0;
        VkShaderStageFlags stages = 0;
        VkSubgroupFeatureFlags operations = 0;
    } subgroup;

    bool headless = false;
    bool validationEnabled = false;
    bool debugUtilsEnabled = false;
    bool memoryBudgetEnabled = false;
    // Phase 6: VK_KHR_pipeline_executable_properties (register/shared-memory statistics of
    // compiled pipelines, `--pipeline-stats`) and shaderFloat16 (render-loop fp16 experiment).
    bool pipelineExecutableInfo = false;
    bool shaderFloat16 = false;
    std::string deviceName;
    std::string driverVersion;

    // Every buffer/image created through core/resources.h (Phase 5 memory report).
    mutable MemoryRegistry memory;

private:
    void createInstance(bool enableValidation);
    void pickPhysicalDevice();
    void createDevice();
    void createAllocator();

    VkCommandPool m_uploadPool = VK_NULL_HANDLE;
};

}  // namespace gsr
