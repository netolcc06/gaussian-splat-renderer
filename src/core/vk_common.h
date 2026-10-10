// Common includes and error-checking helpers shared by all Vulkan code.
#pragma once

#include <vulkan/vulkan.h>

// VMA is a third-party header; silence its warnings so /W4 stays useful for our code.
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <stdexcept>
#include <string>

namespace gsr {

// Human-readable name of a VkResult (e.g. "VK_ERROR_DEVICE_LOST").
const char* vkResultName(VkResult result);

// Throws std::runtime_error with the failing expression, file and line.
[[noreturn]] void throwVkError(VkResult result, const char* expr, const char* file, int line);

}  // namespace gsr

// Every Vulkan call that returns a VkResult goes through VK_CHECK. Errors become
// exceptions that main() reports before exiting with a non-zero code.
#define VK_CHECK(expr)                                                   \
    do {                                                                 \
        VkResult vkCheckResult_ = (expr);                                \
        if (vkCheckResult_ != VK_SUCCESS)                                \
            ::gsr::throwVkError(vkCheckResult_, #expr, __FILE__, __LINE__); \
    } while (0)
