// The single translation unit that compiles the VMA implementation.
// We link vulkan-1.lib directly, so VMA can use the statically linked entry points.
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION

#include <vulkan/vulkan.h>
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
