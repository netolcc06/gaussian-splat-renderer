// Descriptor helpers: a set-layout builder, a fixed-size pool and a writer.
//
// Descriptor sets are how shaders see resources (images, buffers). The model is:
//   VkDescriptorSetLayout  - "binding 0 is a storage image visible to compute"
//   VkDescriptorPool       - memory the sets are allocated from
//   VkDescriptorSet        - an instance of a layout, pointed at concrete resources
#pragma once

#include "core/vk_common.h"

#include <vector>

namespace gsr {

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder& add(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stages,
                                 uint32_t count = 1);
    VkDescriptorSetLayout build(VkDevice device) const;

private:
    std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

class DescriptorPool {
public:
    // `sizes` = how many descriptors of each type the pool holds in total.
    void create(VkDevice device, uint32_t maxSets, const std::vector<VkDescriptorPoolSize>& sizes);
    void destroy(VkDevice device);
    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout) const;

    VkDescriptorPool handle = VK_NULL_HANDLE;
};

// Collects descriptor writes and applies them in one vkUpdateDescriptorSets call.
class DescriptorWriter {
public:
    DescriptorWriter& image(uint32_t binding, VkDescriptorType type, VkImageView view,
                            VkImageLayout layout, VkSampler sampler = VK_NULL_HANDLE);
    DescriptorWriter& buffer(uint32_t binding, VkDescriptorType type, VkBuffer buffer,
                             VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    void update(VkDevice device, VkDescriptorSet set);

private:
    struct Entry {
        uint32_t binding;
        VkDescriptorType type;
        bool isImage;
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
    };
    std::vector<Entry> m_entries;
};

}  // namespace gsr
