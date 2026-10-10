#include "core/descriptors.h"

namespace gsr {

DescriptorLayoutBuilder& DescriptorLayoutBuilder::add(uint32_t binding, VkDescriptorType type,
                                                      VkShaderStageFlags stages, uint32_t count) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = count;
    b.stageFlags = stages;
    m_bindings.push_back(b);
    return *this;
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device) const {
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<uint32_t>(m_bindings.size());
    info.pBindings = m_bindings.data();
    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
    return layout;
}

void DescriptorPool::create(VkDevice device, uint32_t maxSets, const std::vector<VkDescriptorPoolSize>& sizes) {
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.maxSets = maxSets;
    info.poolSizeCount = static_cast<uint32_t>(sizes.size());
    info.pPoolSizes = sizes.data();
    VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &handle));
}

void DescriptorPool::destroy(VkDevice device) {
    if (handle) vkDestroyDescriptorPool(device, handle, nullptr);  // frees all its sets
    handle = VK_NULL_HANDLE;
}

VkDescriptorSet DescriptorPool::allocate(VkDevice device, VkDescriptorSetLayout layout) const {
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = handle;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(device, &info, &set));
    return set;
}

DescriptorWriter& DescriptorWriter::image(uint32_t binding, VkDescriptorType type, VkImageView view,
                                          VkImageLayout layout, VkSampler sampler) {
    Entry e{};
    e.binding = binding;
    e.type = type;
    e.isImage = true;
    e.imageInfo = {sampler, view, layout};
    m_entries.push_back(e);
    return *this;
}

DescriptorWriter& DescriptorWriter::buffer(uint32_t binding, VkDescriptorType type, VkBuffer buf,
                                           VkDeviceSize offset, VkDeviceSize range) {
    Entry e{};
    e.binding = binding;
    e.type = type;
    e.isImage = false;
    e.bufferInfo = {buf, offset, range};
    m_entries.push_back(e);
    return *this;
}

void DescriptorWriter::update(VkDevice device, VkDescriptorSet set) {
    // Build the VkWriteDescriptorSet array only now, so the pointers into m_entries
    // stay valid (the vector no longer grows).
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(m_entries.size());
    for (const Entry& e : m_entries) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = e.binding;
        w.descriptorCount = 1;
        w.descriptorType = e.type;
        if (e.isImage)
            w.pImageInfo = &e.imageInfo;
        else
            w.pBufferInfo = &e.bufferInfo;
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_entries.clear();
}

}  // namespace gsr
