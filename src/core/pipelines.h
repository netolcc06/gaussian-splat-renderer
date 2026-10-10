// Shader modules and pipeline creation (compute + graphics with dynamic rendering).
#pragma once

#include "core/vk_common.h"

#include <filesystem>
#include <string>
#include <vector>

namespace gsr {

// Finds a compiled shader (e.g. "cube.vert.spv"). Looks in <exe dir>/shaders first,
// then in the build-tree shader directory baked in at compile time.
std::filesystem::path findShader(const std::string& spvName);

// Loads a SPIR-V file and wraps it in a VkShaderModule (destroy it after pipeline creation).
VkShaderModule loadShaderModule(VkDevice device, const std::string& spvName);

VkPipelineLayout createPipelineLayout(VkDevice device,
                                      const std::vector<VkDescriptorSetLayout>& setLayouts,
                                      const std::vector<VkPushConstantRange>& pushConstants);

// Specialization constants: uint32 values by constant_id, collected into a
// VkSpecializationInfo (which points into this object: keep it alive while creating the
// pipeline). Phase 5 uses them to specialize the scene decoder on the scene format, so one
// SPIR-V file serves every format and the driver removes the unused decode paths.
class SpecializationConstants {
public:
    SpecializationConstants& set(uint32_t id, uint32_t value);
    const VkSpecializationInfo* info();  // nullptr if empty

private:
    std::vector<VkSpecializationMapEntry> m_entries;
    std::vector<uint32_t> m_data;
    VkSpecializationInfo m_info{};
};

// `specialization`: optional specialization constants. `stageFlags`: e.g.
// VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT for subgroup algorithms.
VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string& spvName,
                                 const VkSpecializationInfo* specialization = nullptr,
                                 VkPipelineShaderStageCreateFlags stageFlags = 0);

// Everything that varies between our graphics pipelines. Viewport and scissor are
// always dynamic state, so pipelines do not depend on the window size.
struct GraphicsPipelineDesc {
    std::string vertexShader;    // .spv file name
    std::string fragmentShader;  // .spv file name
    const VkSpecializationInfo* vertexSpecialization = nullptr;  // optional
    std::vector<VkVertexInputBindingDescription> vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depthTest = false;
    bool depthWrite = false;
    VkCompareOp depthCompare = VK_COMPARE_OP_LESS;
    bool alphaBlend = false;  // premultiplied "over": src + (1 - srcAlpha) * dst
    std::vector<VkFormat> colorFormats;  // dynamic rendering: formats instead of a render pass
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
};

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineLayout layout, const GraphicsPipelineDesc& desc);

// --- Phase 6: compiled-pipeline statistics (VK_KHR_pipeline_executable_properties) ---------
// While enabled, every pipeline is created with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
// (required to query them). Only for `--pipeline-stats` runs: it may change compile caching.
void setPipelineCaptureStatistics(bool enable);
struct PipelineStatistic {
    std::string name, description;
    std::string value;   // formatted
    double number = 0.0; // numeric value (bool: 0/1)
};
struct PipelineExecutableStats {
    std::string name, description;  // e.g. "Compute Shader"
    VkShaderStageFlags stages = 0;
    uint32_t subgroupSize = 0;
    std::vector<PipelineStatistic> statistics;
};
// Empty if the extension is not enabled on `device` (VulkanContext::pipelineExecutableInfo).
std::vector<PipelineExecutableStats> queryPipelineStatistics(VkDevice device, VkPipeline pipeline);

}  // namespace gsr
