#include "core/pipelines.h"

#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gsr {

namespace {

VkPipelineCreateFlags g_captureFlags = 0;  // Phase 6 --pipeline-stats

std::filesystem::path executableDirectory() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    const DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(std::wstring(buffer, len)).parent_path();
#else
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

}  // namespace

std::filesystem::path findShader(const std::string& spvName) {
    const std::filesystem::path candidates[] = {
        executableDirectory() / "shaders" / spvName,
        std::filesystem::path(GSR_SHADER_DIR) / spvName,
    };
    for (const auto& path : candidates)
        if (std::filesystem::exists(path)) return path;
    throw std::runtime_error("Shader not found: " + spvName + " (looked next to the exe and in " +
                             GSR_SHADER_DIR + ")");
}

VkShaderModule loadShaderModule(VkDevice device, const std::string& spvName) {
    const std::filesystem::path path = findShader(spvName);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader " + path.string());
    const size_t size = static_cast<size_t>(file.tellg());
    if (size == 0 || size % 4 != 0) throw std::runtime_error("Invalid SPIR-V size: " + path.string());
    std::vector<uint32_t> code(size / 4);  // SPIR-V is a stream of 32-bit words
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode = code.data();
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkPipelineLayout createPipelineLayout(VkDevice device,
                                      const std::vector<VkDescriptorSetLayout>& setLayouts,
                                      const std::vector<VkPushConstantRange>& pushConstants) {
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    info.pSetLayouts = setLayouts.data();
    info.pushConstantRangeCount = static_cast<uint32_t>(pushConstants.size());
    info.pPushConstantRanges = pushConstants.data();
    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &info, nullptr, &layout));
    return layout;
}

SpecializationConstants& SpecializationConstants::set(uint32_t id, uint32_t value) {
    for (size_t i = 0; i < m_entries.size(); ++i)
        if (m_entries[i].constantID == id) {
            m_data[i] = value;
            return *this;
        }
    m_entries.push_back({id, uint32_t(m_data.size() * sizeof(uint32_t)), sizeof(uint32_t)});
    m_data.push_back(value);
    return *this;
}

const VkSpecializationInfo* SpecializationConstants::info() {
    if (m_entries.empty()) return nullptr;
    m_info.mapEntryCount = uint32_t(m_entries.size());
    m_info.pMapEntries = m_entries.data();
    m_info.dataSize = m_data.size() * sizeof(uint32_t);
    m_info.pData = m_data.data();
    return &m_info;
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string& spvName,
                                 const VkSpecializationInfo* specialization, VkPipelineShaderStageCreateFlags stageFlags) {
    VkShaderModule module = loadShaderModule(device, spvName);
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.flags = stageFlags;
    info.stage.pSpecializationInfo = specialization;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    info.flags = g_captureFlags;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, module, nullptr);  // the pipeline keeps its own copy
    return pipeline;
}

VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineLayout layout, const GraphicsPipelineDesc& desc) {
    VkShaderModule vert = loadShaderModule(device, desc.vertexShader);
    VkShaderModule frag = loadShaderModule(device, desc.fragmentShader);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[0].pSpecializationInfo = desc.vertexSpecialization;
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(desc.vertexBindings.size());
    vertexInput.pVertexBindingDescriptions = desc.vertexBindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(desc.vertexAttributes.size());
    vertexInput.pVertexAttributeDescriptions = desc.vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = desc.topology;

    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;  // actual values come from vkCmdSetViewport/Scissor
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.cullMode;
    raster.frontFace = desc.frontFace;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = desc.depthTest ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = desc.depthCompare;

    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(desc.colorFormats.size());
    for (auto& blend : blendAttachments) {
        blend = {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (desc.alphaBlend) {
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp = VK_BLEND_OP_ADD;
        }
    }
    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
    colorBlend.pAttachments = blendAttachments.data();

    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    // Dynamic rendering: instead of a VkRenderPass, the pipeline only declares the
    // formats of the attachments it will be used with.
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<uint32_t>(desc.colorFormats.size());
    rendering.pColorAttachmentFormats = desc.colorFormats.data();
    rendering.depthAttachmentFormat = desc.depthFormat;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = VK_NULL_HANDLE;
    info.flags = g_captureFlags;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
    return pipeline;
}

void setPipelineCaptureStatistics(bool enable) {
    g_captureFlags = enable ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
}

std::vector<PipelineExecutableStats> queryPipelineStatistics(VkDevice device, VkPipeline pipeline) {
    std::vector<PipelineExecutableStats> out;
    // Extension entry points are not exported by the loader library: fetch them from the device.
    auto getExecutables = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
    auto getStatistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
    if (!getExecutables || !getStatistics || !pipeline || g_captureFlags == 0) return out;

    VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pipelineInfo.pipeline = pipeline;
    uint32_t count = 0;
    VK_CHECK(getExecutables(device, &pipelineInfo, &count, nullptr));
    std::vector<VkPipelineExecutablePropertiesKHR> props(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    VK_CHECK(getExecutables(device, &pipelineInfo, &count, props.data()));
    for (uint32_t e = 0; e < count; ++e) {
        PipelineExecutableStats ex;
        ex.name = props[e].name;
        ex.description = props[e].description;
        ex.stages = props[e].stages;
        ex.subgroupSize = props[e].subgroupSize;
        VkPipelineExecutableInfoKHR execInfo{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        execInfo.pipeline = pipeline;
        execInfo.executableIndex = e;
        uint32_t n = 0;
        VK_CHECK(getStatistics(device, &execInfo, &n, nullptr));
        std::vector<VkPipelineExecutableStatisticKHR> stats(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        VK_CHECK(getStatistics(device, &execInfo, &n, stats.data()));
        for (const auto& s : stats) {
            PipelineStatistic st;
            st.name = s.name;
            st.description = s.description;
            switch (s.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    st.number = s.value.b32 ? 1.0 : 0.0;
                    st.value = s.value.b32 ? "true" : "false";
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    st.number = double(s.value.i64);
                    st.value = std::to_string(s.value.i64);
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    st.number = double(s.value.u64);
                    st.value = std::to_string(s.value.u64);
                    break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                    st.number = s.value.f64;
                    st.value = std::to_string(s.value.f64);
                    break;
                default: break;
            }
            ex.statistics.push_back(std::move(st));
        }
        out.push_back(std::move(ex));
    }
    return out;
}

}  // namespace gsr
