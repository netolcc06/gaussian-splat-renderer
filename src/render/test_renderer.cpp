#include "render/test_renderer.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"

#include <array>
#include <vector>

namespace gsr {

namespace {

struct CubeVertex {
    glm::vec3 position;
    glm::vec3 color;
};

// A unit cube centered at the origin, 4 vertices per face so each face gets a flat
// color. For each face we pick axes u, v with cross(u, v) = outward normal, which
// makes (-u-v, +u-v, +u+v, -u+v) counter-clockwise when seen from outside.
void buildCube(std::vector<CubeVertex>& vertices, std::vector<uint16_t>& indices) {
    struct Face {
        glm::vec3 normal, u, v, color;
    };
    const std::array<Face, 6> faces = {{
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0.90f, 0.25f, 0.20f}},   // +X red
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0.20f, 0.80f, 0.85f}},   // -X cyan
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0.30f, 0.85f, 0.30f}},   // +Y green
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0.85f, 0.30f, 0.85f}},   // -Y magenta
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0.25f, 0.40f, 0.95f}},    // +Z blue
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0.95f, 0.85f, 0.25f}},  // -Z yellow
    }};
    const glm::vec2 corners[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (const Face& f : faces) {
        const auto base = static_cast<uint16_t>(vertices.size());
        for (const glm::vec2& c : corners) {
            // Slightly darken toward one corner so faces read as 3D in screenshots.
            const float shade = 0.8f + 0.1f * (c.x + c.y) * 0.5f + 0.1f;
            vertices.push_back({0.5f * (f.normal + c.x * f.u + c.y * f.v), f.color * shade});
        }
        for (int i : {0, 1, 2, 0, 2, 3}) indices.push_back(static_cast<uint16_t>(base + i));
    }
}

}  // namespace

void TestRenderer::init(VulkanContext& ctx, VkFormat colorFormat, VkFormat depthFormat) {
    VkDevice device = ctx.device;

    // --- Compute: gradient into a storage image -------------------------------
    m_gradientSetLayout = DescriptorLayoutBuilder()
                              .add(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
                              .build(device);
    m_descriptorPool.create(device, 1, {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}});
    m_gradientSet = m_descriptorPool.allocate(device, m_gradientSetLayout);

    const VkPushConstantRange gradientPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::ivec2)};
    m_gradientLayout = createPipelineLayout(device, {m_gradientSetLayout}, {gradientPush});
    m_gradientPipeline = createComputePipeline(device, m_gradientLayout, "gradient.comp.spv");

    // --- Graphics: cube --------------------------------------------------------
    const VkPushConstantRange cubePush{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
    m_cubeLayout = createPipelineLayout(device, {}, {cubePush});

    GraphicsPipelineDesc desc;
    desc.vertexShader = "cube.vert.spv";
    desc.fragmentShader = "cube.frag.spv";
    desc.vertexBindings = {{0, sizeof(CubeVertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    desc.vertexAttributes = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(CubeVertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(CubeVertex, color)},
    };
    // With the y-flipped projection (see Camera::projection) CCW triangles in world
    // space stay CCW on screen, so the usual OpenGL-style culling setup works.
    desc.cullMode = VK_CULL_MODE_BACK_BIT;
    desc.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    desc.depthTest = true;
    desc.depthWrite = true;
    desc.depthCompare = VK_COMPARE_OP_LESS;
    desc.colorFormats = {colorFormat};
    desc.depthFormat = depthFormat;
    m_cubePipeline = createGraphicsPipeline(device, m_cubeLayout, desc);

    std::vector<CubeVertex> vertices;
    std::vector<uint16_t> indices;
    buildCube(vertices, indices);
    m_indexCount = static_cast<uint32_t>(indices.size());
    m_vertexBuffer = createBufferWithData(ctx, vertices.data(), vertices.size() * sizeof(CubeVertex),
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "cube vertices");
    m_indexBuffer = createBufferWithData(ctx, indices.data(), indices.size() * sizeof(uint16_t),
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "cube indices");
}

void TestRenderer::onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& /*depth*/) {
    // The compute shader writes the color target directly, so its descriptor must
    // point at the current image view. (Callers wait for the GPU to be idle first.)
    DescriptorWriter()
        .image(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, color.view, VK_IMAGE_LAYOUT_GENERAL)
        .update(ctx.device, m_gradientSet);
}

RenderOutput TestRenderer::render(const FrameInput& in) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;

    // 1) Compute pass: gradient background ------------------------------------
    {
        auto timer = in.profiler.scope(cmd, "gradient (compute)");
        // Storage images must be in GENERAL layout. UNDEFINED as the old layout tells
        // the driver it may discard the previous contents.
        imageBarrier(cmd, in.color.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     in.colorBefore.stage, in.colorBefore.access,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gradientPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gradientLayout, 0, 1,
                                &m_gradientSet, 0, nullptr);
        const glm::ivec2 size(extent.width, extent.height);
        vkCmdPushConstants(cmd, m_gradientLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);
        // 16x16 threads per workgroup (see local_size in gradient.comp); round up.
        vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);
    }

    // 2) Graphics pass: cube on top -------------------------------------------
    {
        auto timer = in.profiler.scope(cmd, "cube (graphics)");
        // Compute wrote the image; the color attachment stage will read (loadOp LOAD)
        // and write it. This barrier is both the memory dependency and the layout change.
        imageBarrier(cmd, in.color.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        imageBarrier(cmd, in.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     in.depthBefore.stage, in.depthBefore.access,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = in.color.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // keep the gradient
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = in.depth.view;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;  // not needed after the pass
        depthAttachment.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &colorAttachment;
        rendering.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(cmd, &rendering);

        const VkViewport viewport{0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_cubePipeline);
        const glm::mat4 viewProj = in.proj * in.view;
        vkCmdPushConstants(cmd, m_cubeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(viewProj), &viewProj);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);

        vkCmdEndRendering(cmd);
    }

    RenderOutput out;
    out.color = {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    out.depth = {VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    return out;
}

void TestRenderer::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    destroyBuffer(ctx, m_vertexBuffer);
    destroyBuffer(ctx, m_indexBuffer);
    vkDestroyPipeline(device, m_cubePipeline, nullptr);
    vkDestroyPipelineLayout(device, m_cubeLayout, nullptr);
    vkDestroyPipeline(device, m_gradientPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_gradientLayout, nullptr);
    m_descriptorPool.destroy(device);
    vkDestroyDescriptorSetLayout(device, m_gradientSetLayout, nullptr);
}

}  // namespace gsr
