// SplatRenderer (Phase 2): the baseline Gaussian Splatting renderer.
//
// Three stages per frame, deliberately kept separate (later phases replace them one by one):
//
//   1. SORT (CPU)        DepthSorter: view depth of every gaussian from the CPU copy of the
//                        positions, near cull, LSD radix sort -> indices nearest first.
//                        memcpy into this frame slot's host-visible staging buffer.
//                        [Phase 3 replaces this with a GPU radix sort + culling.]
//   2. PREPROCESS (GPU)  copy staging -> device-local index buffer, then a compute pass,
//                        one thread per gaussian: the INRIA preprocess (EWA projection,
//                        conic, radius, SH colour) -> one 48-byte Splat per gaussian.
//   3. DRAW (GPU)        vkCmdDraw(6, visible): instanced screen-space quads read back to
//                        front through the sorted indices; the fragment shader evaluates
//                        the gaussian and outputs premultiplied colour; fixed-function
//                        "over" blending composites. [Phase 4 adds a compute tile
//                        rasterizer with front-to-back early termination instead.]
//
// The draw stage (and the splat target formats) live in render/splat_draw.h, shared with
// the Phase 3 GPU-sort renderer (render/splat_gpusort_renderer.h).
#pragma once

#include "core/descriptors.h"
#include "render/renderer.h"
#include "render/splat_draw.h"
#include "splat/depth_sort.h"
#include "splat/splat_math.h"

#include <vector>

namespace gsr {

struct GpuScene;
struct GaussianScene;

class SplatRenderer final : public Renderer {
public:
    // `colorFormat` is the app's colour target format (RGBA8). `cpuScene` must outlive the
    // renderer: the CPU sort reads its positions every frame.
    void init(VulkanContext& ctx, VkFormat colorFormat, const GpuScene& gpuScene, const GaussianScene& cpuScene,
              uint32_t framesInFlight, const SplatSettings& settings);
    const char* name() const override { return "splat"; }
    void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) override;
    RenderOutput render(const FrameInput& in) override;
    void drawUi() override;
    std::vector<std::pair<std::string, VkPipeline>> pipelines() const override {
        auto out = m_draw.pipelines();
        out.insert(out.begin(), {"splat_preprocess.comp.spv", m_preprocessPipeline});
        return out;
    }
    void destroy(VulkanContext& ctx) override;

    SplatSettings settings() const {
        SplatSettings s = m_settings;
        s.format = m_draw.format();  // the UI may have changed it
        return s;
    }
    int effectiveShDegree() const;
    uint32_t lastVisibleCount() const { return m_lastVisible; }

private:
    VulkanContext* m_ctx = nullptr;
    const GpuScene* m_gpuScene = nullptr;
    const GaussianScene* m_cpuScene = nullptr;
    SplatSettings m_settings;

    // Stage 1: CPU sort + upload.
    DepthSorter m_sorter;
    std::vector<Buffer> m_staging;  // one host-visible index buffer per frame in flight
    Buffer m_indices;               // device-local copy the vertex shader reads
    uint32_t m_lastVisible = 0;

    // Stage 2: preprocess.
    Buffer m_splats;  // count x 48 bytes (struct Splat)
    VkDescriptorSetLayout m_preprocessSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_preprocessSet = VK_NULL_HANDLE;
    VkPipelineLayout m_preprocessLayout = VK_NULL_HANDLE;
    VkPipeline m_preprocessPipeline = VK_NULL_HANDLE;

    // Stage 3: draw (shared implementation, see splat_draw.h).
    VkDescriptorSetLayout m_drawSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_drawSet = VK_NULL_HANDLE;
    VkPipelineLayout m_drawLayout = VK_NULL_HANDLE;
    SplatDrawStage m_draw;
    DescriptorPool m_descriptorPool;
};

}  // namespace gsr
