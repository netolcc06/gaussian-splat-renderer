// SplatGpuSortRenderer (Phase 3, `--renderer splat-gpusort`): the Phase 2 splat renderer
// with the CPU taken out of the per-frame critical path. Per frame, all on the GPU:
//
//   0. RESET      vkCmdUpdateBuffer/vkCmdFillBuffer: draw args {6, 0, 0, 0} and the
//                 per-reason cull counters of the control block.
//   1. PREPROCESS + CULL + COMPACT (splat_preprocess_compact.comp.spv)
//                 the Phase 2 preprocess, plus an opacity cull; every visible gaussian
//                 is appended as (float bits of its view depth, index) to a compact list
//                 with ONE atomicAdd per subgroup on the draw's instanceCount.
//   2. SORT       GpuRadixSort on exactly the visible entries (count read on the GPU,
//                 vkCmdDispatchIndirect): stable LSD radix sort, nearest first.
//   3. DRAW       vkCmdDrawIndirect(control): instanceCount = visible count as written by
//                 the GPU; splat_indirect.vert reads the same count to walk the
//                 nearest-first list backwards (farthest drawn first).
//
// The CPU only records commands: no sort, no upload, no readback on the critical path
// (the cull statistics are copied to a per-frame-slot host buffer and read two frames
// later, never waited for).
//
// Sort frequency (GpuSortSettings): by default every frame sorts. Optionally the sorted
// list is REUSED for several frames (fixed interval and/or until the camera moved by more
// than a threshold): then steps 0 (draw args) and 2 are skipped, the preprocess still
// runs (positions/colours follow the camera) and writes every splat including culled ones,
// and the draw uses the old list and count. See docs/notes/phase3.md for the trade-off.
#pragma once

#include "core/descriptors.h"
#include "render/gpu_radix_sort.h"
#include "render/renderer.h"
#include "render/splat_draw.h"
#include "splat/splat_math.h"

#include <array>
#include <vector>

namespace gsr {

struct GpuScene;
struct GaussianScene;

struct GpuSortSettings {
    // Re-sort at least every `sortEvery` frames (1 = every frame, the default and the
    // reference behaviour; 0 = no forced re-sort, only the threshold below).
    uint32_t sortEvery = 1;
    // > 0: also re-sort as soon as the camera rotated, or moved with a parallax (as seen
    // from the scene centre) of, more than this many degrees since the last sort.
    float sortThresholdDeg = 0.0f;
    // Radix sort scatter implementation (identical output; see GpuRadixSort::Variant).
    GpuRadixSort::Variant radixVariant = GpuRadixSort::Variant::Classic;
};

// The GPU control block (C++ twin of `Control` in splat_preprocess.comp, 48 bytes).
struct SplatControl {
    VkDrawIndirectCommand draw;  // instanceCount = visible count = compact-list length
    uint32_t cullCounts[8];      // [CullReason] this frame; [0] = visible
};
static_assert(sizeof(SplatControl) == 48, "layout must match splat_preprocess.comp");
const char* cullReasonName(uint32_t reason);  // "visible", "near plane", ...

class SplatGpuSortRenderer final : public Renderer {
public:
    void init(VulkanContext& ctx, VkFormat colorFormat, const GpuScene& gpuScene, const GaussianScene& cpuScene,
              uint32_t framesInFlight, const SplatSettings& settings, const GpuSortSettings& sortSettings);
    const char* name() const override { return "splat-gpusort"; }
    void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) override;
    RenderOutput render(const FrameInput& in) override;
    void drawUi() override;
    std::vector<std::pair<std::string, VkPipeline>> pipelines() const override {
        auto out = m_draw.pipelines();
        out.insert(out.begin(), {"splat_preprocess_compact.comp.spv", m_preprocessPipeline});
        for (auto& p : m_sort.pipelines("depth sort")) out.push_back(p);
        return out;
    }
    void destroy(VulkanContext& ctx) override;

    SplatSettings settings() const {
        SplatSettings s = m_settings;
        s.format = m_draw.format();
        return s;
    }
    const GpuSortSettings& sortSettings() const { return m_sortSettings; }
    int effectiveShDegree() const;

    // Statistics of the most recent frame whose readback is complete (2 frames old).
    struct FrameStats {
        bool valid = false;
        uint64_t frameIndex = 0;
        bool sorted = false;      // that frame ran the sort (else it reused an older list)
        SplatControl control{};   // visible count (draw.instanceCount) and cull counters
    };
    const FrameStats& lastStats() const { return m_lastStats; }
    uint64_t framesRendered() const { return m_framesRendered; }
    uint64_t sortsPerformed() const { return m_sortsPerformed; }

    // Testing/debugging (GPU must be idle): copies the sorted compact list and the
    // control block back to the CPU.
    void readBack(std::vector<uint32_t>& keys, std::vector<uint32_t>& values, SplatControl& control);

private:
    bool shouldSort(const SplatCamera& cam, VkExtent2D extent) const;

    VulkanContext* m_ctx = nullptr;
    const GpuScene* m_gpuScene = nullptr;
    SplatSettings m_settings;
    GpuSortSettings m_sortSettings;
    glm::vec3 m_sceneCentre{0.0f};  // data space, for the translation threshold

    // Buffers.
    Buffer m_splats;      // count x 48 bytes (struct Splat), indexed by gaussian index
    Buffer m_sortKeys;    // compact list: depth keys (sorted in place)
    Buffer m_sortValues;  // compact list: gaussian indices (sorted in place)
    Buffer m_control;     // SplatControl: indirect draw args + cull counters
    std::vector<Buffer> m_statsReadback;  // one host-visible SplatControl per frame slot
    struct SlotInfo {
        bool used = false;
        bool sorted = false;
        uint64_t frameIndex = 0;
    };
    std::vector<SlotInfo> m_slotInfo;

    // Passes.
    VkDescriptorSetLayout m_preprocessSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_preprocessSet = VK_NULL_HANDLE;
    VkPipelineLayout m_preprocessLayout = VK_NULL_HANDLE;
    VkPipeline m_preprocessPipeline = VK_NULL_HANDLE;
    GpuRadixSort m_sort;
    VkDescriptorSetLayout m_drawSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_drawSet = VK_NULL_HANDLE;
    VkPipelineLayout m_drawLayout = VK_NULL_HANDLE;
    SplatDrawStage m_draw;
    DescriptorPool m_descriptorPool;

    // Sort-frequency state.
    bool m_haveSortedList = false;
    uint64_t m_lastSortFrame = 0;
    glm::mat4 m_lastSortView{1.0f};
    glm::vec3 m_lastSortPos{0.0f};
    VkExtent2D m_lastSortExtent{};
    uint64_t m_framesRendered = 0;
    uint64_t m_sortsPerformed = 0;

    FrameStats m_lastStats;
};

}  // namespace gsr
