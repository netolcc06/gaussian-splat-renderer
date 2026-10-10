// SplatDrawStage: the "draw" half shared by the splat renderers (Phase 2 `splat`,
// Phase 3 `splat-gpusort`): instanced screen-space quads, premultiplied "over" blending
// back to front into the splat target, then (float targets) a blit into the app's RGBA8
// colour target. The renderers differ only in where the sorted indices and the instance
// count come from, so they pass in the pipeline layout, the vertex shader variant and a
// callback that binds their descriptors and issues the draw (direct or indirect).
//
// The blend target ("splat target") has a selectable format: RGBA8 draws straight into
// the app's RGBA8 colour target; RGBA16F/RGBA32F draw into an intermediate float image
// that is then blitted (converted) into the colour target. See docs/notes/phase2.md for
// why the float targets matter.
#pragma once

#include "render/renderer.h"
#include "splat/splat_math.h"
#include "splat/tile_coverage.h"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace gsr {

enum class SplatTargetFormat : uint32_t { Rgba8 = 0, Rgba16f = 1, Rgba32f = 2 };
const char* splatTargetFormatName(SplatTargetFormat f);
std::optional<SplatTargetFormat> parseSplatTargetFormat(std::string_view text);
std::optional<CoverageBound> parseCoverageBound(std::string_view text);

struct SplatSettings {
    SplatTargetFormat format = SplatTargetFormat::Rgba16f;
    CoverageBound bound = CoverageBound::Tile;
    int maxShDegree = 3;  // evaluated bands = min(scene degree, this)
    // Phase 6 (`--bounds`): raster quads / tile binning clipped to the alpha >= 1/255 ellipse.
    TileBounds bounds = TileBounds::Rect;
};

class SplatDrawStage {
public:
    // Builds one pipeline per supported target format (blending depends on the format)
    // with `layout` and `vertexShader` (+ splat.frag). Throws if `format` is unsupported.
    void init(VulkanContext& ctx, VkFormat colorFormat, SplatTargetFormat format, VkPipelineLayout layout,
              const char* vertexShader);
    void destroy(VulkanContext& ctx);

    // The intermediate target follows the colour target's size (GPU idle when called).
    void onRenderTargetsChanged(VkExtent2D extent);
    // Applies a format change requested from the UI (waits for the GPU if it changed).
    void applyRequestedFormat(VkExtent2D extent);

    // Records: target barrier, begin rendering (clear to black), bind the pipeline,
    // `issueDraw(cmd)` (binds descriptors/push constants, draws), end rendering, and the
    // resolve blit for float targets. GPU scopes "draw splats (graphics)" and
    // "resolve (blit to rgba8)". Returns the final state of the colour target.
    RenderOutput record(const FrameInput& in, const std::function<void(VkCommandBuffer)>& issueDraw);

    SplatTargetFormat format() const { return m_format; }
    // UI: "Splat target" combo (applied on the next applyRequestedFormat()).
    void drawFormatUi();
    // Phase 6 `--pipeline-stats`: one graphics pipeline per target format.
    std::vector<std::pair<std::string, VkPipeline>> pipelines() const {
        std::vector<std::pair<std::string, VkPipeline>> out;
        const char* names[] = {"splat draw (rgba8)", "splat draw (rgba16f)", "splat draw (rgba32f)"};
        for (size_t f = 0; f < 3; ++f)
            if (m_pipelines[f]) out.push_back({names[f], m_pipelines[f]});
        return out;
    }

private:
    void createAccumulationTarget(VkExtent2D extent);
    bool drawsDirectly() const { return m_format == SplatTargetFormat::Rgba8; }

    VulkanContext* m_ctx = nullptr;
    SplatTargetFormat m_format = SplatTargetFormat::Rgba16f;
    SplatTargetFormat m_requestedFormat = SplatTargetFormat::Rgba16f;
    std::array<bool, 3> m_formatSupported{};
    std::array<VkPipeline, 3> m_pipelines{};

    // Intermediate float target (RGBA16F/RGBA32F modes) and the state its last use left it in.
    Image m_accum;
    ImageState m_accumState{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    VkExtent2D m_extent{};
};

}  // namespace gsr
