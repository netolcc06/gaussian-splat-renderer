#include "render/splat_tile_renderer.h"

#include "core/context.h"
#include "core/pipelines.h"
#include "core/profiler.h"
#include "core/memory_report.h"
#include "render/gpu_scene.h"

#include <imgui.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gsr {

namespace {

// Must match the push_constant block of shaders/splat_preprocess.comp (GSR_TILES build:
// the GSR_COMPACT layout, `flags` unused).
struct PreprocessPushConstants {
    glm::mat4 viewInria;
    glm::vec4 camPos;
    glm::vec2 focal;
    glm::vec2 tanHalfFov;
    glm::uvec2 imageSize;
    uint32_t count;
    uint32_t shCoeffs;
    uint32_t shDegree;
    uint32_t flags;
};
static_assert(sizeof(PreprocessPushConstants) == 120, "layout must match splat_preprocess.comp (GSR_TILES)");

struct DuplicatePushConstants {  // tile_duplicate.comp
    glm::uvec2 tiles;    // bin grid
    uint32_t capacity;
    uint32_t pad;
    glm::uvec2 tiles16;  // INRIA's 16-px getRect grid
};
struct RenderPushConstants {  // tile_render.comp
    glm::uvec2 imageSize;
    uint32_t tilesX;
    uint32_t bound;
    glm::vec2 focal;  // StopThePop ray directions
};
constexpr uint32_t kFlagRayDepth = 4;  // splat_preprocess.comp (GSR_TILES)

// Specialization constant ids (shaders/tile_common.glsl, tile_render.comp).
constexpr uint32_t kBoundsSpecId = 12;
constexpr uint32_t kBinTileSpecId = 13;
constexpr uint32_t kBatchSpecId = 20;
constexpr uint32_t kVoteSpecId = 21;
constexpr uint32_t kSortWindowSpecId = 24;

constexpr uint32_t kGroupSize = 256;  // local_size_x of the one-thread-per-item passes
constexpr VkDeviceSize kOffsetGaussianCount = offsetof(TileControl, gaussianCount);
constexpr VkDeviceSize kOffsetVisible = offsetof(TileControl, visible);
constexpr VkDeviceSize kOffsetDuplicates = offsetof(TileControl, duplicates);
constexpr VkDeviceSize kOffsetEntries = offsetof(TileControl, entries);

VkFormat vkFormatOf(SplatTargetFormat f) {
    switch (f) {
        case SplatTargetFormat::Rgba8: return VK_FORMAT_R8G8B8A8_UNORM;
        case SplatTargetFormat::Rgba16f: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case SplatTargetFormat::Rgba32f: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

void computeToCompute(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

uint32_t groupsFor(uint32_t n) { return std::max(1u, (n + kGroupSize - 1) / kGroupSize); }

// Number of key bits needed for tile ids 0 .. tileCount-1 (at least 1).
uint32_t tileBits(uint32_t tileCount) { return std::max(1u, uint32_t(std::bit_width(std::max(tileCount, 2u) - 1u))); }

}  // namespace

const char* tileSortModeName(TileSortMode m) { return m == TileSortMode::Binning ? "binning" : "global"; }
const char* tileDuplicateModeName(TileDuplicateMode m) {
    return m == TileDuplicateMode::PerSplat ? "per-splat" : "per-entry";
}

const char* renderSplatModeName(RenderSplatMode m) {
    switch (m) {
        case RenderSplatMode::Full: return "full";
        case RenderSplatMode::Packed32: return "p32";
        case RenderSplatMode::Packed24: return "p24";
    }
    return "?";
}

uint32_t renderSplatBytes(RenderSplatMode m) {
    switch (m) {
        case RenderSplatMode::Full: return uint32_t(sizeof(ProjectedSplat));
        case RenderSplatMode::Packed32: return 32;
        case RenderSplatMode::Packed24: return 24;
    }
    return 0;
}

namespace {
constexpr uint32_t kMinAdaptiveCapacity = 256u * 1024u;  // entries (4 MB of list)
constexpr uint32_t kRenderSplatSpecId = 10;               // constant_id of kRenderSplatMode (render_splat.glsl)
uint32_t roundUpToBlock(uint64_t n) {
    const uint64_t b = GpuRadixSort::kBlockSize;
    return uint32_t(std::min<uint64_t>((n + b - 1) / b * b, 0xFFFFF800ull));
}
}  // namespace

uint32_t SplatTileRenderer::capacityForBytes(double bytes) const {
    // Per entry: key + value (+ high key word in global mode), the sort's ping-pong copy of
    // each, and its block histogram (256 uints per 2048 keys = 0.5 B per entry).
    const double bytesPerEntry = m_tileSettings.sortMode == TileSortMode::Global ? 24.5 : 16.5;
    const double maxByRange = double(m_ctx->properties.limits.maxStorageBufferRange) / 4.0;
    uint32_t c = uint32_t(std::clamp(bytes / bytesPerEntry, double(GpuRadixSort::kBlockSize), maxByRange));
    return c - c % GpuRadixSort::kBlockSize;
}

void SplatTileRenderer::init(VulkanContext& ctx, const GpuScene& gpuScene, uint32_t framesInFlight,
                             const SplatSettings& settings, const TileSettings& tileSettings) {
    VkDevice device = ctx.device;
    MemoryScope memoryScope(ctx, "splat-tiles");
    m_ctx = &ctx;
    m_gpuScene = &gpuScene;
    m_settings = settings;
    m_settings.maxShDegree = std::clamp(m_settings.maxShDegree, 0, 3);
    m_tileSettings = tileSettings;
    m_framesInFlight = std::max(framesInFlight, 1u);
    const bool global = m_tileSettings.sortMode == TileSortMode::Global;
    const RenderSplatMode mode = m_tileSettings.renderSplat;
    const bool packed = mode != RenderSplatMode::Full;
    if (m_tileSettings.tileSize != 8 && m_tileSettings.tileSize != 16)
        throw std::runtime_error("splat-tiles: --tile-size must be 8 or 16 (a divisor of INRIA's 16-px tiles)");
    const bool stp = m_tileSettings.sortWindow > 0;
    if (m_tileSettings.sortWindow > 64) throw std::runtime_error("splat-tiles: --sort-window must be <= 64");
    if (m_settings.bounds == TileBounds::CutoffOnly && m_settings.bound == CoverageBound::Radius)
        throw std::runtime_error("--bounds cutoff-only needs --splat-bound tile");
    if (m_tileSettings.fp16Math && !ctx.shaderFloat16)
        throw std::runtime_error("splat-tiles: --render-fp16 needs the shaderFloat16 feature");
    {
        const uint32_t threads = m_tileSettings.tileSize * m_tileSettings.tileSize;
        const uint32_t b = renderBatch();
        if (b == 0 || (b % threads != 0 && threads % b != 0))
            throw std::runtime_error("splat-tiles: --render-batch must divide or be a multiple of the tile's pixel count");
    }

    // The output is a storage image (written by the render pass) that is blitted into the
    // colour target, or the colour target itself for rgba8.
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, vkFormatOf(m_settings.format), &props);
        VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if (m_settings.format != SplatTargetFormat::Rgba8) need |= VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if ((props.optimalTilingFeatures & need) != need)
            throw std::runtime_error(std::string("splat-tiles: format ") + splatTargetFormatName(m_settings.format) +
                                     " cannot be a storage image + blit source on this GPU");
    }

    // --- Capacity of the duplicated list from the memory budget ------------------------
    m_maxCapacity = capacityForBytes(double(std::max(m_tileSettings.budgetMB, 1u)) * 1024.0 * 1024.0);
    uint32_t initialCapacity = m_maxCapacity;
    if (m_tileSettings.adaptiveList) {
        // A first guess (4 entries per gaussian); the measured count corrects it 2 frames later.
        const uint64_t guess = m_tileSettings.adaptiveInitialEntries
                                   ? m_tileSettings.adaptiveInitialEntries
                                   : std::max<uint64_t>(4ull * gpuScene.count, 1u << 20);
        initialCapacity = std::min(m_maxCapacity, roundUpToBlock(guess));
    }

    const uint32_t count = gpuScene.count;
    const VkDeviceSize n = std::max<VkDeviceSize>(count, 1);
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags readable = storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // --- Buffers ---------------------------------------------------------------------
    if (!packed) {
        m_splats = createBuffer(ctx, n * sizeof(ProjectedSplat), readable, MemoryUsage::GpuOnly, "splat-tiles: splats");
    } else {
        m_renderSplats = createBuffer(ctx, n * renderSplatBytes(mode), readable, MemoryUsage::GpuOnly,
                                      mode == RenderSplatMode::Packed32 ? "splat-tiles: render splats (p32)"
                                                                       : "splat-tiles: render splats (p24)");
        m_rects = createBuffer(ctx, n * 8, readable, MemoryUsage::GpuOnly, "splat-tiles: tile rectangles");
        m_depths = createBuffer(ctx, n * 4, readable, MemoryUsage::GpuOnly, "splat-tiles: depths");
    }
    if (stp)
        m_rayDepth = createBuffer(ctx, n * 48, storage, MemoryUsage::GpuOnly, "splat-tiles: ray-depth params (StopThePop)");
    m_dummy = createBuffer(ctx, 256, storage, MemoryUsage::GpuOnly, "splat-tiles: unused binding");
    m_tileCounts = createBuffer(ctx, n * 4, readable, MemoryUsage::GpuOnly, "splat-tiles: tiles touched");
    m_offsets = createBuffer(ctx, n * 4, storage, MemoryUsage::GpuOnly, "splat-tiles: duplicate offsets (scan)");
    if (!global) {
        m_slots = createBuffer(ctx, n * 4, storage, MemoryUsage::GpuOnly, "splat-tiles: compact slots (scan)");
        m_depthKeys = createBuffer(ctx, n * 4, storage, MemoryUsage::GpuOnly, "splat-tiles: depth keys");
        m_depthValues = createBuffer(ctx, n * 4, storage, MemoryUsage::GpuOnly, "splat-tiles: depth-sorted indices");
    }
    m_control = createBuffer(ctx, sizeof(TileControl),
                             storage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             MemoryUsage::GpuOnly, "splat-tiles: control");
    m_readback.resize(framesInFlight);
    m_slotInfo.assign(framesInFlight, SlotInfo{});
    // Per-tile buffers and the readbacks are (re)created in onRenderTargetsChanged.

    // --- Scans and the depth sort (the tile sort belongs to the list, createList) ------
    if (!global) {
        GpuScan::Bindings flags;
        flags.input = m_tileCounts.buffer;
        flags.output = m_slots.buffer;
        flags.countBuffer = m_control.buffer;
        flags.countOffset = uint32_t(kOffsetGaussianCount);
        flags.totalBuffer = m_control.buffer;
        flags.totalOffset = uint32_t(kOffsetVisible);
        flags.flags = GpuScan::kNonZero;
        m_scanFlags.init(ctx, count, flags);
        m_depthSort.init(ctx, count, m_depthKeys.buffer, m_depthValues.buffer, m_control.buffer,
                         uint32_t(kOffsetVisible));
        m_depthSort.setVariant(m_tileSettings.radixVariant);
    }
    GpuScan::Bindings tiles;
    tiles.input = m_tileCounts.buffer;
    tiles.output = m_offsets.buffer;
    tiles.countBuffer = m_control.buffer;
    tiles.totalBuffer = m_control.buffer;
    tiles.totalOffset = uint32_t(kOffsetDuplicates);
    if (global) {
        tiles.countOffset = uint32_t(kOffsetGaussianCount);  // every gaussian, index order
    } else {
        tiles.index = m_depthValues.buffer;  // the j-th nearest visible splat's count
        tiles.countOffset = uint32_t(kOffsetVisible);
        tiles.flags = GpuScan::kGather;
    }
    m_scanTiles.init(ctx, count, tiles);

    // --- Descriptor sets and pipelines --------------------------------------------------
    // Sets of the passes that read the duplicated list (duplicate, ranges, render) belong to
    // the list (ListResources) and are allocated in createList.
    m_pool.create(device, 4, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32}});
    auto makeLayout = [&](Pass& pass, uint32_t storageBuffers, uint32_t pushBytes, int imageBinding = -1) {
        DescriptorLayoutBuilder b;
        for (uint32_t i = 0; i < storageBuffers; ++i)
            b.add(i, int(i) == imageBinding ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_SHADER_STAGE_COMPUTE_BIT);
        pass.setLayout = b.build(device);
        std::vector<VkPushConstantRange> push;
        if (pushBytes) push.push_back({VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes});
        pass.layout = createPipelineLayout(device, {pass.setLayout}, push);
    };
    constexpr auto kSB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const VkBuffer splatsOrDummy = packed ? m_dummy.buffer : m_splats.buffer;
    const VkBuffer rectsOrDummy = packed ? m_rects.buffer : m_dummy.buffer;
    const VkBuffer depthsOrDummy = packed ? m_depths.buffer : m_dummy.buffer;
    SpecializationConstants modeSpec;
    modeSpec.set(kRenderSplatSpecId, uint32_t(mode));
    SpecializationConstants binSpec;  // duplicate pass: render splat mode, bounds, bin tile size
    binSpec.set(kRenderSplatSpecId, uint32_t(mode))
        .set(kBoundsSpecId, uint32_t(m_settings.bounds))
        .set(kBinTileSpecId, m_tileSettings.tileSize);

    // Preprocess: 0-3 scene, 4 Splat (full), 5 tile counts, 6 unused, 7 control,
    // 8 p32 render splats, 9 p24 render splats, 10 rects, 11 depths (packed),
    // 12 ray-depth params (StopThePop).
    makeLayout(m_preprocessPass, 13, sizeof(PreprocessPushConstants));
    m_preprocessPass.set = m_pool.allocate(device, m_preprocessPass.setLayout);
    {
        DescriptorWriter w;
        gpuScene.bind(w);
        w.buffer(4, kSB, splatsOrDummy)
            .buffer(5, kSB, m_tileCounts.buffer)
            .buffer(6, kSB, m_dummy.buffer)
            .buffer(7, kSB, m_control.buffer)
            .buffer(8, kSB, mode == RenderSplatMode::Packed32 ? m_renderSplats.buffer : m_dummy.buffer)
            .buffer(9, kSB, mode == RenderSplatMode::Packed24 ? m_renderSplats.buffer : m_dummy.buffer)
            .buffer(10, kSB, rectsOrDummy)
            .buffer(11, kSB, depthsOrDummy)
            .buffer(12, kSB, stp ? m_rayDepth.buffer : m_dummy.buffer)
            .update(device, m_preprocessPass.set);
    }
    {
        SpecializationConstants spec;
        gpuScene.addSpecialization(spec);  // the scene decoder's format (scene_decode.glsl)
        spec.set(kRenderSplatSpecId, uint32_t(mode))
            .set(kBoundsSpecId, uint32_t(m_settings.bounds))
            .set(kBinTileSpecId, m_tileSettings.tileSize);
        // Subgroup ballots for the cull counters: every lane of a subgroup must exist.
        m_preprocessPass.pipeline =
            createComputePipeline(device, m_preprocessPass.layout, "splat_preprocess_tiles.comp.spv", spec.info(),
                                  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
    }

    if (!global) {
        makeLayout(m_compactPass, 7, 0);
        m_compactPass.set = m_pool.allocate(device, m_compactPass.setLayout);
        DescriptorWriter()
            .buffer(0, kSB, splatsOrDummy)
            .buffer(1, kSB, m_tileCounts.buffer)
            .buffer(2, kSB, m_slots.buffer)
            .buffer(3, kSB, m_depthKeys.buffer)
            .buffer(4, kSB, m_depthValues.buffer)
            .buffer(5, kSB, m_control.buffer)
            .buffer(6, kSB, depthsOrDummy)
            .update(device, m_compactPass.set);
        m_compactPass.pipeline =
            createComputePipeline(device, m_compactPass.layout, "tile_compact.comp.spv", modeSpec.info());
    }

    makeLayout(m_setupPass, 1, sizeof(uint32_t));
    m_setupPass.set = m_pool.allocate(device, m_setupPass.setLayout);
    DescriptorWriter().buffer(0, kSB, m_control.buffer).update(device, m_setupPass.set);
    m_setupPass.pipeline = createComputePipeline(device, m_setupPass.layout, "tile_setup.comp.spv");

    // Duplicate: 0 Splat (full), 1 order, 2 counts, 3 offsets, 4-6 list, 7 control, 8 rects, 9 depths,
    // 10 p32 / 11 p24 render splats (Phase 6 coverage).
    makeLayout(m_duplicatePass, 12, sizeof(DuplicatePushConstants));
    const bool perEntry = m_tileSettings.duplicateMode == TileDuplicateMode::PerEntry;
    m_duplicateShaderName = global ? (perEntry ? "tile_duplicate_global_per_entry.comp.spv"
                                               : "tile_duplicate_global.comp.spv")
                                   : (perEntry ? "tile_duplicate_per_entry.comp.spv" : "tile_duplicate.comp.spv");
    m_duplicatePass.pipeline =
        createComputePipeline(device, m_duplicatePass.layout, m_duplicateShaderName, binSpec.info());

    makeLayout(m_rangesPass, 3, sizeof(uint32_t));
    m_rangesPass.pipeline = createComputePipeline(device, m_rangesPass.layout, "tile_ranges.comp.spv");

    // Render: 0 Splat (full), 1 list values, 2 ranges, 3 output image, 4 tile stats,
    // 5 p32 render splats, 6 p24 render splats, 7 ray-depth params (StopThePop).
    makeLayout(m_renderPass, 8, sizeof(RenderPushConstants), 3);
    {
        // SPIR-V variant: tile_render[_packed][_stp|_aos|_fp16][_rgba8|_rgba32f|_stats].comp.spv
        // (CMakeLists.txt builds the combinations below; others are rejected here).
        std::string prefix = packed ? "tile_render_packed" : "tile_render";
        const int experiments = int(stp) + int(m_tileSettings.sharedAoS) + int(m_tileSettings.fp16Math);
        if (experiments > 1) throw std::runtime_error("splat-tiles: --sort-window, --render-shared aos and --render-fp16 "
                                                      "are separate experiments (one at a time)");
        if (stp) prefix += "_stp";
        if (m_tileSettings.sharedAoS) prefix += "_aos";
        if (m_tileSettings.fp16Math) prefix += "_fp16";
        const bool variant = experiments > 0;
        m_renderShaderNames = {variant ? "" : prefix + "_rgba8.comp.spv", prefix + ".comp.spv",
                               variant && !stp ? "" : prefix + "_rgba32f.comp.spv"};
        m_renderStatsShaderName = variant ? "" : prefix + "_stats.comp.spv";
        if (m_renderShaderNames[size_t(m_settings.format)].empty())
            throw std::runtime_error("splat-tiles: this render experiment is only built for the rgba16f target"
                                     " (StopThePop: rgba16f/rgba32f)");
        if (m_tileSettings.stats && m_renderStatsShaderName.empty())
            throw std::runtime_error("splat-tiles: --tile-stats is not built for this render experiment");
        SpecializationConstants renderSpec;
        renderSpec.set(kRenderSplatSpecId, uint32_t(mode))
            .set(kBinTileSpecId, m_tileSettings.tileSize)
            .set(kBatchSpecId, renderBatch())
            .set(kVoteSpecId, m_tileSettings.voteEvery);
        if (stp) renderSpec.set(kSortWindowSpecId, m_tileSettings.sortWindow);
        for (size_t f = 0; f < 3; ++f)
            if (!m_renderShaderNames[f].empty())
                m_renderPipelines[f] =
                    createComputePipeline(device, m_renderPass.layout, m_renderShaderNames[f], renderSpec.info());
        if (!m_renderStatsShaderName.empty())
            m_renderStatsPipeline =
                createComputePipeline(device, m_renderPass.layout, m_renderStatsShaderName, renderSpec.info());
    }
    if (m_tileSettings.stats && m_settings.format != SplatTargetFormat::Rgba16f)
        throw std::runtime_error("splat-tiles: --tile-stats needs the rgba16f target");

    m_list = createList(initialCapacity);
}

std::unique_ptr<SplatTileRenderer::ListResources> SplatTileRenderer::createList(uint32_t capacity) {
    VulkanContext& ctx = *m_ctx;
    MemoryScope memoryScope(ctx, "splat-tiles");
    const bool global = m_tileSettings.sortMode == TileSortMode::Global;
    auto list = std::make_unique<ListResources>();
    list->capacity = capacity;
    const VkDeviceSize c = capacity;
    const VkBufferUsageFlags readable = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    list->keys = createBuffer(ctx, c * 4, readable, MemoryUsage::GpuOnly,
                              global ? "splat-tiles: list keys (depth bits)" : "splat-tiles: list keys (tile)");
    if (global)
        list->keysHi = createBuffer(ctx, c * 4, readable, MemoryUsage::GpuOnly, "splat-tiles: list keys (tile)");
    list->values = createBuffer(ctx, c * 4, readable, MemoryUsage::GpuOnly, "splat-tiles: list values (gaussian)");
    list->sort.init(ctx, capacity, list->keys.buffer, list->values.buffer, m_control.buffer, uint32_t(kOffsetEntries),
                    global ? list->keysHi.buffer : VK_NULL_HANDLE);
    list->sort.setVariant(m_tileSettings.radixVariant);
    list->pool.create(ctx.device, 3,
                      {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}});
    list->duplicateSet = list->pool.allocate(ctx.device, m_duplicatePass.setLayout);
    list->rangesSet = list->pool.allocate(ctx.device, m_rangesPass.setLayout);
    list->renderSet = list->pool.allocate(ctx.device, m_renderPass.setLayout);
    writeListSets(*list);
    return list;
}

void SplatTileRenderer::writeListSets(ListResources& list) {
    VkDevice device = m_ctx->device;
    constexpr auto kSB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const bool global = m_tileSettings.sortMode == TileSortMode::Global;
    const RenderSplatMode mode = m_tileSettings.renderSplat;
    const bool packed = mode != RenderSplatMode::Full;
    const VkBuffer splatsOrDummy = packed ? m_dummy.buffer : m_splats.buffer;
    DescriptorWriter()
        .buffer(0, kSB, splatsOrDummy)
        .buffer(1, kSB, global ? m_tileCounts.buffer : m_depthValues.buffer)
        .buffer(2, kSB, m_tileCounts.buffer)
        .buffer(3, kSB, m_offsets.buffer)
        .buffer(4, kSB, list.keys.buffer)
        .buffer(5, kSB, global ? list.keysHi.buffer : list.keys.buffer)
        .buffer(6, kSB, list.values.buffer)
        .buffer(7, kSB, m_control.buffer)
        .buffer(8, kSB, packed ? m_rects.buffer : m_dummy.buffer)
        .buffer(9, kSB, packed ? m_depths.buffer : m_dummy.buffer)
        .buffer(10, kSB, mode == RenderSplatMode::Packed32 ? m_renderSplats.buffer : m_dummy.buffer)
        .buffer(11, kSB, mode == RenderSplatMode::Packed24 ? m_renderSplats.buffer : m_dummy.buffer)
        .update(device, list.duplicateSet);
    if (!m_ranges.buffer) return;  // the per-tile buffers come with the render targets
    DescriptorWriter()
        .buffer(0, kSB, global ? list.keysHi.buffer : list.keys.buffer)
        .buffer(1, kSB, m_ranges.buffer)
        .buffer(2, kSB, m_control.buffer)
        .update(device, list.rangesSet);
    DescriptorWriter()
        .buffer(0, kSB, splatsOrDummy)
        .buffer(1, kSB, list.values.buffer)
        .buffer(2, kSB, m_ranges.buffer)
        .image(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_boundView, VK_IMAGE_LAYOUT_GENERAL)
        .buffer(4, kSB, m_tileStats.buffer)
        .buffer(5, kSB, mode == RenderSplatMode::Packed32 ? m_renderSplats.buffer : m_dummy.buffer)
        .buffer(6, kSB, mode == RenderSplatMode::Packed24 ? m_renderSplats.buffer : m_dummy.buffer)
        .buffer(7, kSB, m_rayDepth.buffer ? m_rayDepth.buffer : m_dummy.buffer)
        .update(device, list.renderSet);
}

void SplatTileRenderer::destroyList(ListResources& list) {
    list.sort.destroy(*m_ctx);
    list.pool.destroy(m_ctx->device);
    destroyBuffer(*m_ctx, list.keys);
    destroyBuffer(*m_ctx, list.keysHi);
    destroyBuffer(*m_ctx, list.values);
}

void SplatTileRenderer::adaptCapacity(uint64_t frameIndex) {
    // Retired lists whose last frame has completed: the app waited for frame
    // frameIndex - framesInFlight (this slot's previous frame) before calling render().
    for (size_t i = 0; i < m_retiredLists.size();) {
        if (m_retiredLists[i]->lastUsedFrame + m_framesInFlight <= frameIndex) {
            destroyList(*m_retiredLists[i]);
            m_retiredLists.erase(m_retiredLists.begin() + std::ptrdiff_t(i));
        } else {
            ++i;
        }
    }
    if (!m_tileSettings.adaptiveList || !m_lastStats.valid || m_lastStats.frameIndex == m_lastAdaptedFrame) return;
    m_lastAdaptedFrame = m_lastStats.frameIndex;

    // M is the UNCLAMPED entry count of a frame two frames ago, so it is the true need
    // even when that frame overflowed.
    const uint32_t m = m_lastStats.control.duplicates;
    const uint32_t cap = m_list->capacity;
    uint64_t target = 0;
    if (double(m) > 0.9 * cap) {
        target = uint64_t(1.5 * m);  // grow at once, with 50% headroom
        m_shrinkFrames = 0;
        m_shrinkWindowMax = 0;
    } else if (double(m) < 0.4 * cap) {
        m_shrinkWindowMax = std::max(m_shrinkWindowMax, m);
        if (++m_shrinkFrames >= 120) {  // shrink slowly: 120 consecutive readbacks below 40%
            target = uint64_t(1.5 * m_shrinkWindowMax);
            m_shrinkFrames = 0;
            m_shrinkWindowMax = 0;
        }
    } else {
        m_shrinkFrames = 0;
        m_shrinkWindowMax = 0;
    }
    if (target == 0) return;
    const uint32_t newCap = std::clamp(roundUpToBlock(target), std::min(kMinAdaptiveCapacity, m_maxCapacity), m_maxCapacity);
    if (newCap == cap || (newCap < cap && double(newCap) > 0.75 * cap)) return;
    // No stall: the old list (and its descriptor sets) may still be read by the frames in
    // flight; it is destroyed once they have completed (top of this function).
    m_retiredLists.push_back(std::move(m_list));
    m_list = createList(newCap);
    ++m_listResizes;
}

int SplatTileRenderer::effectiveShDegree() const { return std::min(m_gpuScene->shDegree, m_settings.maxShDegree); }

uint32_t SplatTileRenderer::renderBatch() const {
    return m_tileSettings.renderBatch ? m_tileSettings.renderBatch : m_tileSettings.tileSize * m_tileSettings.tileSize;
}

std::vector<std::pair<std::string, VkPipeline>> SplatTileRenderer::pipelines() const {
    std::vector<std::pair<std::string, VkPipeline>> out;
    out.push_back({"splat_preprocess_tiles.comp.spv", m_preprocessPass.pipeline});
    if (m_compactPass.pipeline) out.push_back({"tile_compact.comp.spv", m_compactPass.pipeline});
    out.push_back({"tile_setup.comp.spv", m_setupPass.pipeline});
    out.push_back({m_duplicateShaderName, m_duplicatePass.pipeline});
    out.push_back({"tile_ranges.comp.spv", m_rangesPass.pipeline});
    for (size_t f = 0; f < 3; ++f)
        if (m_renderPipelines[f]) out.push_back({m_renderShaderNames[f], m_renderPipelines[f]});
    if (m_renderStatsPipeline) out.push_back({m_renderStatsShaderName, m_renderStatsPipeline});
    if (m_compactPass.pipeline) {
        for (auto& p : m_scanFlags.pipelines("scan: visible flags")) out.push_back(p);
        for (auto& p : m_depthSort.pipelines("depth sort")) out.push_back(p);
    }
    for (auto& p : m_scanTiles.pipelines("scan: tiles touched")) out.push_back(p);
    if (m_list) for (auto& p : m_list->sort.pipelines("tile sort")) out.push_back(p);
    return out;
}

VkDeviceSize SplatTileRenderer::listBytes() const {
    if (!m_list) return 0;
    return m_list->keys.size + m_list->keysHi.size + m_list->values.size + m_list->sort.scratchBytes();
}

VkDeviceSize SplatTileRenderer::totalBytes() const {
    VkDeviceSize b = listBytes();
    for (const Buffer* x : {&m_splats, &m_renderSplats, &m_rects, &m_depths, &m_rayDepth, &m_dummy, &m_tileCounts,
                            &m_slots, &m_depthKeys, &m_depthValues, &m_offsets, &m_ranges, &m_tileStats, &m_control})
        b += x->size;
    b += m_depthSort.scratchBytes();
    const uint32_t bpp = m_settings.format == SplatTargetFormat::Rgba32f ? 16 : 8;
    if (m_output.image) b += VkDeviceSize(m_output.extent.width) * m_output.extent.height * bpp;
    return b;
}

void SplatTileRenderer::createOutputTarget(VkExtent2D extent) {
    destroyImage(*m_ctx, m_output);
    m_outputState = {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    if (m_settings.format == SplatTargetFormat::Rgba8) return;  // renders into the colour target
    m_output = createImage2D(*m_ctx, extent.width, extent.height, vkFormatOf(m_settings.format),
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "splat-tiles: output");
}

void SplatTileRenderer::onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image&) {
    MemoryScope memoryScope(ctx, "splat-tiles");
    createOutputTarget(color.extent);
    const uint32_t S = m_tileSettings.tileSize;  // the binning grid (INRIA's getRect stays on 16 px)
    m_tilesX = (color.extent.width + S - 1) / S;
    m_tilesY = (color.extent.height + S - 1) / S;
    const uint32_t tileCount = m_tilesX * m_tilesY;

    // Called with the GPU idle: lists retired by the adaptive capacity can go now.
    for (auto& l : m_retiredLists) destroyList(*l);
    m_retiredLists.clear();

    // Per-tile buffers follow the tile grid.
    destroyBuffer(ctx, m_ranges);
    destroyBuffer(ctx, m_tileStats);
    m_ranges = createBuffer(ctx, VkDeviceSize(tileCount) * 8,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            MemoryUsage::GpuOnly, "splat-tiles: tile ranges");
    m_tileStats = createBuffer(ctx, VkDeviceSize(tileCount) * kTileStatCount * 4,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               MemoryUsage::GpuOnly, "splat-tiles: tile stats");
    for (size_t s = 0; s < m_readback.size(); ++s) {
        destroyBuffer(ctx, m_readback[s]);
        m_readback[s] = createBuffer(ctx, sizeof(TileControl) + VkDeviceSize(tileCount) * kTileStatCount * 4,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback,
                                     "splat-tiles: stats readback");
        m_slotInfo[s] = SlotInfo{};
    }

    m_boundView = m_settings.format == SplatTargetFormat::Rgba8 ? color.view : m_output.view;
    writeListSets(*m_list);
}

void SplatTileRenderer::readSlot(uint32_t s) {
    SlotInfo& info = m_slotInfo[s];
    if (!info.used) return;
    info.used = false;  // consumed once
    const Buffer& rb = m_readback[s];
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    TileControl control;
    std::memcpy(&control, rb.mapped, sizeof(TileControl));

    m_maxDuplicates = std::max(m_maxDuplicates, control.duplicates);
    m_coverageMismatches += control.coverageMismatch;
    if (control.coverageMismatch > 0)
        std::printf("warning: splat-tiles: frame %llu: %u splats wrote a different number of tiles than counted\n",
                    static_cast<unsigned long long>(info.frameIndex), control.coverageMismatch);
    if (control.overflow > 0) {
        ++m_overflowFrames;
        m_maxOverflow = std::max(m_maxOverflow, control.overflow);
        // Adaptive list: overflow while the capacity catches up is expected (first frames,
        // sudden jumps); warn only when the budget itself is too small.
        if (!m_overflowWarned && (!m_tileSettings.adaptiveList || capacity() >= m_maxCapacity)) {
            std::printf("warning: splat-tiles: frame %llu produced %u (tile, splat) entries but the list holds %u; "
                        "%u entries dropped (raise --tile-budget-mb, currently %u MB)\n",
                        static_cast<unsigned long long>(info.frameIndex), control.duplicates, capacity(),
                        control.overflow, m_tileSettings.budgetMB);
            std::fflush(stdout);
            m_overflowWarned = true;
        }
    }
    if (m_lastStats.valid && info.frameIndex < m_lastStats.frameIndex) return;
    m_lastStats.valid = true;
    m_lastStats.frameIndex = info.frameIndex;
    m_lastStats.control = control;
    m_lastStats.tilesX = info.tilesX;
    m_lastStats.tilesY = info.tilesY;
    if (m_tileSettings.stats) {
        const size_t words = size_t(info.tilesX) * info.tilesY * kTileStatCount;
        const auto* src = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(rb.mapped) + sizeof(TileControl));
        m_lastStats.tiles.assign(src, src + words);
    } else {
        m_lastStats.tiles.clear();
    }
}

void SplatTileRenderer::collectPending() {
    // Oldest first, so that the newest frame ends up in m_lastStats.
    std::vector<uint32_t> order(m_slotInfo.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return m_slotInfo[a].frameIndex < m_slotInfo[b].frameIndex; });
    for (uint32_t s : order) readSlot(s);
}

RenderOutput SplatTileRenderer::render(const FrameInput& in) {
    VkCommandBuffer cmd = in.cmd;
    const VkExtent2D extent = in.color.extent;
    const bool global = m_tileSettings.sortMode == TileSortMode::Global;
    const bool direct = m_settings.format == SplatTargetFormat::Rgba8;
    const bool stats = m_tileSettings.stats && m_settings.format == SplatTargetFormat::Rgba16f;

    // The app waited for the frame that last used this slot: its readback is complete.
    readSlot(in.frameSlot);
    adaptCapacity(in.frameIndex);  // retire/replace the list (adaptive capacity), no stall
    ListResources& list = *m_list;
    list.lastUsedFrame = in.frameIndex;
    const uint32_t capacity = list.capacity;

    const SplatCamera cam = splatCameraFromViewer(in.view, in.proj, in.model, extent.width, extent.height);
    const uint32_t count = m_gpuScene->count;
    const uint32_t tileCount = m_tilesX * m_tilesY;

    // Frame boundary: the previous frame's compute passes read and wrote every buffer here,
    // read the control block as indirect arguments, and copied it (transfer). This frame's
    // transfers and compute passes overwrite them: wait for all of it (WAR) and make the
    // previous writes available (WAW).
    memoryBarrier(cmd,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 0. Reset: control block {N, 0, ...}, empty tile ranges, tile stats ===========
    {
        auto timer = in.profiler.scope(cmd, "reset (transfer)");
        TileControl reset{};
        reset.gaussianCount = count;
        vkCmdUpdateBuffer(cmd, m_control.buffer, 0, sizeof(reset), &reset);
        vkCmdFillBuffer(cmd, m_ranges.buffer, 0, VK_WHOLE_SIZE, 0);
        if (stats) vkCmdFillBuffer(cmd, m_tileStats.buffer, 0, VK_WHOLE_SIZE, 0);
    }
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 1. Preprocess + cull: Splat[i], tileCounts[i], cull counters =================
    {
        auto timer = in.profiler.scope(cmd, "preprocess + cull + tile counts (compute)");
        PreprocessPushConstants pc{};
        pc.viewInria = cam.viewInria;
        pc.camPos = glm::vec4(cam.position, 1.0f);
        pc.focal = {cam.focalX, cam.focalY};
        pc.tanHalfFov = {cam.tanHalfFovX, cam.tanHalfFovY};
        pc.imageSize = {extent.width, extent.height};
        pc.count = count;
        pc.shCoeffs = m_gpuScene->shCoeffsPerChannel;
        pc.shDegree = uint32_t(effectiveShDegree());
        pc.flags = m_tileSettings.sortWindow > 0 ? kFlagRayDepth : 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessPass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preprocessPass.layout, 0, 1, &m_preprocessPass.set, 0,
                                nullptr);
        vkCmdPushConstants(cmd, m_preprocessPass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groupsFor(count), 1, 1);
    }

    if (!global) {
        // === 2-5 (binning). Order-preserving compaction, depth sort, scan in depth order ==
        m_scanFlags.record(cmd, &in.profiler, "scan: visible flags");  // starts with C -> C
        computeToCompute(cmd);  // slots (scan) -> compact
        {
            auto timer = in.profiler.scope(cmd, "compact (index order)");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compactPass.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compactPass.layout, 0, 1, &m_compactPass.set, 0,
                                    nullptr);
            vkCmdDispatch(cmd, groupsFor(count), 1, 1);
        }
        {
            auto timer = in.profiler.scope(cmd, "depth sort (total)");
            m_depthSort.setVariant(m_tileSettings.radixVariant);
            GpuRadixSort::RecordOptions o;
            o.keyBits = 32;
            o.label = "depth sort";
            o.perPassScopes = true;
            m_depthSort.record(cmd, &in.profiler, o);  // starts with C -> C
        }
        m_scanTiles.record(cmd, &in.profiler, "scan: tiles touched");  // starts with C -> C
    } else {
        // === 2 (global). Scan of the touched-tile counts in index order ==================
        m_scanTiles.record(cmd, &in.profiler, "scan: tiles touched");
    }
    computeToCompute(cmd);  // M (scan total) -> setup

    // === 6. Setup: clamp to the capacity, overflow, ranges dispatch args =================
    {
        auto timer = in.profiler.scope(cmd, "tile setup (1 thread)");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_setupPass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_setupPass.layout, 0, 1, &m_setupPass.set, 0, nullptr);
        vkCmdPushConstants(cmd, m_setupPass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &capacity);
        vkCmdDispatch(cmd, 1, 1, 1);
    }
    // Per-entry duplication reads `entries` and is launched with the indirect arguments
    // setup wrote (one thread per entry, the same arguments as the ranges pass). The
    // per-splat variant reads neither, but both touch the control block: always ordered.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 7. Duplicate: one entry per (tile, splat) at the scanned offsets ==============
    {
        auto timer = in.profiler.scope(cmd, "duplicate (compute)");
        const DuplicatePushConstants pc{{m_tilesX, m_tilesY}, capacity, 0, {cam.tilesX, cam.tilesY}};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_duplicatePass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_duplicatePass.layout, 0, 1, &list.duplicateSet, 0,
                                nullptr);
        vkCmdPushConstants(cmd, m_duplicatePass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (m_tileSettings.duplicateMode == TileDuplicateMode::PerEntry)
            vkCmdDispatchIndirect(cmd, m_control.buffer, offsetof(TileControl, rangesArgs));  // one per entry
        else
            vkCmdDispatch(cmd, groupsFor(count), 1, 1);  // V <= N (binning) or N (global) threads
    }

    // === 8. Sort the list: stable by tile (binning) or by 64-bit tile|depth (global) ====
    {
        auto timer = in.profiler.scope(cmd, "tile sort (total)");
        list.sort.setVariant(m_tileSettings.radixVariant);
        GpuRadixSort::RecordOptions o;
        // Phase 6 coverage modes may write the sentinel tile id = tileCount (never seen):
        // it must fit in the sorted bits too.
        const bool rectOnly = m_settings.bounds == TileBounds::Rect && m_tileSettings.tileSize == 16;
        const uint32_t keyTiles = rectOnly ? tileCount : tileCount + 1;
        o.keyBits = global ? 32 + tileBits(keyTiles) : tileBits(keyTiles);
        o.label = "tile sort";
        o.perPassScopes = true;
        list.sort.record(cmd, &in.profiler, o);  // starts with C -> C
    }
    // Sorted keys (sort) and the ranges dispatch arguments (setup) -> ranges pass, which
    // reads the arguments as INDIRECT parameters.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // === 9. Tile ranges ================================================================
    {
        auto timer = in.profiler.scope(cmd, "tile ranges (compute)");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rangesPass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rangesPass.layout, 0, 1, &list.rangesSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_rangesPass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &tileCount);
        vkCmdDispatchIndirect(cmd, m_control.buffer, offsetof(TileControl, rangesArgs));
    }
    computeToCompute(cmd);  // ranges -> render

    // === 10. Render: one workgroup per tile =============================================
    const Image& target = direct ? in.color : m_output;
    const ImageState before = direct ? in.colorBefore : m_outputState;
    {
        auto timer = in.profiler.scope(cmd, "render tiles (compute)");
        // Previous contents are discarded; every pixel is written by exactly one thread.
        imageBarrier(cmd, target.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     before.stage, before.access, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        const RenderPushConstants pc{{extent.width, extent.height}, m_tilesX, uint32_t(m_settings.bound),
                                     {cam.focalX, cam.focalY}};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          stats ? m_renderStatsPipeline : m_renderPipelines[uint32_t(m_settings.format)]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_renderPass.layout, 0, 1, &list.renderSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_renderPass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, m_tilesX, m_tilesY, 1);
    }

    // Control block (+ tile stats) -> this slot's host buffer (read 2 frames later).
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    {
        const VkBufferCopy region{0, 0, sizeof(TileControl)};
        vkCmdCopyBuffer(cmd, m_control.buffer, m_readback[in.frameSlot].buffer, 1, &region);
        if (stats) {
            const VkBufferCopy statsRegion{0, sizeof(TileControl), m_tileStats.size};
            vkCmdCopyBuffer(cmd, m_tileStats.buffer, m_readback[in.frameSlot].buffer, 1, &statsRegion);
        }
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    }
    m_slotInfo[in.frameSlot] = {true, in.frameIndex, m_tilesX, m_tilesY};

    RenderOutput out;
    out.depth = in.depthBefore;  // the depth target is not used
    if (direct) {
        out.color = {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
        return out;
    }

    // === 11. Resolve: float target -> RGBA8 colour target (as SplatDrawStage) =========
    {
        auto timer = in.profiler.scope(cmd, "resolve (blit to rgba8)");
        imageBarrier(cmd, m_output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        imageBarrier(cmd, in.color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, in.colorBefore.stage, in.colorBefore.access,
                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
        vkCmdBlitImage(cmd, m_output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, in.color.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    // Next frame's render writes m_output only after this blit's reads (execution dependency).
    m_outputState = {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE};
    out.color = {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
    return out;
}

void SplatTileRenderer::readBack(std::vector<uint32_t>& tileKeys, std::vector<uint32_t>& values,
                                 std::vector<uint32_t>& ranges, TileControl& control) {
    const bool global = m_tileSettings.sortMode == TileSortMode::Global;
    const VkDeviceSize c = m_list->capacity;
    const VkDeviceSize rangesBytes = m_ranges.size;
    Buffer rb = createBuffer(*m_ctx, 2 * c * 4 + rangesBytes + sizeof(TileControl), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::Readback, "splat-tiles: debug readback");
    m_ctx->immediateSubmit([&](VkCommandBuffer cmd) {
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy k{0, 0, c * 4}, v{0, c * 4, c * 4}, r{0, 2 * c * 4, rangesBytes},
            ctl{0, 2 * c * 4 + rangesBytes, sizeof(TileControl)};
        vkCmdCopyBuffer(cmd, global ? m_list->keysHi.buffer : m_list->keys.buffer, rb.buffer, 1, &k);
        vkCmdCopyBuffer(cmd, m_list->values.buffer, rb.buffer, 1, &v);
        vkCmdCopyBuffer(cmd, m_ranges.buffer, rb.buffer, 1, &r);
        vkCmdCopyBuffer(cmd, m_control.buffer, rb.buffer, 1, &ctl);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* data = static_cast<const uint8_t*>(rb.mapped);
    std::memcpy(&control, data + 2 * c * 4 + rangesBytes, sizeof(TileControl));
    const uint32_t n = std::min(control.entries, m_list->capacity);
    tileKeys.resize(n);
    values.resize(n);
    std::memcpy(tileKeys.data(), data, size_t(n) * 4);
    std::memcpy(values.data(), data + c * 4, size_t(n) * 4);
    ranges.resize(size_t(rangesBytes / 4));
    std::memcpy(ranges.data(), data + 2 * c * 4, size_t(rangesBytes));
    destroyBuffer(*m_ctx, rb);
}

void SplatTileRenderer::readBackSplats(std::vector<ProjectedSplat>& splats, std::vector<uint32_t>& tileCounts) {
    if (m_tileSettings.renderSplat != RenderSplatMode::Full)
        throw std::runtime_error("splat-tiles: readBackSplats needs --render-splat full");
    const uint32_t n = m_gpuScene->count;
    const VkDeviceSize splatBytes = VkDeviceSize(n) * sizeof(ProjectedSplat), countBytes = VkDeviceSize(n) * 4;
    Buffer rb = createBuffer(*m_ctx, std::max<VkDeviceSize>(splatBytes + countBytes, 4),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback, "splat-tiles: splat readback");
    m_ctx->immediateSubmit([&](VkCommandBuffer cmd) {
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        if (n > 0) {
            const VkBufferCopy a{0, 0, splatBytes}, b{0, splatBytes, countBytes};
            vkCmdCopyBuffer(cmd, m_splats.buffer, rb.buffer, 1, &a);
            vkCmdCopyBuffer(cmd, m_tileCounts.buffer, rb.buffer, 1, &b);
        }
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* data = static_cast<const uint8_t*>(rb.mapped);
    splats.resize(n);
    tileCounts.resize(n);
    std::memcpy(splats.data(), data, size_t(splatBytes));
    std::memcpy(tileCounts.data(), data + splatBytes, size_t(countBytes));
    destroyBuffer(*m_ctx, rb);
}

void SplatTileRenderer::readBackPackedSplats(std::vector<uint32_t>& words, std::vector<uint32_t>& rects,
                                             std::vector<uint32_t>& depths, std::vector<uint32_t>& tileCounts) {
    if (m_tileSettings.renderSplat == RenderSplatMode::Full)
        throw std::runtime_error("splat-tiles: readBackPackedSplats needs a packed render-splat mode");
    const uint32_t n = m_gpuScene->count;
    const VkDeviceSize splatBytes = m_renderSplats.size, rectBytes = VkDeviceSize(n) * 8, depthBytes = VkDeviceSize(n) * 4,
                       countBytes = VkDeviceSize(n) * 4;
    const VkDeviceSize total = splatBytes + rectBytes + depthBytes + countBytes;
    Buffer rb = createBuffer(*m_ctx, std::max<VkDeviceSize>(total, 4), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MemoryUsage::Readback, "splat-tiles: packed splat readback");
    m_ctx->immediateSubmit([&](VkCommandBuffer cmd) {
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        if (n > 0) {
            const VkBufferCopy a{0, 0, splatBytes}, b{0, splatBytes, rectBytes}, c{0, splatBytes + rectBytes, depthBytes},
                d{0, splatBytes + rectBytes + depthBytes, countBytes};
            vkCmdCopyBuffer(cmd, m_renderSplats.buffer, rb.buffer, 1, &a);
            vkCmdCopyBuffer(cmd, m_rects.buffer, rb.buffer, 1, &b);
            vkCmdCopyBuffer(cmd, m_depths.buffer, rb.buffer, 1, &c);
            vkCmdCopyBuffer(cmd, m_tileCounts.buffer, rb.buffer, 1, &d);
        }
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* data = static_cast<const uint32_t*>(rb.mapped);
    words.assign(data, data + splatBytes / 4);
    data += splatBytes / 4;
    rects.assign(data, data + rectBytes / 4);
    data += rectBytes / 4;
    depths.assign(data, data + depthBytes / 4);
    data += depthBytes / 4;
    tileCounts.assign(data, data + countBytes / 4);
    destroyBuffer(*m_ctx, rb);
}

void SplatTileRenderer::readBackImage(std::vector<float>& rgba) {
    if (m_settings.format != SplatTargetFormat::Rgba32f || !m_output.image)
        throw std::runtime_error("splat-tiles: readBackImage needs the rgba32f target");
    const VkExtent2D e = m_output.extent;
    const VkDeviceSize bytes = VkDeviceSize(e.width) * e.height * 16;
    Buffer rb = createBuffer(*m_ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback,
                             "splat-tiles: image readback");
    m_ctx->immediateSubmit([&](VkCommandBuffer cmd) {
        // After a frame the output is in TRANSFER_SRC_OPTIMAL (the resolve blit read it).
        imageBarrier(cmd, m_output.image, VK_IMAGE_ASPECT_COLOR_BIT, m_outputState.layout,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {e.width, e.height, 1};
        vkCmdCopyImageToBuffer(cmd, m_output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, 1, &region);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    m_outputState = {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_NONE};
    VK_CHECK(vmaInvalidateAllocation(m_ctx->allocator, rb.allocation, 0, VK_WHOLE_SIZE));
    const auto* src = static_cast<const float*>(rb.mapped);
    rgba.assign(src, src + size_t(e.width) * e.height * 4);
    destroyBuffer(*m_ctx, rb);
}

void SplatTileRenderer::drawUi() {
    ImGui::Text("Target: %s (storage image)  |  sort: %s  |  duplicate: %s  |  radix: %s",
                splatTargetFormatName(m_settings.format), tileSortModeName(m_tileSettings.sortMode),
                tileDuplicateModeName(m_tileSettings.duplicateMode),
                m_tileSettings.radixVariant == GpuRadixSort::Variant::Classic ? "classic" : "reorder");
    const char* bounds[] = {"tile (INRIA)", "radius"};
    int bound = int(m_settings.bound);
    if (ImGui::Combo("Coverage", &bound, bounds, 2)) m_settings.bound = CoverageBound(bound);
    ImGui::SliderInt("Max SH degree", &m_settings.maxShDegree, 0, 3);
    const char* variants[] = {"classic (reference)", "reorder (local sort + coalesced writes)"};
    int variant = int(m_tileSettings.radixVariant);
    if (ImGui::Combo("Radix scatter", &variant, variants, 2))
        m_tileSettings.radixVariant = GpuRadixSort::Variant(variant);
    ImGui::BeginDisabled(m_settings.format != SplatTargetFormat::Rgba16f);
    ImGui::Checkbox("Instrumented render (per-tile stats, slower)", &m_tileSettings.stats);
    ImGui::EndDisabled();
    ImGui::Text("Render splat: %s (%u B per entry fetch)", renderSplatModeName(m_tileSettings.renderSplat),
                renderSplatBytes(m_tileSettings.renderSplat));
    ImGui::Text("Bounds: %s  |  StopThePop window: %u  |  batch %u, vote every %u", tileBoundsName(m_settings.bounds),
                m_tileSettings.sortWindow, renderBatch(), m_tileSettings.voteEvery);
    ImGui::Text("Tiles %u x %u (%ux%u px), list capacity %u entries = %.1f MB (%s, budget %u MB, %u resizes)",
                m_tilesX, m_tilesY, m_tileSettings.tileSize, m_tileSettings.tileSize, capacity(), toMiB(listBytes()),
                m_tileSettings.adaptiveList ? "adaptive" : "fixed", m_tileSettings.budgetMB, m_listResizes);
    if (!m_lastStats.valid) return;
    const TileControl& c = m_lastStats.control;
    const uint32_t tileCount = std::max(1u, m_lastStats.tilesX * m_lastStats.tilesY);
    ImGui::Text("Frame %llu: visible %u of %u, (tile, splat) entries %u (%.1f per tile)",
                static_cast<unsigned long long>(m_lastStats.frameIndex), c.cullCounts[0], c.gaussianCount, c.duplicates,
                double(c.duplicates) / tileCount);
    if (c.overflow > 0 || m_overflowFrames > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f),
                           "LIST OVERFLOW: %u entries dropped this frame (%llu frames so far); raise --tile-budget-mb",
                           c.overflow, static_cast<unsigned long long>(m_overflowFrames));
    if (!m_lastStats.tiles.empty()) {
        uint64_t evaluated = 0, blended = 0, terminated = 0, slots = 0, pixels = 0, full = 0, listed = 0, fetched = 0;
        for (uint32_t t = 0; t < tileCount; ++t) {
            const uint32_t* s = &m_lastStats.tiles[size_t(t) * kTileStatCount];
            evaluated += s[kTileStatEvaluated];
            blended += s[kTileStatBlended];
            terminated += s[kTileStatTerminated];
            slots += s[kTileStatLaneSlots];
            pixels += s[kTileStatPixels];
            full += uint64_t(s[kTileStatListLength]) * s[kTileStatPixels];
            listed += s[kTileStatListLength];
            fetched += s[kTileStatFetched];
        }
        ImGui::Text("Pairs evaluated %.3f G of %.3f G without early stop (%.1f%% skipped)", evaluated * 1e-9, full * 1e-9,
                    full ? 100.0 * (1.0 - double(evaluated) / double(full)) : 0.0);
        ImGui::Text("Blended %.3f G; pixels terminated %.1f%%; SIMD lane use %.1f%%", blended * 1e-9,
                    pixels ? 100.0 * double(terminated) / double(pixels) : 0.0,
                    slots ? 100.0 * double(evaluated) / double(slots) : 0.0);
        ImGui::Text("List entries fetched %.1f%% (whole-tile early exit)",
                    listed ? 100.0 * double(fetched) / double(listed) : 0.0);
    }
}

void SplatTileRenderer::destroy(VulkanContext& ctx) {
    VkDevice device = ctx.device;
    m_scanFlags.destroy(ctx);
    m_scanTiles.destroy(ctx);
    m_depthSort.destroy(ctx);
    if (m_list) destroyList(*m_list);
    m_list.reset();
    for (auto& l : m_retiredLists) destroyList(*l);
    m_retiredLists.clear();
    for (Pass* p : {&m_preprocessPass, &m_compactPass, &m_setupPass, &m_duplicatePass, &m_rangesPass, &m_renderPass}) {
        if (p->pipeline) vkDestroyPipeline(device, p->pipeline, nullptr);
        if (p->layout) vkDestroyPipelineLayout(device, p->layout, nullptr);
        if (p->setLayout) vkDestroyDescriptorSetLayout(device, p->setLayout, nullptr);
        *p = Pass{};
    }
    for (VkPipeline& p : m_renderPipelines) {
        if (p) vkDestroyPipeline(device, p, nullptr);
        p = VK_NULL_HANDLE;
    }
    if (m_renderStatsPipeline) vkDestroyPipeline(device, m_renderStatsPipeline, nullptr);
    m_renderStatsPipeline = VK_NULL_HANDLE;
    m_pool.destroy(device);
    destroyImage(ctx, m_output);
    for (Buffer* b : {&m_splats, &m_renderSplats, &m_rects, &m_depths, &m_rayDepth, &m_dummy, &m_tileCounts, &m_slots,
                      &m_depthKeys, &m_depthValues, &m_offsets, &m_ranges, &m_tileStats, &m_control})
        destroyBuffer(ctx, *b);
    for (Buffer& b : m_readback) destroyBuffer(ctx, b);
    m_readback.clear();
}

}  // namespace gsr
