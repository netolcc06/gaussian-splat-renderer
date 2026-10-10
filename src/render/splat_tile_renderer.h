// SplatTileRenderer (Phase 4, `--renderer splat-tiles`): the INRIA 3DGS rasterizer,
// entirely in compute. No triangles, no fixed-function blending: the screen is cut into
// 16x16-pixel tiles, every splat is duplicated into the tiles its 3-sigma rectangle
// touches, the (tile, splat) pairs are sorted so that each tile's splats are contiguous
// and nearest first, and one workgroup per tile blends them FRONT TO BACK with early
// termination once a pixel is opaque (T < 1e-4), exactly like the CPU reference.
//
// Per frame, all on the GPU, two interchangeable ways to build the sorted list
// (TileSortMode, `--tile-sort`); both give the same per-tile lists, bit for bit:
//
//   binning (default)                          global (INRIA's scheme)
//   ------------------------------------------ -------------------------------------------
//   0 reset control block + tile ranges        0 reset
//   1 preprocess + cull: Splat[i], tileCounts[i] (0 = culled), cull counters (both modes)
//   2 scan of visibility flags over N          2 scan of tileCounts over N -> offsets, M
//     -> slot of each visible splat, V
//   3 compact (index order): (depth, i)
//   4 depth sort, 32-bit keys, V entries
//   5 scan of tileCounts[sorted j] over V
//     -> offsets, M
//   6 setup: entries = min(M, capacity), overflow, ranges dispatch args (both modes)
//   7 duplicate in depth order: key = tile     7 duplicate in index order:
//                                                key = tile << 32 | depth bits
//   8 tile sort: STABLE, log2(#tiles) bits     8 sort: 64-bit keys, 32 + log2(#tiles)
//     (2 passes at 1080p)                        bits (6 passes at 1080p)
//   9 tile ranges [start, end) per tile (both modes, indirect dispatch over the entries)
//  10 render: one workgroup per tile, shared-memory batches, front to back (both modes)
//  11 resolve: blit the float target into the RGBA8 colour target (same as the raster path)
//
// The CPU only records commands. Counts (V, M, entries) only exist on the GPU: scans and
// passes over N are dispatched for N and read their count from the control block; passes
// over the duplicated list are dispatched indirectly. The control block is copied to a
// per-frame-slot host buffer and read two frames later (statistics, overflow warning).
//
// Capacity: the duplicated list is sized from a memory budget (`--tile-budget-mb`). If a
// frame produces more entries than fit, the extra ones are dropped (binning mode drops
// the farthest splats' entries, global mode the highest gaussian indices), the overflow is
// counted, and a warning is shown; nothing is written out of bounds.
//
// Phase 5:
//   - `--render-splat p32|p24`: the preprocess writes a packed 32- or 24-byte render splat
//     (shaders/render_splat.glsl) plus the tile rectangle and depth in their own arrays,
//     instead of the 48-byte Splat; duplicate/compact/render read only what they need.
//   - `--tile-list adaptive`: the list capacity follows the measured entry count M (read
//     back two frames late, no stall): grow to 1.5 M when M > 90% of the capacity, shrink
//     to 1.5 x the recent maximum after 120 frames below 40%; the budget is the upper
//     limit. The old list stays alive until the frames that used it have completed.
//
// Phase 6:
//   - `--bounds tight|cutoff-only` (SplatSettings::bounds): a splat is binned only into the
//     tiles its alpha >= 1/255 ellipse overlaps (shaders/tile_coverage.glsl), counted in the
//     preprocess and enumerated row by row in the duplicate pass. tight is exact (same image).
//   - `--tile-size 8|16`: the binning/render tile (a divisor of INRIA's 16-px getRect tiles).
//   - `--sort-window K`: StopThePop-style per-pixel re-sorting window in the render pass.
//   - render-loop experiments (batch size, vote granularity, shared layout, fp16 math).
#pragma once

#include "core/descriptors.h"
#include "render/gpu_radix_sort.h"
#include "render/gpu_scan.h"
#include "render/renderer.h"
#include "render/splat_draw.h"
#include "splat/splat_math.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace gsr {

struct GpuScene;
struct GaussianScene;

enum class TileSortMode : uint32_t {
    Binning = 0,  // depth sort of the visible splats, then a stable sort of tile ids
    Global = 1,   // one sort of 64-bit (tile | depth) keys, as INRIA
};
const char* tileSortModeName(TileSortMode m);

enum class TileDuplicateMode : uint32_t {
    PerSplat = 0,  // one thread per splat loops over its tiles (INRIA duplicateWithKeys)
    PerEntry = 1,  // one thread per output entry, binary search in the offsets (load-balanced)
};
const char* tileDuplicateModeName(TileDuplicateMode m);

// What the duplicate and render passes read per (tile, splat) entry (shaders/render_splat.glsl).
enum class RenderSplatMode : uint32_t {
    Full = 0,      // the 48-byte Splat (Phase 4)
    Packed32 = 1,  // 32 bytes: fp32 centre/conic/opacity, fp16 colour, 16-bit radius
    Packed24 = 2,  // 24 bytes: fp32 centre, fp16 conic (x 2^12) / opacity / colour, 16-bit radius
};
const char* renderSplatModeName(RenderSplatMode m);
uint32_t renderSplatBytes(RenderSplatMode m);

struct TileSettings {
    TileSortMode sortMode = TileSortMode::Binning;
    TileDuplicateMode duplicateMode = TileDuplicateMode::PerEntry;
    GpuRadixSort::Variant radixVariant = GpuRadixSort::Variant::Reorder;
    uint32_t budgetMB = 256;  // device memory for the duplicated (tile, splat) list incl. sort scratch
                              // (adaptive list: the upper limit)
    bool stats = false;       // instrumented render shader + per-tile statistics readback (never timed)
    RenderSplatMode renderSplat = RenderSplatMode::Full;
    bool adaptiveList = false;  // size the list from the measured entry count (Phase 5)
    uint32_t adaptiveInitialEntries = 0;  // adaptive: first capacity (0 = max(4 N, 1 M) entries; tests)
    // Phase 6. The bounds mode itself is SplatSettings::bounds (shared with the raster path).
    uint32_t tileSize = 16;     // binning + render tile: 16 (INRIA) or 8
    uint32_t sortWindow = 0;    // StopThePop per-pixel re-sorting window (0 = off: INRIA order)
    // Render-loop experiments (docs/notes/phase6.md "Profiling"); defaults = Phase 4 loop.
    uint32_t renderBatch = 0;   // splats per shared-memory batch (0 = one per thread)
    uint32_t voteEvery = 1;     // whole-tile early-exit vote every N batches (0 = never)
    bool sharedAoS = false;     // one struct array in shared memory instead of one per field
    bool fp16Math = false;      // exp/alpha in fp16 (needs shaderFloat16)
};

// GPU control block (C++ twin of TILE_CONTROL_MEMBERS in shaders/tile_common.glsl).
struct TileControl {
    uint32_t gaussianCount;  // N
    uint32_t visible;        // V (binning mode; global mode leaves it 0, see cullCounts[0])
    uint32_t duplicates;     // M, unclamped
    uint32_t entries;        // min(M, capacity)
    uint32_t cullCounts[8];  // byte offset 16, as in SplatControl
    uint32_t overflow;       // M - entries
    VkDispatchIndirectCommand rangesArgs;
    uint32_t coverageMismatch;  // Phase 6: splats whose duplicated tiles != their count (always 0)
    uint32_t pad[3];
};
static_assert(sizeof(TileControl) == 80, "layout must match shaders/tile_common.glsl");

// Per-tile counters of the stats build (shaders/tile_render.comp, kStat*).
enum TileStat : uint32_t {
    kTileStatListLength = 0,  // entries in the tile's range (splats binned to the tile)
    kTileStatFetched,         // entries loaded before the whole tile stopped
    kTileStatEvaluated,       // (pixel, splat) pairs evaluated
    kTileStatBlended,         // pairs blended
    kTileStatTerminated,      // pixels that reached T < 1e-4
    kTileStatLaneSlots,       // SIMD lane slots spent in the blend loop
    kTileStatMaxDepth,        // max evaluated count of a pixel
    kTileStatPixels,          // pixels of the tile inside the image
    kTileStatCount
};

class SplatTileRenderer final : public Renderer {
public:
    void init(VulkanContext& ctx, const GpuScene& gpuScene, uint32_t framesInFlight, const SplatSettings& settings,
              const TileSettings& tileSettings);
    const char* name() const override { return "splat-tiles"; }
    void onRenderTargetsChanged(VulkanContext& ctx, const Image& color, const Image& depth) override;
    RenderOutput render(const FrameInput& in) override;
    void drawUi() override;
    void destroy(VulkanContext& ctx) override;

    const SplatSettings& settings() const { return m_settings; }
    const TileSettings& tileSettings() const { return m_tileSettings; }
    int effectiveShDegree() const;
    uint32_t capacity() const { return m_list ? m_list->capacity : 0; }
    uint32_t maxCapacity() const { return m_maxCapacity; }
    uint32_t listResizes() const { return m_listResizes; }
    VkDeviceSize listBytes() const;   // duplicated list + its sort scratch
    VkDeviceSize totalBytes() const;  // every buffer this renderer owns (+ the output image)
    uint32_t tilesX() const { return m_tilesX; }
    uint32_t tilesY() const { return m_tilesY; }
    uint32_t tileSize() const { return m_tileSettings.tileSize; }
    uint32_t renderBatch() const;     // effective splats per batch
    uint64_t coverageMismatches() const { return m_coverageMismatches; }  // summed over read frames

    // Statistics of the most recent frame whose readback is complete (2 frames old, or
    // after collectPending() once the GPU is idle).
    struct FrameStats {
        bool valid = false;
        uint64_t frameIndex = 0;
        TileControl control{};
        std::vector<uint32_t> tiles;  // stats mode: tilesX * tilesY * kTileStatCount
        uint32_t tilesX = 0, tilesY = 0;
    };
    const FrameStats& lastStats() const { return m_lastStats; }
    // Reads every completed slot (call with the GPU idle, e.g. at the end of a run).
    void collectPending();
    uint64_t overflowFrames() const { return m_overflowFrames; }
    uint32_t maxOverflow() const { return m_maxOverflow; }
    uint32_t maxDuplicates() const { return m_maxDuplicates; }

    // Testing/debugging (GPU must be idle): the sorted list (entries), tile ranges, control.
    void readBack(std::vector<uint32_t>& tileKeys, std::vector<uint32_t>& values, std::vector<uint32_t>& ranges,
                  TileControl& control);
    // Testing (GPU idle): the per-gaussian Splat buffer and touched-tile counts (index order).
    // Full render-splat mode only.
    void readBackSplats(std::vector<ProjectedSplat>& splats, std::vector<uint32_t>& tileCounts);
    // Testing (GPU idle), packed modes: raw render-splat words (8 or 6 per gaussian), packed
    // tile rectangles (2 words), depth bits, touched-tile counts.
    void readBackPackedSplats(std::vector<uint32_t>& words, std::vector<uint32_t>& rects, std::vector<uint32_t>& depths,
                              std::vector<uint32_t>& tileCounts);
    // Testing (GPU idle, rgba32f target only): the float output image, RGBA, row 0 = top.
    void readBackImage(std::vector<float>& rgba);
    // Pipelines of this renderer by name (Phase 6 `--pipeline-stats`).
    std::vector<std::pair<std::string, VkPipeline>> pipelines() const override;

private:
    struct Pass {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    // The duplicated (tile, splat) list, its tile sort and the descriptor sets that point at
    // it. One object per capacity: the adaptive list replaces it while the old one may still
    // be in use by frames in flight (destroyed once they completed).
    struct ListResources {
        uint32_t capacity = 0;
        Buffer keys;     // binning: tile id; global: low word (depth bits)
        Buffer keysHi;   // global: high word (tile id)
        Buffer values;   // gaussian index
        GpuRadixSort sort;
        DescriptorPool pool;
        VkDescriptorSet duplicateSet = VK_NULL_HANDLE, rangesSet = VK_NULL_HANDLE, renderSet = VK_NULL_HANDLE;
        uint64_t lastUsedFrame = 0;
    };

    void createOutputTarget(VkExtent2D extent);
    void readSlot(uint32_t slot);
    std::unique_ptr<ListResources> createList(uint32_t capacity);
    void destroyList(ListResources& list);
    void writeListSets(ListResources& list);  // duplicate set; ranges/render sets once the tile buffers exist
    void adaptCapacity(uint64_t frameIndex);  // adaptive list: maybe replace m_list (no stall)
    uint32_t capacityForBytes(double bytes) const;

    VulkanContext* m_ctx = nullptr;
    const GpuScene* m_gpuScene = nullptr;
    SplatSettings m_settings;
    TileSettings m_tileSettings;
    uint32_t m_framesInFlight = 2;
    uint32_t m_maxCapacity = 0;  // capacity of the full budget
    uint32_t m_listResizes = 0;
    uint32_t m_shrinkFrames = 0, m_shrinkWindowMax = 0;  // adaptive shrink hysteresis
    uint64_t m_lastAdaptedFrame = ~0ull;  // frame index of the readback last used to adapt
    uint32_t m_maxTiles = 0;  // tile buffers' size (grows with the render size)
    uint32_t m_tilesX = 0, m_tilesY = 0;

    // Per-gaussian buffers (N).
    Buffer m_splats;       // full mode: N x 48 B (struct Splat)
    Buffer m_renderSplats; // packed modes: N x 32 / 24 B
    Buffer m_rects;        // packed modes: N x 8 B tile rectangles
    Buffer m_depths;       // packed modes: N x 4 B depth bits
    Buffer m_rayDepth;     // StopThePop: N x 48 B ray-depth parameters
    Buffer m_dummy;        // bound to the bindings a mode does not use (never accessed)
    Buffer m_tileCounts;   // N uints: tiles touched (0 = culled)
    Buffer m_slots;        // binning: scan of the visibility flags (N)
    Buffer m_depthKeys;    // binning: compact list, depth keys (N), sorted in place
    Buffer m_depthValues;  // binning: compact list, gaussian indices (N), sorted in place
    Buffer m_offsets;      // scan of the touched-tile counts (N)
    // Duplicated list (capacity).
    std::unique_ptr<ListResources> m_list;
    std::vector<std::unique_ptr<ListResources>> m_retiredLists;
    // Per tile.
    Buffer m_ranges;       // uvec2 per tile
    Buffer m_tileStats;    // stats mode: kTileStatCount uints per tile
    Buffer m_control;      // TileControl
    std::vector<Buffer> m_readback;  // per frame slot: TileControl (+ tile stats)
    struct SlotInfo {
        bool used = false;
        uint64_t frameIndex = 0;
        uint32_t tilesX = 0, tilesY = 0;
    };
    std::vector<SlotInfo> m_slotInfo;

    GpuScan m_scanFlags;    // binning: visibility flags -> slots, total V
    GpuScan m_scanTiles;    // tile counts -> offsets, total M
    GpuRadixSort m_depthSort;

    DescriptorPool m_pool;
    Pass m_preprocessPass, m_compactPass, m_setupPass, m_duplicatePass, m_rangesPass, m_renderPass;
    std::array<VkPipeline, 3> m_renderPipelines{};  // per SplatTargetFormat (null: variant not built)
    VkPipeline m_renderStatsPipeline = VK_NULL_HANDLE;
    std::array<std::string, 3> m_renderShaderNames;
    std::string m_renderStatsShaderName;
    std::string m_duplicateShaderName;

    // Output: float storage image (rgba16f/32f) blitted into the colour target, or the
    // colour target itself (rgba8).
    Image m_output;
    ImageState m_outputState{VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    VkImageView m_boundView = VK_NULL_HANDLE;  // image currently bound to the render set

    FrameStats m_lastStats;
    uint64_t m_overflowFrames = 0;
    uint32_t m_maxOverflow = 0;
    uint32_t m_maxDuplicates = 0;
    bool m_overflowWarned = false;
    uint64_t m_coverageMismatches = 0;
};

}  // namespace gsr
