// WgpuTileRenderer (web phase W2, renderer "splat-tiles"): the web port of src/render/splat_tile_renderer.*
// in its default configuration: tight bounds, binning sort mode, per-entry duplication, adaptive list,
// 16x16 tiles, full 48-byte splats, rgba16float target, front to back with INRIA's early termination.
// Everything runs on the GPU; the CPU only records commands:
//
//   0 reset: control block {N, 0, ...} (queue.writeBuffer), tile ranges cleared
//   1 preprocess + cull: Splat[i], tileCounts[i] (0 = culled), coverage spans[i], cull counters
//   2 scan of the visibility flags over N -> slot of each visible splat, V
//   3 compact (index order): (depth bits, i)
//   4 depth sort, 32-bit keys, V entries (8 passes of 4 bits)
//   5 scan of tileCounts[sorted j] over V -> offsets, M
//   6 setup: entries = min(M, capacity), overflow, indirect arguments of 7 and 9 (2D-clamped)
//   7 duplicate, one thread per entry: key = tile from the STORED spans (no float math)
//   8 tile sort: STABLE, bit_width(#tiles) bits (4 passes at 1080p)
//   9 tile ranges [start, end) per tile
//  10 render: one workgroup per tile, shared-memory batches, front to back, whole-tile exit
// then the app's resolve pass copies the float target to the canvas (as W0/W1).
//
// Each stage is its own compute pass (GPU timestamps at pass boundaries, when available). The
// control block is copied into a ring of MAP_READ buffers and read asynchronously (no stall): it
// feeds the statistics, the overflow / mismatch counters and the adaptive list capacity, like
// native's 2-frame-late readback.
//
// Differences from native, all explained in docs/web/W2.md: coverage spans stored as integers (no
// `precise`), a 4-bit portable radix sort, 2D-clamped indirect grids, separate indirect argument
// buffers (WebGPU usage rules), and subgroup variants chosen at run time (identical images).
#pragma once

#include "splat/splat_math.h"
#include "webgpu/shader_constants.h"
#include "webgpu/wgpu_compute.h"
#include "webgpu/wgpu_radix_sort.h"
#include "webgpu/wgpu_resources.h"
#include "webgpu/wgpu_scan.h"

#include <webgpu/webgpu.h>

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gsr {

class WgslShaderCache;
class WgpuScene;

struct WgpuTileSettings {
    bool subgroups = false;       // subgroup variants (only if the device has the feature)
    uint32_t budgetMB = 256;      // upper limit of the duplicated list incl. its sort scratch
    bool adaptiveList = true;     // capacity from the measured entry count (native --tile-list adaptive)
    uint32_t fixedCapacity = 0;   // > 0: exactly this many entries, never adapted (tests)
    uint32_t maxGroupsX = 0;      // workgroups per dimension (0 = the device limit; tests: smaller)
    uint32_t boundsMode = 1;      // 0 rect, 1 tight (tile_coverage.wgsl)
    int maxShDegree = 3;
};

class WgpuTileRenderer {
public:
    WgpuTileRenderer() = default;
    WgpuTileRenderer(const WgpuTileRenderer&) = delete;
    WgpuTileRenderer& operator=(const WgpuTileRenderer&) = delete;
    ~WgpuTileRenderer() { release(); }

    bool create(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders, WGPUBindGroupLayout sceneLayout,
                const WGPULimits& limits, const WgpuTileSettings& settings);
    void release();
    bool valid() const { return render16_ != nullptr; }
    const WgpuTileSettings& settings() const { return settings_; }

    // Records one frame into `encoder`: `scene` seen by `cam`, written to `target` (rgba16float, or
    // rgba32float for tests). `timestamps` (optional) times each pass. Call afterSubmit() once the
    // encoder's command buffer is submitted.
    void encode(WGPUCommandEncoder encoder, const WgpuScene& scene, const WgpuTexture2D& target, const SplatCamera& cam,
                uint64_t frameIndex, WgpuTimestamps* timestamps = nullptr);
    void afterSubmit();

    // Statistics from the asynchronous readbacks of the control block.
    struct FrameStats {
        bool valid = false;
        uint64_t frameIndex = 0;
        web::WebTileControl control{};
        uint32_t capacity = 0;
    };
    const FrameStats& lastStats() const { return lastStats_; }
    std::optional<FrameStats> statsForFrame(uint64_t frameIndex) const;
    std::string statsJson() const;

    uint32_t capacity() const { return list_ ? list_->capacity : 0; }
    uint32_t maxCapacity() const { return maxCapacity_; }
    uint64_t listBytes() const;
    uint64_t totalBytes() const;
    uint32_t tilesX() const { return tilesX_; }
    uint32_t tilesY() const { return tilesY_; }

    // Tests (GPU readback by the caller): the renderer's buffers after a frame.
    struct DebugBuffers {
        WGPUBuffer splats, tileCounts, spans, slots, depthKeys, depthValues, offsets, listKeys, listValues, ranges, control;
        uint32_t count, capacity, tilesX, tilesY;
    };
    DebugBuffers debugBuffers() const;

private:
    struct ListResources {
        uint32_t capacity = 0;
        WgpuBuffer keys, values;
        WgpuRadixSort sort;
        WGPUBindGroup duplicateGroup = nullptr, rangesGroup = nullptr, renderGroup = nullptr;
        uint64_t renderTargetGeneration = 0;
        ~ListResources();
    };

    void ensureScene(const WgpuScene& scene);
    void ensureTiles(uint32_t width, uint32_t height);
    std::unique_ptr<ListResources> createList(uint32_t capacity);
    void ensureListGroups(ListResources& list, const WgpuTexture2D& target);
    void adaptCapacity();
    void onControl(const uint8_t* data, uint64_t frameIndex);
    WGPUComputePipeline preprocessPipeline(const WgpuScene& scene);
    uint32_t capacityForBytes(double bytes) const;

    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WgslShaderCache* shaders_ = nullptr;
    WGPUBindGroupLayout sceneLayout_ = nullptr;  // owned by the caller
    WgpuTileSettings settings_;
    uint32_t maxGroupsX_ = 65535;
    uint64_t maxBindingBytes_ = 0;
    uint32_t maxCapacity_ = 0;

    // Pipelines and layouts.
    WgpuScanKernels scanKernels_;
    WgpuSortKernels sortKernels_;
    WGPUShaderModule preprocessModule_ = nullptr;  // owned by the shader cache
    WGPUBindGroupLayout preprocessLayout_ = nullptr, compactLayout_ = nullptr, setupLayout_ = nullptr,
                        duplicateLayout_ = nullptr, rangesLayout_ = nullptr, render16Layout_ = nullptr,
                        render32Layout_ = nullptr;
    WGPUPipelineLayout preprocessPipelineLayout_ = nullptr;
    std::map<std::string, WGPUComputePipeline> preprocessPipelines_;  // per scene decoder variant
    WGPUComputePipeline compact_ = nullptr, setup_ = nullptr, duplicate_ = nullptr, ranges_ = nullptr,
                        render16_ = nullptr, render32_ = nullptr;

    // Uniforms.
    WgpuBuffer preprocessParams_, setupParams_, duplicateParams_, rangesParams_, renderParams_;

    // Per-gaussian buffers (N) and the bind groups that use them.
    const WgpuScene* scene_ = nullptr;
    uint32_t count_ = 0;
    WgpuBuffer splats_, tileCounts_, spans_, slots_, depthKeys_, depthValues_, offsets_;
    WgpuBuffer control_, listArgs_, mismatch_;
    WgpuScan scanFlags_, scanTiles_;
    WgpuRadixSort depthSort_;
    WGPUBindGroup preprocessGroup_ = nullptr, compactGroup_ = nullptr, setupGroup_ = nullptr;

    // Per tile.
    uint32_t tilesX_ = 0, tilesY_ = 0, width_ = 0, height_ = 0;
    WgpuBuffer ranges_buffer_;

    // The duplicated list (replaced by the adaptive capacity; the old one is released, and work
    // already submitted keeps it alive until it completes).
    std::unique_ptr<ListResources> list_;
    uint32_t listResizes_ = 0;
    uint32_t shrinkFrames_ = 0, shrinkWindowMax_ = 0;
    uint64_t lastAdaptedFrame_ = ~0ull;

    // Readback of the control block.
    WgpuReadbackRing controlRing_;
    int pendingSlot_ = -1;
    uint64_t pendingFrame_ = 0;
    FrameStats lastStats_;
    std::deque<FrameStats> history_;  // the last few frames (tests: the captured frame's counters)
    uint64_t framesRead_ = 0, overflowFrames_ = 0, mismatchTotal_ = 0;
    uint32_t maxOverflow_ = 0, maxDuplicates_ = 0;
    std::vector<std::array<uint64_t, 5>> mismatchFrames_;  // diagnostics: frame, count, example
};

}  // namespace gsr
