#include "webgpu/wgpu_tile_renderer.h"

#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_scene.h"
#include "webgpu/wgpu_shaders.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>

namespace gsr {

namespace {

// Must match PreprocessParams in shaders/wgsl/splat_preprocess.wgsl (128 bytes).
struct PreprocessParams {
    glm::mat4 viewInria;
    glm::vec4 camPos;
    glm::vec2 focal;
    glm::vec2 tanHalfFov;
    glm::uvec2 imageSize;
    uint32_t count;
    uint32_t shDegree;
    glm::uvec2 tiles16;
    glm::uvec2 grid;
};
static_assert(sizeof(PreprocessParams) == 128, "layout must match splat_preprocess.wgsl");

struct SetupParams {  // tile_setup.wgsl
    uint32_t capacity, maxGroupsX, pad0, pad1;
};
struct DuplicateParams {  // tile_duplicate.wgsl
    uint32_t tilesX, tileCount, pad0, pad1;
};
struct RangesParams {  // tile_ranges.wgsl
    uint32_t tileCount, pad0, pad1, pad2;
};
struct RenderParams {  // tile_render.wgsl
    uint32_t width, height, tilesX, pad0;
};

constexpr uint32_t kMinAdaptiveCapacity = 256u * 1024u;  // entries (native: the same)
constexpr size_t kHistory = 64;  // frames of counters kept (gsr_tiles_frame_json)

uint32_t groupsFor(uint32_t n) { return (n + web::kGroupSize - 1) / web::kGroupSize; }

uint32_t roundUpToBlock(uint64_t n) {
    const uint64_t b = web::kSortBlockSize;
    return uint32_t(std::min<uint64_t>((n + b - 1) / b * b, 0xFFFFFC00ull));
}

// Key bits for tile ids 0 .. tileCount (tileCount itself = the sentinel).
uint32_t tileKeyBits(uint32_t tileCount) { return std::max(1u, uint32_t(std::bit_width(tileCount))); }

void releaseGroup(WGPUBindGroup& g) {
    if (g) wgpuBindGroupRelease(g);
    g = nullptr;
}

}  // namespace

WgpuTileRenderer::ListResources::~ListResources() {
    releaseGroup(duplicateGroup);
    releaseGroup(rangesGroup);
    releaseGroup(renderGroup);
    sort.release();
    keys.release();
    values.release();
}

uint32_t WgpuTileRenderer::capacityForBytes(double bytes) const {
    // Per entry: key + value, the sort's ping-pong copy of each, and its block histograms
    // (16 counts per 1024 keys).
    const double bytesPerEntry = 16.0 + 64.0 / web::kSortBlockSize;
    const double maxByBinding = double(maxBindingBytes_) / 4.0;
    uint32_t c = uint32_t(std::clamp(bytes / bytesPerEntry, double(web::kSortBlockSize), maxByBinding));
    return c - c % web::kSortBlockSize;
}

bool WgpuTileRenderer::create(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders,
                              WGPUBindGroupLayout sceneLayout, const WGPULimits& limits,
                              const WgpuTileSettings& settings) {
    release();
    device_ = device;
    queue_ = queue;
    shaders_ = &shaders;
    sceneLayout_ = sceneLayout;
    settings_ = settings;
    maxGroupsX_ = settings.maxGroupsX ? settings.maxGroupsX : std::max(1u, limits.maxComputeWorkgroupsPerDimension);
    maxBindingBytes_ = std::min<uint64_t>(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    maxCapacity_ = settings.fixedCapacity ? roundUpToBlock(settings.fixedCapacity)
                                          : capacityForBytes(double(std::max(settings.budgetMB, 1u)) * 1048576.0);
    const bool sg = settings.subgroups;

    if (!scanKernels_.create(device, shaders, sg) || !sortKernels_.create(device, shaders, sg)) return false;

    std::vector<std::string> sgDefine;
    if (sg) sgDefine.push_back("GSR_SUBGROUPS");
    preprocessModule_ = shaders.get(device, {"splat_preprocess.wgsl", sgDefine});
    WGPUShaderModule compactModule = shaders.get(device, {"tile_compact.wgsl", {}});
    WGPUShaderModule setupModule = shaders.get(device, {"tile_setup.wgsl", {}});
    WGPUShaderModule duplicateModule = shaders.get(device, {"tile_duplicate.wgsl", {}});
    WGPUShaderModule rangesModule = shaders.get(device, {"tile_ranges.wgsl", {}});
    WGPUShaderModule render16Module = shaders.get(device, {"tile_render.wgsl", sgDefine});
    std::vector<std::string> render32Defines = sgDefine;
    render32Defines.push_back("GSR_OUT_RGBA32F");
    WGPUShaderModule render32Module = shaders.get(device, {"tile_render.wgsl", render32Defines});
    if (!preprocessModule_ || !compactModule || !setupModule || !duplicateModule || !rangesModule || !render16Module ||
        !render32Module)
        return false;

    using K = BindKind;
    preprocessLayout_ = makeComputeLayout(
        device, {{0, K::Uniform}, {1, K::Storage}, {2, K::Storage}, {3, K::Storage}, {4, K::Storage}}, "preprocess");
    compactLayout_ = makeComputeLayout(device,
                                       {{0, K::ReadOnlyStorage},
                                        {1, K::ReadOnlyStorage},
                                        {2, K::ReadOnlyStorage},
                                        {3, K::Storage},
                                        {4, K::Storage},
                                        {5, K::ReadOnlyStorage}},
                                       "compact");
    setupLayout_ = makeComputeLayout(device, {{0, K::Storage}, {1, K::Uniform}, {2, K::Storage}}, "tile setup");
    duplicateLayout_ = makeComputeLayout(device,
                                         {{0, K::ReadOnlyStorage},
                                          {1, K::ReadOnlyStorage},
                                          {2, K::ReadOnlyStorage},
                                          {3, K::ReadOnlyStorage},
                                          {4, K::Storage},
                                          {5, K::Storage},
                                          {6, K::ReadOnlyStorage},
                                          {7, K::Uniform},
                                          {8, K::Storage}},
                                         "duplicate");
    rangesLayout_ = makeComputeLayout(
        device, {{0, K::ReadOnlyStorage}, {1, K::Storage}, {2, K::ReadOnlyStorage}, {3, K::Uniform}}, "tile ranges");
    auto renderLayout = [&](BindKind texture, const char* label) {
        return makeComputeLayout(device,
                                 {{0, K::ReadOnlyStorage},
                                  {1, K::ReadOnlyStorage},
                                  {2, K::ReadOnlyStorage},
                                  {3, texture},
                                  {4, K::Uniform}},
                                 label);
    };
    render16Layout_ = renderLayout(K::StorageTexture16F, "tile render rgba16f");
    render32Layout_ = renderLayout(K::StorageTexture32F, "tile render rgba32f");

    preprocessPipelineLayout_ = makePipelineLayout(device, {sceneLayout_, preprocessLayout_}, "preprocess");
    auto single = [&](WGPUBindGroupLayout l, WGPUShaderModule m, const char* label) {
        WGPUPipelineLayout pl = makePipelineLayout(device, {l}, label);
        WGPUComputePipeline p = makeComputePipeline(device, pl, m, "main", {}, label);
        wgpuPipelineLayoutRelease(pl);
        return p;
    };
    compact_ = single(compactLayout_, compactModule, "tile compact");
    setup_ = single(setupLayout_, setupModule, "tile setup");
    duplicate_ = single(duplicateLayout_, duplicateModule, "tile duplicate");
    ranges_ = single(rangesLayout_, rangesModule, "tile ranges");
    render16_ = single(render16Layout_, render16Module, "tile render rgba16f");
    render32_ = single(render32Layout_, render32Module, "tile render rgba32f");

    const WGPUBufferUsage uniform = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    preprocessParams_.create(device, sizeof(PreprocessParams), uniform, "preprocess params");
    setupParams_.create(device, sizeof(SetupParams), uniform, "tile setup params");
    duplicateParams_.create(device, sizeof(DuplicateParams), uniform, "duplicate params");
    rangesParams_.create(device, sizeof(RangesParams), uniform, "tile ranges params");
    renderParams_.create(device, sizeof(RenderParams), uniform, "tile render params");

    controlRing_.create(device, sizeof(web::WebTileControl), 8, "tile control readback",
                        [this](const uint8_t* data, uint64_t, uint64_t tag) { onControl(data, tag); });
    return compact_ && setup_ && duplicate_ && ranges_ && render16_ && render32_;
}

void WgpuTileRenderer::release() {
    controlRing_.release();
    list_.reset();
    scanFlags_.release();
    scanTiles_.release();
    depthSort_.release();
    for (WGPUBindGroup* g : {&preprocessGroup_, &compactGroup_, &setupGroup_}) releaseGroup(*g);
    for (WgpuBuffer* b : {&splats_, &tileCounts_, &spans_, &slots_, &depthKeys_, &depthValues_, &offsets_, &control_,
                          &listArgs_, &mismatch_, &ranges_buffer_, &preprocessParams_, &setupParams_, &duplicateParams_,
                          &rangesParams_, &renderParams_})
        b->release();
    for (auto& [key, p] : preprocessPipelines_)
        if (p) wgpuComputePipelineRelease(p);
    preprocessPipelines_.clear();
    for (WGPUComputePipeline* p : {&compact_, &setup_, &duplicate_, &ranges_, &render16_, &render32_}) {
        if (*p) wgpuComputePipelineRelease(*p);
        *p = nullptr;
    }
    if (preprocessPipelineLayout_) wgpuPipelineLayoutRelease(preprocessPipelineLayout_);
    preprocessPipelineLayout_ = nullptr;
    for (WGPUBindGroupLayout* l : {&preprocessLayout_, &compactLayout_, &setupLayout_, &duplicateLayout_, &rangesLayout_,
                                   &render16Layout_, &render32Layout_}) {
        if (*l) wgpuBindGroupLayoutRelease(*l);
        *l = nullptr;
    }
    scanKernels_.release();
    sortKernels_.release();
    scene_ = nullptr;
    count_ = 0;
    tilesX_ = tilesY_ = width_ = height_ = 0;
    lastStats_ = {};
    history_.clear();
    pendingSlot_ = -1;
}

WGPUComputePipeline WgpuTileRenderer::preprocessPipeline(const WgpuScene& scene) {
    const std::string key = scene.decoderKey();
    if (auto it = preprocessPipelines_.find(key); it != preprocessPipelines_.end()) return it->second;
    // The scene format is fixed per pipeline (scene_decode.wgsl overrides), plus the bounds mode.
    std::vector<WGPUConstantEntry> constants = scene.decodeConstants();
    WGPUConstantEntry bounds = WGPU_CONSTANT_ENTRY_INIT;
    bounds.key = toStringView("kBoundsMode");
    bounds.value = double(settings_.boundsMode);
    constants.push_back(bounds);
    WGPUComputePipeline p =
        makeComputePipeline(device_, preprocessPipelineLayout_, preprocessModule_, "main", constants, "splat preprocess");
    preprocessPipelines_[key] = p;
    return p;
}

void WgpuTileRenderer::ensureScene(const WgpuScene& scene) {
    if (scene_ == &scene && count_ == scene.count && splats_) return;
    scene_ = &scene;
    count_ = scene.count;
    const uint64_t n = std::max(scene.count, 1u);
    const WGPUBufferUsage storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    splats_.create(device_, n * sizeof(ProjectedSplat), storage, "splat-tiles: splats");
    tileCounts_.create(device_, n * 4, storage, "splat-tiles: tiles touched");
    spans_.create(device_, n * web::kCoverageSpanWords * 4, storage, "splat-tiles: coverage spans");
    slots_.create(device_, n * 4, storage, "splat-tiles: compact slots (scan)");
    depthKeys_.create(device_, n * 4, storage, "splat-tiles: depth keys");
    depthValues_.create(device_, n * 4, storage, "splat-tiles: depth-sorted indices");
    offsets_.create(device_, n * 4, storage, "splat-tiles: duplicate offsets (scan)");
    control_.create(device_, sizeof(web::WebTileControl),
                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc, "splat-tiles: control");
    listArgs_.create(device_, 16, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, "splat-tiles: list dispatch args");
    mismatch_.create(device_, 16, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
                     "splat-tiles: coverage mismatch counter");

    WgpuScan::Bindings flags;
    flags.input = tileCounts_.buffer;
    flags.output = slots_.buffer;
    flags.control = control_.buffer;
    flags.countOffset = 0;  // N
    flags.totalOffset = 1;  // V
    flags.flags = WgpuScan::kNonZero;
    scanFlags_.create(device_, queue_, scanKernels_, uint32_t(n), flags, maxGroupsX_);
    WgpuScan::Bindings tiles;
    tiles.input = tileCounts_.buffer;
    tiles.index = depthValues_.buffer;  // the j-th nearest visible splat's count
    tiles.output = offsets_.buffer;
    tiles.control = control_.buffer;
    tiles.countOffset = 1;  // V
    tiles.totalOffset = 2;  // M
    tiles.flags = WgpuScan::kGather;
    scanTiles_.create(device_, queue_, scanKernels_, uint32_t(n), tiles, maxGroupsX_);
    depthSort_.create(device_, queue_, sortKernels_, uint32_t(n), depthKeys_.buffer, depthValues_.buffer,
                      control_.buffer, 1 /* V */, maxGroupsX_);

    releaseGroup(preprocessGroup_);
    releaseGroup(compactGroup_);
    releaseGroup(setupGroup_);
    preprocessGroup_ = makeBindGroup(device_, preprocessLayout_,
                                     {bindBuffer(0, preprocessParams_.buffer), bindBuffer(1, splats_.buffer),
                                      bindBuffer(2, tileCounts_.buffer), bindBuffer(3, spans_.buffer),
                                      bindBuffer(4, control_.buffer)},
                                     "preprocess");
    compactGroup_ = makeBindGroup(device_, compactLayout_,
                                  {bindBuffer(0, splats_.buffer), bindBuffer(1, tileCounts_.buffer),
                                   bindBuffer(2, slots_.buffer), bindBuffer(3, depthKeys_.buffer),
                                   bindBuffer(4, depthValues_.buffer), bindBuffer(5, control_.buffer)},
                                  "compact");
    setupGroup_ = makeBindGroup(device_, setupLayout_,
                                {bindBuffer(0, control_.buffer), bindBuffer(1, setupParams_.buffer),
                                 bindBuffer(2, listArgs_.buffer)},
                                "tile setup");

    // A new scene: a new list (its duplicate bind group points at the per-gaussian buffers).
    uint32_t initial = maxCapacity_;
    if (settings_.adaptiveList && !settings_.fixedCapacity) {
        // A first guess (4 entries per gaussian); the measured count corrects it a few frames later.
        initial = std::min(maxCapacity_, roundUpToBlock(std::max<uint64_t>(4ull * scene.count, 1u << 20)));
    }
    list_ = createList(initial);
    shrinkFrames_ = shrinkWindowMax_ = 0;
    lastAdaptedFrame_ = ~0ull;
}

std::unique_ptr<WgpuTileRenderer::ListResources> WgpuTileRenderer::createList(uint32_t capacity) {
    auto list = std::make_unique<ListResources>();
    list->capacity = capacity;
    const WGPUBufferUsage storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    list->keys.create(device_, uint64_t(capacity) * 4, storage, "splat-tiles: list keys (tile)");
    list->values.create(device_, uint64_t(capacity) * 4, storage, "splat-tiles: list values (gaussian)");
    list->sort.create(device_, queue_, sortKernels_, capacity, list->keys.buffer, list->values.buffer, control_.buffer,
                      3 /* entries */, maxGroupsX_);
    list->duplicateGroup = makeBindGroup(
        device_, duplicateLayout_,
        {bindBuffer(0, depthValues_.buffer), bindBuffer(1, tileCounts_.buffer), bindBuffer(2, offsets_.buffer),
         bindBuffer(3, spans_.buffer), bindBuffer(4, list->keys.buffer), bindBuffer(5, list->values.buffer),
         bindBuffer(6, control_.buffer), bindBuffer(7, duplicateParams_.buffer), bindBuffer(8, mismatch_.buffer)},
        "duplicate");
    return list;
}

void WgpuTileRenderer::ensureTiles(uint32_t width, uint32_t height) {
    if (width == width_ && height == height_ && ranges_buffer_) return;
    width_ = width;
    height_ = height;
    tilesX_ = (width + web::kTileSize - 1) / web::kTileSize;
    tilesY_ = (height + web::kTileSize - 1) / web::kTileSize;
    ranges_buffer_.create(device_, uint64_t(tilesX_) * tilesY_ * 8,
                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                          "splat-tiles: tile ranges");
    if (list_) {  // groups that point at the ranges buffer
        releaseGroup(list_->rangesGroup);
        releaseGroup(list_->renderGroup);
    }
}

void WgpuTileRenderer::ensureListGroups(ListResources& list, const WgpuTexture2D& target) {
    if (!list.rangesGroup) {
        list.rangesGroup = makeBindGroup(device_, rangesLayout_,
                                         {bindBuffer(0, list.keys.buffer), bindBuffer(1, ranges_buffer_.buffer),
                                          bindBuffer(2, control_.buffer), bindBuffer(3, rangesParams_.buffer)},
                                         "tile ranges");
    }
    if (!list.renderGroup || list.renderTargetGeneration != target.generation) {
        releaseGroup(list.renderGroup);
        const bool f32 = target.format == WGPUTextureFormat_RGBA32Float;
        list.renderGroup = makeBindGroup(device_, f32 ? render32Layout_ : render16Layout_,
                                         {bindBuffer(0, splats_.buffer), bindBuffer(1, list.values.buffer),
                                          bindBuffer(2, ranges_buffer_.buffer), bindTexture(3, target.view),
                                          bindBuffer(4, renderParams_.buffer)},
                                         "tile render");
        list.renderTargetGeneration = target.generation;
    }
}

void WgpuTileRenderer::adaptCapacity() {
    // Native SplatTileRenderer::adaptCapacity: grow at once to 1.5 M when M > 90% of the capacity,
    // shrink to 1.5 x the recent maximum after 120 consecutive readbacks below 40%.
    if (!settings_.adaptiveList || settings_.fixedCapacity || !lastStats_.valid ||
        lastStats_.frameIndex == lastAdaptedFrame_ || !list_)
        return;
    lastAdaptedFrame_ = lastStats_.frameIndex;
    const uint32_t m = lastStats_.control.duplicates;  // unclamped: the true need
    const uint32_t cap = list_->capacity;
    uint64_t target = 0;
    if (double(m) > 0.9 * cap) {
        target = uint64_t(1.5 * m);
        shrinkFrames_ = shrinkWindowMax_ = 0;
    } else if (double(m) < 0.4 * cap) {
        shrinkWindowMax_ = std::max(shrinkWindowMax_, m);
        if (++shrinkFrames_ >= 120) {
            target = uint64_t(1.5 * shrinkWindowMax_);
            shrinkFrames_ = shrinkWindowMax_ = 0;
        }
    } else {
        shrinkFrames_ = shrinkWindowMax_ = 0;
    }
    if (target == 0) return;
    const uint32_t newCap =
        std::clamp(roundUpToBlock(target), std::min(kMinAdaptiveCapacity, maxCapacity_), maxCapacity_);
    if (newCap == cap || (newCap < cap && double(newCap) > 0.75 * cap)) return;
    // No stall: the old list is released here, but the frames already submitted keep its buffers
    // alive until they complete (WebGPU reference counting).
    list_ = createList(newCap);
    ++listResizes_;
}

void WgpuTileRenderer::encode(WGPUCommandEncoder encoder, const WgpuScene& scene, const WgpuTexture2D& target,
                              const SplatCamera& cam, uint64_t frameIndex, WgpuTimestamps* timestamps) {
    ensureScene(scene);
    ensureTiles(target.width, target.height);
    adaptCapacity();
    ListResources& list = *list_;
    ensureListGroups(list, target);
    const uint32_t count = scene.count;
    const uint32_t tileCount = tilesX_ * tilesY_;

    // === 0. Reset: control block {N, 0, ...}, empty tile ranges; this frame's parameters ======
    web::WebTileControl reset{};
    reset.gaussianCount = count;
    wgpuQueueWriteBuffer(queue_, control_.buffer, 0, &reset, sizeof(reset));
    PreprocessParams pp{};
    pp.viewInria = cam.viewInria;
    pp.camPos = glm::vec4(cam.position, 1.0f);
    pp.focal = {cam.focalX, cam.focalY};
    pp.tanHalfFov = {cam.tanHalfFovX, cam.tanHalfFovY};
    pp.imageSize = {target.width, target.height};
    pp.count = count;
    pp.shDegree = uint32_t(std::clamp(std::min(scene.shDegree, settings_.maxShDegree), 0, 3));
    pp.tiles16 = {cam.tilesX, cam.tilesY};
    pp.grid = {tilesX_, tilesY_};
    wgpuQueueWriteBuffer(queue_, preprocessParams_.buffer, 0, &pp, sizeof(pp));
    const SetupParams sp{list.capacity, maxGroupsX_, 0, 0};
    wgpuQueueWriteBuffer(queue_, setupParams_.buffer, 0, &sp, sizeof(sp));
    const DuplicateParams dp{tilesX_, tileCount, 0, 0};
    wgpuQueueWriteBuffer(queue_, duplicateParams_.buffer, 0, &dp, sizeof(dp));
    const RangesParams rp{tileCount, 0, 0, 0};
    wgpuQueueWriteBuffer(queue_, rangesParams_.buffer, 0, &rp, sizeof(rp));
    const RenderParams rdp{target.width, target.height, tilesX_, 0};
    wgpuQueueWriteBuffer(queue_, renderParams_.buffer, 0, &rdp, sizeof(rdp));
    wgpuCommandEncoderClearBuffer(encoder, ranges_buffer_.buffer, 0, ranges_buffer_.size);
    wgpuCommandEncoderClearBuffer(encoder, mismatch_.buffer, 0, mismatch_.size);

    auto ts = [&](const char* name) { return timestamps ? timestamps->writes(name) : nullptr; };
    auto end = [](WGPUComputePassEncoder pass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    };

    // === 1. Preprocess + cull: Splat[i], tileCounts[i], spans[i], cull counters ================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "preprocess", ts("preprocess"));
        wgpuComputePassEncoderSetPipeline(pass, preprocessPipeline(scene));
        wgpuComputePassEncoderSetBindGroup(pass, 0, scene.bindGroup(), 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(pass, 1, preprocessGroup_, 0, nullptr);
        dispatch1D(pass, groupsFor(count), maxGroupsX_);
        end(pass);
    }
    // === 2-3. Order-preserving compaction: scan of the visibility flags, compact ===============
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "scan: visible flags", ts("scan: visible flags"));
        scanFlags_.encode(pass);
        end(pass);
    }
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "compact", ts("compact (index order)"));
        wgpuComputePassEncoderSetPipeline(pass, compact_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, compactGroup_, 0, nullptr);
        dispatch1D(pass, groupsFor(count), maxGroupsX_);
        end(pass);
    }
    // === 4. Depth sort (32-bit keys, stable) ===================================================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "depth sort", ts("depth sort (8 passes)"));
        depthSort_.encode(pass, 32);
        end(pass);
    }
    // === 5. Scan of the touched-tile counts in depth order -> offsets, M =======================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "scan: tiles touched", ts("scan: tiles touched"));
        scanTiles_.encode(pass);
        end(pass);
    }
    // === 6. Setup: clamp to the capacity, overflow, list dispatch arguments ===================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "tile setup", ts("tile setup (1 thread)"));
        wgpuComputePassEncoderSetPipeline(pass, setup_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, setupGroup_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        end(pass);
    }
    // === 7. Duplicate: one entry per (tile, splat), from the stored spans ======================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "duplicate", ts("duplicate"));
        wgpuComputePassEncoderSetPipeline(pass, duplicate_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, list.duplicateGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, listArgs_.buffer, 0);
        end(pass);
    }
    // The mismatch counter (+ example) -> control words 16..19, for the readback.
    wgpuCommandEncoderCopyBufferToBuffer(encoder, mismatch_.buffer, 0, control_.buffer,
                                         offsetof(web::WebTileControl, coverageMismatch), 16);
    // === 8. Tile sort: stable, by tile id (incl. the sentinel id = tileCount) ==================
    {
        const uint32_t bits = tileKeyBits(tileCount);
        WGPUComputePassEncoder pass = beginComputePass(encoder, "tile sort", ts("tile sort"));
        list.sort.encode(pass, bits);
        end(pass);
    }
    // === 9. Tile ranges =========================================================================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "tile ranges", ts("tile ranges"));
        wgpuComputePassEncoderSetPipeline(pass, ranges_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, list.rangesGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, listArgs_.buffer, 0);
        end(pass);
    }
    // === 10. Render: one workgroup per tile ======================================================
    {
        WGPUComputePassEncoder pass = beginComputePass(encoder, "render tiles", ts("render tiles"));
        const bool f32 = target.format == WGPUTextureFormat_RGBA32Float;
        wgpuComputePassEncoderSetPipeline(pass, f32 ? render32_ : render16_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, list.renderGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, tilesX_, tilesY_, 1);
        end(pass);
    }

    // Control block -> a readback slot, read asynchronously. Every frame is read back (the ring
    // grows while the browser is slow to deliver the maps): tests look up the counters of the very
    // frame they captured.
    pendingSlot_ = controlRing_.acquire(64);
    pendingFrame_ = frameIndex;
    if (pendingSlot_ >= 0)
        wgpuCommandEncoderCopyBufferToBuffer(encoder, control_.buffer, 0, controlRing_.buffer(pendingSlot_), 0,
                                             sizeof(web::WebTileControl));
}

void WgpuTileRenderer::afterSubmit() {
    if (pendingSlot_ < 0) return;
    controlRing_.mapAfterSubmit(pendingSlot_, pendingFrame_);
    pendingSlot_ = -1;
}

void WgpuTileRenderer::onControl(const uint8_t* data, uint64_t frameIndex) {
    FrameStats s;
    s.valid = true;
    s.frameIndex = frameIndex;
    std::memcpy(&s.control, data, sizeof(s.control));
    s.capacity = list_ ? list_->capacity : 0;
    ++framesRead_;
    maxDuplicates_ = std::max(maxDuplicates_, s.control.duplicates);
    mismatchTotal_ += s.control.coverageMismatch;
    if (s.control.coverageMismatch > 0 && mismatchFrames_.size() < 8)
        mismatchFrames_.push_back({frameIndex, s.control.coverageMismatch, s.control.mismatchExample[0],
                                   s.control.mismatchExample[1], s.control.mismatchExample[2]});
    if (s.control.overflow > 0) {
        ++overflowFrames_;
        maxOverflow_ = std::max(maxOverflow_, s.control.overflow);
    }
    history_.push_back(s);
    while (history_.size() > kHistory) history_.pop_front();
    if (!lastStats_.valid || frameIndex >= lastStats_.frameIndex) lastStats_ = s;
}

std::optional<WgpuTileRenderer::FrameStats> WgpuTileRenderer::statsForFrame(uint64_t frameIndex) const {
    for (const FrameStats& s : history_)
        if (s.frameIndex == frameIndex) return s;
    return std::nullopt;
}

uint64_t WgpuTileRenderer::listBytes() const {
    return list_ ? list_->keys.size + list_->values.size + list_->sort.scratchBytes() : 0;
}

uint64_t WgpuTileRenderer::totalBytes() const {
    uint64_t b = listBytes();
    for (const WgpuBuffer* x : {&splats_, &tileCounts_, &spans_, &slots_, &depthKeys_, &depthValues_, &offsets_,
                                &control_, &listArgs_, &ranges_buffer_})
        b += x->size;
    return b + depthSort_.scratchBytes() + scanFlags_.scratchBytes() + scanTiles_.scratchBytes();
}

std::string WgpuTileRenderer::statsJson() const {
    nlohmann::json j;
    j["subgroups"] = settings_.subgroups;
    j["capacity"] = capacity();
    j["maxCapacity"] = maxCapacity_;
    j["listResizes"] = listResizes_;
    j["listBytes"] = listBytes();
    j["totalBytes"] = totalBytes();
    j["tiles"] = {tilesX_, tilesY_};
    j["framesRead"] = framesRead_;
    j["overflowFrames"] = overflowFrames_;
    j["maxOverflow"] = maxOverflow_;
    j["maxDuplicates"] = maxDuplicates_;
    j["mismatchTotal"] = mismatchTotal_;
    j["mismatchFrames"] = nlohmann::json::array();
    for (const auto& m : mismatchFrames_)
        j["mismatchFrames"].push_back({{"frame", m[0]}, {"count", m[1]}, {"splat", m[2]}, {"recordCount", m[3]},
                                       {"tileCount", m[4]}});
    if (lastStats_.valid) {
        const web::WebTileControl& c = lastStats_.control;
        j["last"] = {{"frame", lastStats_.frameIndex}, {"gaussians", c.gaussianCount}, {"visible", c.visible},
                     {"duplicates", c.duplicates}, {"entries", c.entries}, {"overflow", c.overflow},
                     {"mismatch", c.coverageMismatch}, {"capacity", lastStats_.capacity},
                     {"mismatchExample", std::vector<uint32_t>(c.mismatchExample, c.mismatchExample + 3)},
                     {"cullCounts", std::vector<uint32_t>(c.cullCounts, c.cullCounts + 5)},
                     {"listGroups", std::vector<uint32_t>(c.listGroups, c.listGroups + 3)}};
    }
    return j.dump();
}

WgpuTileRenderer::DebugBuffers WgpuTileRenderer::debugBuffers() const {
    return {splats_.buffer, tileCounts_.buffer, spans_.buffer, slots_.buffer, depthKeys_.buffer, depthValues_.buffer,
            offsets_.buffer, list_ ? list_->keys.buffer : nullptr, list_ ? list_->values.buffer : nullptr,
            ranges_buffer_.buffer, control_.buffer, count_, capacity(), tilesX_, tilesY_};
}

}  // namespace gsr
