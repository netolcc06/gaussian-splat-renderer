// Small helpers for the compute passes of the web renderers (web phase W2): buffers, explicit bind
// group layouts, bind groups, compute pipelines, 1D dispatches split into 2D grids, an asynchronous
// readback ring (MAP_READ buffers + mapAsync: the web counterpart of native's host-visible
// per-frame-slot readback buffers, read "2 frames late" without a stall), and GPU timestamps per
// compute pass (timestamp-query). webgpu.h only.
#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace gsr {

// A storage/uniform/readback buffer.
struct WgpuBuffer {
    WGPUBuffer buffer = nullptr;
    uint64_t size = 0;

    void create(WGPUDevice device, uint64_t bytes, WGPUBufferUsage usage, const char* label);
    // Drops our reference. Work already submitted keeps the buffer alive until it completes.
    void release();
    explicit operator bool() const { return buffer != nullptr; }
};

enum class BindKind {
    ReadOnlyStorage,   // var<storage, read>
    Storage,           // var<storage, read_write>
    Uniform,           // var<uniform>
    UniformDynamic,    // var<uniform> with a dynamic offset (one 256-byte slot per pass)
    StorageTexture16F, // texture_storage_2d<rgba16float, write>
    StorageTexture32F, // texture_storage_2d<rgba32float, write>
};

struct LayoutEntry {
    uint32_t binding;
    BindKind kind;
};

// An explicit compute bind group layout.
WGPUBindGroupLayout makeComputeLayout(WGPUDevice device, std::initializer_list<LayoutEntry> entries, const char* label);
WGPUPipelineLayout makePipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> groups,
                                      const char* label);

struct BindEntry {
    uint32_t binding;
    WGPUBuffer buffer = nullptr;  // or
    WGPUTextureView view = nullptr;
    uint64_t offset = 0;
    uint64_t size = WGPU_WHOLE_SIZE;
};
inline BindEntry bindBuffer(uint32_t binding, WGPUBuffer buffer, uint64_t size = WGPU_WHOLE_SIZE, uint64_t offset = 0) {
    return {binding, buffer, nullptr, offset, size};
}
inline BindEntry bindTexture(uint32_t binding, WGPUTextureView view) { return {binding, nullptr, view, 0, 0}; }

WGPUBindGroup makeBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, std::initializer_list<BindEntry> entries,
                            const char* label);

WGPUComputePipeline makeComputePipeline(WGPUDevice device, WGPUPipelineLayout layout, WGPUShaderModule module,
                                        const char* entryPoint, const std::vector<WGPUConstantEntry>& constants,
                                        const char* label);

// Dispatches `groups` workgroups of a 1D problem, as a (min(groups, maxX), ceil(groups / x)) grid
// when it exceeds the per-dimension limit; the shader linearizes with num_workgroups.
void dispatch1D(WGPUComputePassEncoder pass, uint32_t groups, uint32_t maxGroupsX);

// Begins a compute pass, with timestamp writes when `timestamps` is non-null.
WGPUComputePassEncoder beginComputePass(WGPUCommandEncoder encoder, const char* label,
                                        const WGPUPassTimestampWrites* timestamps = nullptr);

// A ring of MAP_READ buffers for asynchronous readbacks. Per use: acquire() a free slot (or -1 if
// all are still in flight: skip this readback), encode copies into buffer(slot), submit, then
// mapAfterSubmit(slot, tag). When the map completes, onData(bytes, tag) runs and the slot is free
// again. Safe to destroy with maps in flight (the callbacks see a dead ring and only clean up).
class WgpuReadbackRing {
public:
    using DataFn = std::function<void(const uint8_t* data, uint64_t bytes, uint64_t tag)>;

    WgpuReadbackRing() = default;
    WgpuReadbackRing(const WgpuReadbackRing&) = delete;
    WgpuReadbackRing& operator=(const WgpuReadbackRing&) = delete;
    ~WgpuReadbackRing() { release(); }

    void create(WGPUDevice device, uint64_t bytes, uint32_t slots, const char* label, DataFn onData);
    void release();
    // A free slot, or -1 when all are in flight; with `growTo` > current slots, a new slot is added
    // instead of failing (frames whose readback must not be skipped).
    int acquire(uint32_t growTo = 0);
    WGPUBuffer buffer(int slot) const;
    void mapAfterSubmit(int slot, uint64_t tag);
    uint64_t bytes() const;
    uint32_t inFlight() const;

    struct State;  // shared with the map callbacks (public for them only)

private:
    std::shared_ptr<State> state_;
};

// GPU timestamps around compute passes (feature timestamp-query). Per frame: beginFrame() (false if
// no readback slot is free: no timestamps this frame), writes(name) for each pass (null when not
// timing), endFrame(encoder) to resolve, then afterSubmit(). The results arrive asynchronously and
// are accumulated per pass name; Chrome quantizes them to 100 us unless the WebGPU developer
// features are enabled (docs/web/W2.md).
class WgpuTimestamps {
public:
    static constexpr uint32_t kMaxPasses = 24;

    bool create(WGPUDevice device, bool supported);
    void release();
    bool enabled() const { return querySet_ != nullptr; }

    void beginFrame();
    const WGPUPassTimestampWrites* writes(const char* name);
    void endFrame(WGPUCommandEncoder encoder);
    void afterSubmit();

    struct PassStats {
        std::string name;
        double lastMs = 0.0;
        double sumMs = 0.0;
        uint64_t samples = 0;
        double meanMs() const { return samples ? sumMs / double(samples) : 0.0; }
    };
    // Per pass name in first-seen order, plus "frame" (first begin to last end).
    const std::vector<PassStats>& stats() const { return stats_; }
    void resetStats();
    uint64_t framesMeasured() const { return framesMeasured_; }
    bool quantized() const { return quantized_; }  // every delta a multiple of 100 us (heuristic)

private:
    void onData(const uint8_t* data, uint64_t tag);
    PassStats& statsFor(const std::string& name);

    WGPUQuerySet querySet_ = nullptr;
    WgpuBuffer resolve_;
    WgpuReadbackRing ring_;
    int slot_ = -1;                   // this frame's readback slot (-1: not timing)
    std::vector<std::string> names_;  // this frame's passes
    std::vector<WGPUPassTimestampWrites> writes_;
    std::vector<std::vector<std::string>> slotNames_;  // pass names per readback slot
    std::vector<PassStats> stats_;
    uint64_t framesMeasured_ = 0;
    uint64_t deltas_ = 0, deltasOn100us_ = 0;
    bool quantized_ = false;
};

}  // namespace gsr
