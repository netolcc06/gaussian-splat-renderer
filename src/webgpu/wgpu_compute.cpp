#include "webgpu/wgpu_compute.h"

#include "webgpu/wgpu_context.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace gsr {

void WgpuBuffer::create(WGPUDevice device, uint64_t bytes, WGPUBufferUsage usage, const char* label) {
    release();
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.usage = usage;
    desc.size = std::max<uint64_t>((bytes + 3) & ~uint64_t(3), 16);  // never empty, 4-byte multiple
    buffer = wgpuDeviceCreateBuffer(device, &desc);
    size = desc.size;
}

void WgpuBuffer::release() {
    if (buffer) wgpuBufferRelease(buffer);
    buffer = nullptr;
    size = 0;
}

WGPUBindGroupLayout makeComputeLayout(WGPUDevice device, std::initializer_list<LayoutEntry> entries, const char* label) {
    std::vector<WGPUBindGroupLayoutEntry> e;
    for (const LayoutEntry& le : entries) {
        WGPUBindGroupLayoutEntry x = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        x.binding = le.binding;
        x.visibility = WGPUShaderStage_Compute;
        switch (le.kind) {
        case BindKind::ReadOnlyStorage: x.buffer.type = WGPUBufferBindingType_ReadOnlyStorage; break;
        case BindKind::Storage: x.buffer.type = WGPUBufferBindingType_Storage; break;
        case BindKind::Uniform: x.buffer.type = WGPUBufferBindingType_Uniform; break;
        case BindKind::UniformDynamic:
            x.buffer.type = WGPUBufferBindingType_Uniform;
            x.buffer.hasDynamicOffset = WGPU_TRUE;
            break;
        case BindKind::StorageTexture16F:
        case BindKind::StorageTexture32F:
            x.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
            x.storageTexture.format = le.kind == BindKind::StorageTexture16F ? WGPUTextureFormat_RGBA16Float
                                                                            : WGPUTextureFormat_RGBA32Float;
            x.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
            break;
        }
        e.push_back(x);
    }
    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.entryCount = e.size();
    desc.entries = e.data();
    return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUPipelineLayout makePipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> groups,
                                      const char* label) {
    const std::vector<WGPUBindGroupLayout> g(groups);
    WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.bindGroupLayoutCount = g.size();
    desc.bindGroupLayouts = g.data();
    return wgpuDeviceCreatePipelineLayout(device, &desc);
}

WGPUBindGroup makeBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, std::initializer_list<BindEntry> entries,
                            const char* label) {
    std::vector<WGPUBindGroupEntry> e;
    for (const BindEntry& be : entries) {
        WGPUBindGroupEntry x = WGPU_BIND_GROUP_ENTRY_INIT;
        x.binding = be.binding;
        if (be.view) {
            x.textureView = be.view;
        } else {
            x.buffer = be.buffer;
            x.offset = be.offset;
            x.size = be.size;
        }
        e.push_back(x);
    }
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.layout = layout;
    desc.entryCount = e.size();
    desc.entries = e.data();
    return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUComputePipeline makeComputePipeline(WGPUDevice device, WGPUPipelineLayout layout, WGPUShaderModule module,
                                        const char* entryPoint, const std::vector<WGPUConstantEntry>& constants,
                                        const char* label) {
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.layout = layout;
    desc.compute.module = module;
    desc.compute.entryPoint = toStringView(entryPoint);
    desc.compute.constantCount = constants.size();
    desc.compute.constants = constants.empty() ? nullptr : constants.data();
    return wgpuDeviceCreateComputePipeline(device, &desc);
}

void dispatch1D(WGPUComputePassEncoder pass, uint32_t groups, uint32_t maxGroupsX) {
    if (groups == 0) return;
    const uint32_t x = std::min(groups, std::max(maxGroupsX, 1u));
    const uint32_t y = (groups + x - 1) / x;
    wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, 1);
}

WGPUComputePassEncoder beginComputePass(WGPUCommandEncoder encoder, const char* label,
                                        const WGPUPassTimestampWrites* timestamps) {
    WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.timestampWrites = timestamps;
    return wgpuCommandEncoderBeginComputePass(encoder, &desc);
}

// --- Readback ring ---------------------------------------------------------------------------------

struct WgpuReadbackRing::State {
    struct Slot {
        WGPUBuffer buffer = nullptr;
        bool busy = false;  // copy encoded or map pending
        uint64_t tag = 0;
    };
    std::vector<Slot> slots;
    WGPUDevice device = nullptr;  // not owned
    uint64_t bytes = 0;
    DataFn onData;
    bool alive = true;
};

namespace {
struct MapContext {
    std::shared_ptr<WgpuReadbackRing::State> state;  // keeps the slots alive until the callback
    int slot = 0;
};
}  // namespace

void WgpuReadbackRing::create(WGPUDevice device, uint64_t bytes, uint32_t slots, const char* label, DataFn onData) {
    release();
    state_ = std::make_shared<State>();
    state_->device = device;
    state_->bytes = (bytes + 3) & ~uint64_t(3);
    state_->onData = std::move(onData);
    state_->slots.resize(slots);
    for (State::Slot& s : state_->slots) {
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.label = toStringView(label);
        desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        desc.size = state_->bytes;
        s.buffer = wgpuDeviceCreateBuffer(device, &desc);
    }
}

void WgpuReadbackRing::release() {
    if (!state_) return;
    state_->alive = false;
    for (State::Slot& s : state_->slots) {
        // A pending map is aborted by the release; its callback only drops its reference.
        if (s.buffer && !s.busy) wgpuBufferRelease(s.buffer);
        if (!s.busy) s.buffer = nullptr;
    }
    state_.reset();
}

int WgpuReadbackRing::acquire(uint32_t growTo) {
    if (!state_) return -1;
    for (size_t i = 0; i < state_->slots.size(); ++i) {
        if (!state_->slots[i].busy) {
            state_->slots[i].busy = true;
            return int(i);
        }
    }
    if (state_->slots.size() >= growTo || !state_->device) return -1;
    State::Slot s;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.label = toStringView("readback (grown)");
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    desc.size = state_->bytes;
    s.buffer = wgpuDeviceCreateBuffer(state_->device, &desc);
    s.busy = true;
    state_->slots.push_back(s);
    return int(state_->slots.size() - 1);
}

WGPUBuffer WgpuReadbackRing::buffer(int slot) const { return state_->slots[size_t(slot)].buffer; }
uint64_t WgpuReadbackRing::bytes() const { return state_ ? state_->bytes : 0; }

uint32_t WgpuReadbackRing::inFlight() const {
    uint32_t n = 0;
    if (state_)
        for (const State::Slot& s : state_->slots) n += s.busy;
    return n;
}

void WgpuReadbackRing::mapAfterSubmit(int slot, uint64_t tag) {
    State::Slot& s = state_->slots[size_t(slot)];
    s.tag = tag;
    auto* ctx = new MapContext{state_, slot};
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        std::unique_ptr<MapContext> c(static_cast<MapContext*>(userdata1));
        State& st = *c->state;
        State::Slot& sl = st.slots[size_t(c->slot)];
        if (!st.alive) {  // the ring was released meanwhile
            if (sl.buffer) wgpuBufferRelease(sl.buffer);
            sl.buffer = nullptr;
            return;
        }
        if (status == WGPUMapAsyncStatus_Success) {
            const auto* data = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(sl.buffer, 0, size_t(st.bytes)));
            if (data && st.onData) st.onData(data, st.bytes, sl.tag);
            wgpuBufferUnmap(sl.buffer);
        }
        sl.busy = false;
    };
    cb.userdata1 = ctx;
    wgpuBufferMapAsync(s.buffer, WGPUMapMode_Read, 0, size_t(state_->bytes), cb);
}

// --- Timestamps -----------------------------------------------------------------------------------

bool WgpuTimestamps::create(WGPUDevice device, bool supported) {
    release();
    if (!supported) return false;
    WGPUQuerySetDescriptor desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    desc.label = toStringView("pass timestamps");
    desc.type = WGPUQueryType_Timestamp;
    desc.count = 2 * kMaxPasses;
    querySet_ = wgpuDeviceCreateQuerySet(device, &desc);
    resolve_.create(device, 2 * kMaxPasses * 8, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
                    "timestamps resolve");
    constexpr uint32_t kSlots = 4;
    slotNames_.assign(kSlots, {});
    ring_.create(device, 2 * kMaxPasses * 8, kSlots, "timestamps readback",
                 [this](const uint8_t* data, uint64_t, uint64_t tag) { onData(data, tag); });
    return querySet_ != nullptr;
}

void WgpuTimestamps::release() {
    ring_.release();
    resolve_.release();
    if (querySet_) {
        wgpuQuerySetDestroy(querySet_);
        wgpuQuerySetRelease(querySet_);
    }
    querySet_ = nullptr;
    slot_ = -1;
}

void WgpuTimestamps::beginFrame() {
    names_.clear();
    writes_.clear();
    writes_.reserve(kMaxPasses);
    slot_ = enabled() ? ring_.acquire() : -1;
}

const WGPUPassTimestampWrites* WgpuTimestamps::writes(const char* name) {
    if (slot_ < 0 || names_.size() >= kMaxPasses) return nullptr;
    WGPUPassTimestampWrites w = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    w.querySet = querySet_;
    w.beginningOfPassWriteIndex = uint32_t(2 * names_.size());
    w.endOfPassWriteIndex = uint32_t(2 * names_.size() + 1);
    names_.push_back(name);
    writes_.push_back(w);
    return &writes_.back();
}

void WgpuTimestamps::endFrame(WGPUCommandEncoder encoder) {
    if (slot_ < 0) return;
    if (names_.empty()) {
        // Nothing timed: give the slot back via an empty map would be wasteful; just copy nothing.
        wgpuCommandEncoderCopyBufferToBuffer(encoder, resolve_.buffer, 0, ring_.buffer(slot_), 0, 8);
        slotNames_[size_t(slot_)].clear();
        return;
    }
    const uint32_t queries = uint32_t(2 * names_.size());
    wgpuCommandEncoderResolveQuerySet(encoder, querySet_, 0, queries, resolve_.buffer, 0);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, resolve_.buffer, 0, ring_.buffer(slot_), 0, uint64_t(queries) * 8);
    slotNames_[size_t(slot_)] = names_;
}

void WgpuTimestamps::afterSubmit() {
    if (slot_ < 0) return;
    ring_.mapAfterSubmit(slot_, uint64_t(slot_));
    slot_ = -1;
}

WgpuTimestamps::PassStats& WgpuTimestamps::statsFor(const std::string& name) {
    for (PassStats& s : stats_)
        if (s.name == name) return s;
    stats_.push_back({name});
    return stats_.back();
}

void WgpuTimestamps::onData(const uint8_t* data, uint64_t tag) {
    const std::vector<std::string>& names = slotNames_[size_t(tag)];
    if (names.empty()) return;
    std::vector<uint64_t> t(names.size() * 2);
    std::memcpy(t.data(), data, t.size() * 8);
    // A pass whose end is before its begin (or zero) is invalid (e.g. the GPU clock reset).
    uint64_t first = UINT64_MAX, last = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const uint64_t b = t[2 * i], e = t[2 * i + 1];
        if (b == 0 || e < b) return;
        first = std::min(first, b);
        last = std::max(last, e);
    }
    for (size_t i = 0; i < names.size(); ++i) {
        const uint64_t d = t[2 * i + 1] - t[2 * i];
        PassStats& s = statsFor(names[i]);
        s.lastMs = double(d) * 1e-6;
        s.sumMs += s.lastMs;
        ++s.samples;
        ++deltas_;
        deltasOn100us_ += d % 100000 == 0;
    }
    PassStats& f = statsFor("frame (first pass begin .. last pass end)");
    f.lastMs = double(last - first) * 1e-6;
    f.sumMs += f.lastMs;
    ++f.samples;
    ++framesMeasured_;
    quantized_ = deltas_ > 0 && deltasOn100us_ == deltas_;
}

void WgpuTimestamps::resetStats() {
    stats_.clear();
    framesMeasured_ = 0;
    deltas_ = deltasOn100us_ = 0;
    quantized_ = false;
}

}  // namespace gsr
