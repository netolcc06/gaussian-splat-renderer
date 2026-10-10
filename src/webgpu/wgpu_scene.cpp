#include "webgpu/wgpu_scene.h"

#include "scene/gaussian_scene.h"
#include "webgpu/wgpu_context.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iterator>

namespace gsr {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

CompressionOptions encodeOptionsFor(SceneFormat format, int maxShDegree) {
    CompressionOptions o;
    o.format = format;
    o.layout = format == SceneFormat::F32 ? SceneLayout::Soa : SceneLayout::Split;
    o.maxShDegree = maxShDegree;
    o.threads = 1;  // single-threaded wasm
    return o;
}

// The encoding of one dummy gaussian: strides, encodings and words per gaussian of every buffer,
// straight from the encoder (so they can never disagree with what upload() writes).
EncodedScene probeEncoding(const CompressionOptions& options, int shDegree) {
    GaussianScene g;
    g.count = 1;
    g.shDegree = shDegree;
    g.posAlpha = {glm::vec4(0.0f)};
    g.scales = {glm::vec4(1.0f, 1.0f, 1.0f, 0.0f)};
    g.rotations = {glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)};
    g.sh.assign(size_t(g.shCoeffsPerChannel()) * 3, 0.0f);
    return encodeScene(g, options);
}

const char* formatName(SceneFormat f) { return sceneFormatName(f); }

std::string mib(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", double(bytes) / (1024.0 * 1024.0));
    return buf;
}

}  // namespace

WGPUBindGroupLayout WgpuScene::createBindGroupLayout(WGPUDevice device) {
    std::array<WGPUBindGroupLayoutEntry, 4> entries{};
    for (uint32_t b = 0; b < 4; ++b) {
        WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        e.binding = b;
        e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
        e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        entries[b] = e;
    }
    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.label = toStringView("scene");
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

bool WgpuScene::create(WGPUDevice device, WGPUBindGroupLayout layout, const WGPULimits& limits, uint32_t n,
                       int degree, const WgpuSceneOptions& options, std::string* error) {
    release();
    requestedFormat = options.format;
    uint64_t maxBytes = std::min<uint64_t>(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    if (options.debugMaxBufferBytes) maxBytes = std::min(maxBytes, options.debugMaxBufferBytes);

    // Formats in order of precision; start at the requested one and fall back while the
    // largest buffer does not fit the device.
    const SceneFormat order[] = {SceneFormat::F32, SceneFormat::F16, SceneFormat::Q8};
    const size_t start = size_t(std::find(std::begin(order), std::end(order), options.format) - std::begin(order));
    if (start >= std::size(order)) {
        if (error) *error = std::string("scene format '") + formatName(options.format) + "' is not available on the web";
        return false;
    }
    std::string tried;
    for (size_t f = start; f < std::size(order); ++f) {
        const CompressionOptions opt = encodeOptionsFor(order[f], options.maxShDegree);
        const EncodedScene probe = probeEncoding(opt, degree);
        std::array<uint64_t, 4> sizes{};
        std::array<uint32_t, 4> words{};
        uint64_t largest = 0;
        for (size_t b = 0; b < probe.buffers.size() && b < 4; ++b) {
            words[b] = uint32_t(probe.buffers[b].size());
            sizes[b] = uint64_t(n) * words[b] * 4;
            largest = std::max(largest, sizes[b]);
        }
        if (largest > maxBytes) {
            tried += std::string(tried.empty() ? "" : ", ") + formatName(order[f]) + " needs a " + mib(largest) +
                     " buffer";
            continue;
        }
        encodeOptions_ = opt;
        wordsPerGaussian_ = words;
        count = n;
        format = order[f];
        sceneLayout = opt.layout;
        geometry = probe.geometry;
        color = probe.color;
        shDegree = probe.shDegree;
        shCoeffs = probe.shCoeffs;
        geometryStride = probe.geometryStride;
        colorStride = probe.colorStride;
        colorOffset = probe.colorOffset;
        variableSh = VariableSh::Off;
        zeroCopy_ = format == SceneFormat::F32 && options.maxShDegree >= degree;
        fallbackNote = f == start ? std::string()
                                  : std::string("scene format ") + formatName(format) + " instead of " +
                                        formatName(options.format) + " (" + tried + ", device limit " + mib(maxBytes) + ")";
        return createBuffers(device, layout, sizes, error);
    }
    if (error) *error = "the scene does not fit this GPU's buffer limit of " + mib(maxBytes) + " (" + tried + ")";
    return false;
}

bool WgpuScene::createFromEncoded(WGPUDevice device, WGPUQueue queue, WGPUBindGroupLayout layout,
                                  const EncodedScene& e, std::string* error) {
    release();
    if (e.color == ColorEncoding::Codebook) {
        if (error) *error = "the cb format is not available on the web";
        return false;
    }
    count = e.count;
    requestedFormat = format = e.format;
    sceneLayout = e.layout;
    geometry = e.geometry;
    color = e.color;
    shDegree = e.shDegree;
    shCoeffs = e.shCoeffs;
    geometryStride = e.geometryStride;
    colorStride = e.colorStride;
    colorOffset = e.colorOffset;
    variableSh = e.variableSh;
    std::array<uint64_t, 4> sizes{};
    for (size_t b = 0; b < e.buffers.size() && b < 4; ++b) sizes[b] = e.buffers[b].size() * 4;
    if (!createBuffers(device, layout, sizes, error)) return false;
    for (size_t b = 0; b < 4; ++b)
        if (sizes[b]) wgpuQueueWriteBuffer(queue, buffers_[b], 0, e.buffers[b].data(), size_t(sizes[b]));
    return true;
}

bool WgpuScene::createBuffers(WGPUDevice device, WGPUBindGroupLayout layout, const std::array<uint64_t, 4>& sizes,
                              std::string* error) {
    static const char* const kLabels[4] = {"scene buffer 0", "scene buffer 1", "scene buffer 2", "scene buffer 3"};
    totalBytes = 0;
    bufferBytes = sizes;
    for (size_t b = 0; b < 4; ++b) {
        if (!sizes[b]) continue;
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.label = toStringView(kLabels[b]);
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        desc.size = (sizes[b] + 3) & ~uint64_t(3);
        buffers_[b] = wgpuDeviceCreateBuffer(device, &desc);
        if (!buffers_[b]) {
            if (error) *error = std::string("cannot create ") + kLabels[b];
            release();
            return false;
        }
        totalBytes += desc.size;
    }
    if (!buffers_[0]) {
        if (error) *error = "empty scene";
        release();
        return false;
    }
    // Unused bindings get buffer 0 (never read: the decoder's branches are fixed by overrides).
    std::array<WGPUBindGroupEntry, 4> entries{};
    for (uint32_t b = 0; b < 4; ++b) {
        WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
        e.binding = b;
        e.buffer = buffers_[b] ? buffers_[b] : buffers_[0];
        e.size = WGPU_WHOLE_SIZE;
        entries[b] = e;
    }
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.label = toStringView("scene");
    desc.layout = layout;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    bindGroup_ = wgpuDeviceCreateBindGroup(device, &desc);
    return bindGroup_ != nullptr;
}

void WgpuScene::upload(WGPUQueue queue, uint32_t first, const GaussianScene& batch) {
    const uint32_t n = batch.count;
    if (n == 0 || first + uint64_t(n) > count) return;
    ++batches;
    auto write = [&](size_t b, const void* data, size_t bytes) {
        const auto t = Clock::now();
        wgpuQueueWriteBuffer(queue, buffers_[b], uint64_t(first) * wordsPerGaussian_[b] * 4, data, bytes);
        uploadMs += msSince(t);
    };
    if (zeroCopy_) {
        // f32 soa at full degree: the decoded arrays are already the GPU layout (as native's
        // GpuScene::zeroCopy path).
        write(0, batch.posAlpha.data(), batch.posAlpha.size() * sizeof(glm::vec4));
        write(1, batch.scales.data(), batch.scales.size() * sizeof(glm::vec4));
        write(2, batch.rotations.data(), batch.rotations.size() * sizeof(glm::vec4));
        write(3, batch.sh.data(), batch.sh.size() * sizeof(float));
        return;
    }
    const auto t = Clock::now();
    const EncodedScene e = encodeScene(batch, encodeOptions_);
    encodeMs += msSince(t);
    for (size_t b = 0; b < e.buffers.size() && b < 4; ++b)
        if (wordsPerGaussian_[b]) write(b, e.buffers[b].data(), e.buffers[b].size() * 4);
}

std::vector<WGPUConstantEntry> WgpuScene::decodeConstants() const {
    const std::pair<const char*, uint32_t> values[] = {
        {"kGeometryEncoding", uint32_t(geometry)}, {"kColorEncoding", uint32_t(color)},
        {"kGeometryStride", geometryStride},       {"kColorStride", colorStride},
        {"kColorOffset", colorOffset},             {"kStoredCoeffs", shCoeffs},
        {"kVariableSh", uint32_t(variableSh)},
    };
    std::vector<WGPUConstantEntry> out;
    for (const auto& [key, value] : values) {
        WGPUConstantEntry c = WGPU_CONSTANT_ENTRY_INIT;
        c.key = toStringView(key);  // string literals: valid for the program's lifetime
        c.value = double(value);
        out.push_back(c);
    }
    return out;
}

std::string WgpuScene::decoderKey() const {
    return std::to_string(uint32_t(geometry)) + "/" + std::to_string(uint32_t(color)) + "/" +
           std::to_string(geometryStride) + "/" + std::to_string(colorStride) + "/" + std::to_string(colorOffset) +
           "/" + std::to_string(shCoeffs) + "/" + std::to_string(uint32_t(variableSh));
}

void WgpuScene::release() {
    if (bindGroup_) wgpuBindGroupRelease(bindGroup_);
    bindGroup_ = nullptr;
    for (WGPUBuffer& b : buffers_) {
        if (b) {
            wgpuBufferDestroy(b);
            wgpuBufferRelease(b);
        }
        b = nullptr;
    }
    count = 0;
    totalBytes = 0;
    bufferBytes = {};
    batches = 0;
    encodeMs = uploadMs = 0.0;
}

}  // namespace gsr
