#include "webgpu/wgpu_decode_selftest.h"

#include "scene/gaussian_scene.h"
#include "scene/scene_compression.h"
#include "webgpu/wgpu_context.h"
#include "webgpu/wgpu_scene.h"
#include "webgpu/wgpu_shaders.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gsr {

namespace {

constexpr uint32_t kGaussians = 1000;
constexpr uint32_t kVec4PerGaussian = 20;  // decode_test.wgsl output layout

// Deterministic pseudo-random gaussians, SH degree 3. Higher SH bands are scaled by 0, 1e-3 or 1
// at random, so the variable-SH cases see every degree.
GaussianScene randomScene(uint32_t n) {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return float(uint32_t(state >> 40)) / float(1u << 24);  // [0, 1)
    };
    auto range = [&](float lo, float hi) { return lo + (hi - lo) * next(); };
    GaussianScene s;
    s.count = n;
    s.shDegree = 3;
    s.posAlpha.resize(n);
    s.scales.resize(n);
    s.rotations.resize(n);
    s.sh.resize(size_t(n) * 16 * 3);
    for (uint32_t i = 0; i < n; ++i) {
        s.posAlpha[i] = glm::vec4(range(-5, 5), range(-5, 5), range(-5, 5), range(0.0f, 1.0f));
        s.scales[i] = glm::vec4(std::exp(range(-5, 0)), std::exp(range(-5, 0)), std::exp(range(-5, 0)), 0.0f);
        glm::vec4 q(range(-1, 1), range(-1, 1), range(-1, 1), range(-1, 1));
        s.rotations[i] = q / std::sqrt(glm::dot(q, q));
        const float bandScale[4] = {1.0f, next() < 0.3f ? 0.0f : 1.0f, next() < 0.3f ? 1e-3f : 1.0f,
                                    next() < 0.5f ? 0.0f : 1.0f};
        for (uint32_t k = 0; k < 16; ++k) {
            const uint32_t band = k == 0 ? 0 : k < 4 ? 1 : k < 9 ? 2 : 3;
            for (uint32_t c = 0; c < 3; ++c)
                s.sh[(size_t(i) * 16 + k) * 3 + c] = (k == 0 ? range(-2, 2) : range(-0.5f, 0.5f)) * bandScale[band];
        }
    }
    return s;
}

GaussianScene sliceScene(const GaussianScene& s, uint32_t first, uint32_t n) {
    GaussianScene b;
    b.count = n;
    b.shDegree = s.shDegree;
    const uint32_t K = s.shCoeffsPerChannel();
    b.posAlpha.assign(s.posAlpha.begin() + first, s.posAlpha.begin() + first + n);
    b.scales.assign(s.scales.begin() + first, s.scales.begin() + first + n);
    b.rotations.assign(s.rotations.begin() + first, s.rotations.begin() + first + n);
    b.sh.assign(s.sh.begin() + size_t(first) * K * 3, s.sh.begin() + size_t(first + n) * K * 3);
    return b;
}

struct Case {
    std::string name;
    CompressionOptions options;
    bool streamed = false;  // upload through WgpuScene::create + upload in batches
};

}  // namespace

struct WgpuDecodeSelftest::Impl {
    struct Run {
        Case c;
        EncodedScene encoded;  // CPU reference (decodeGaussian)
        WgpuScene scene;
        WGPUComputePipeline pipeline = nullptr;
        WGPUBuffer params = nullptr, output = nullptr, readback = nullptr;
        WGPUBindGroup group = nullptr;
        std::vector<float> data;
        bool mapped = false, mapFailed = false;
        std::string error;
        Impl* owner = nullptr;
    };
    std::vector<std::unique_ptr<Run>> runs;
    bool started = false;
    uint32_t pendingMaps = 0;
    WGPUBindGroupLayout testLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    nlohmann::json result;
    bool finished = false;

    void compare();
    static void onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2);
    ~Impl();
};

WgpuDecodeSelftest::WgpuDecodeSelftest() : impl_(std::make_unique<Impl>()) {}
WgpuDecodeSelftest::~WgpuDecodeSelftest() = default;

bool WgpuDecodeSelftest::started() const { return impl_->started; }
bool WgpuDecodeSelftest::done() const { return impl_->finished; }

std::string WgpuDecodeSelftest::resultJson() const {
    if (!impl_->started) return R"({"pass":false,"pending":false,"error":"not started"})";
    if (!impl_->finished) return R"({"pending":true})";
    return impl_->result.dump();
}

void WgpuDecodeSelftest::start(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders,
                               WGPUBindGroupLayout sceneLayout) {
    using Run = Impl::Run;
    Impl& s = *impl_;
    s.started = true;
    WGPUShaderModule module = shaders.get(device, {"decode_test.wgsl", {}});

    // Group 1 of decode_test.wgsl: params (uniform) + output (storage).
    WGPUBindGroupLayoutEntry entries[2] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.type = WGPUBufferBindingType_Storage;
    WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bglDesc.entryCount = 2;
    bglDesc.entries = entries;
    s.testLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
    const WGPUBindGroupLayout layouts[2] = {sceneLayout, s.testLayout};
    WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    plDesc.bindGroupLayoutCount = 2;
    plDesc.bindGroupLayouts = layouts;
    s.pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    auto opts = [](SceneFormat f, SceneLayout l, int maxDegree = 3, float shThreshold = -1.0f) {
        CompressionOptions o;
        o.format = f;
        o.layout = l;
        o.maxShDegree = maxDegree;
        o.shThreshold = shThreshold;
        o.threads = 1;
        return o;
    };
    const Case cases[] = {
        {"f32 soa", opts(SceneFormat::F32, SceneLayout::Soa)},
        {"f32 soa, SH degree 1", opts(SceneFormat::F32, SceneLayout::Soa, 1)},
        {"f32 soa, variable SH", opts(SceneFormat::F32, SceneLayout::Soa, 3, 0.25f)},
        {"f32 split", opts(SceneFormat::F32, SceneLayout::Split)},
        {"f16 split", opts(SceneFormat::F16, SceneLayout::Split)},
        {"f16 split, SH degree 2", opts(SceneFormat::F16, SceneLayout::Split, 2)},
        {"f16 split, variable SH", opts(SceneFormat::F16, SceneLayout::Split, 3, 0.25f)},
        {"q8 split", opts(SceneFormat::Q8, SceneLayout::Split)},
        {"q8 split, variable SH", opts(SceneFormat::Q8, SceneLayout::Split, 3, 0.25f)},
        {"q8 split, SH degree 0", opts(SceneFormat::Q8, SceneLayout::Split, 0)},
        {"f32 soa, streamed in batches", opts(SceneFormat::F32, SceneLayout::Soa), true},
        {"f16 split, streamed in batches", opts(SceneFormat::F16, SceneLayout::Split), true},
        {"q8 split, streamed in batches", opts(SceneFormat::Q8, SceneLayout::Split), true},
    };
    const GaussianScene source = randomScene(kGaussians);
    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuDeviceGetLimits(device, &limits);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    for (const Case& c : cases) {
        auto run = std::make_unique<Run>();
        Run& r = *run;
        r.c = c;
        r.owner = &s;
        r.encoded = encodeScene(source, c.options);
        bool ok = false;
        if (c.streamed) {
            WgpuSceneOptions so;
            so.format = c.options.format;
            so.maxShDegree = c.options.maxShDegree;
            ok = r.scene.create(device, sceneLayout, limits, source.count, source.shDegree, so, &r.error);
            // Uneven batches, like the stream decoder's last partial batch.
            const uint32_t sizes[] = {1, 300, 77, 600, 22};
            uint32_t first = 0;
            for (uint32_t n : sizes) {
                if (!ok) break;
                r.scene.upload(queue, first, sliceScene(source, first, n));
                first += n;
            }
        } else {
            ok = r.scene.createFromEncoded(device, queue, sceneLayout, r.encoded, &r.error);
        }
        if (!ok) {
            s.runs.push_back(std::move(run));
            continue;
        }

        const std::vector<WGPUConstantEntry> constants = r.scene.decodeConstants();
        WGPUComputePipelineDescriptor pDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pDesc.layout = s.pipelineLayout;
        pDesc.compute.module = module;
        pDesc.compute.entryPoint = toStringView("main");
        pDesc.compute.constantCount = constants.size();
        pDesc.compute.constants = constants.data();
        r.pipeline = wgpuDeviceCreateComputePipeline(device, &pDesc);

        const uint32_t params[4] = {kGaussians, r.scene.shCoeffs, 0, 0};
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        bd.size = sizeof(params);
        r.params = wgpuDeviceCreateBuffer(device, &bd);
        wgpuQueueWriteBuffer(queue, r.params, 0, params, sizeof(params));
        const uint64_t outBytes = uint64_t(kGaussians) * kVec4PerGaussian * 16;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
        bd.size = outBytes;
        r.output = wgpuDeviceCreateBuffer(device, &bd);
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        r.readback = wgpuDeviceCreateBuffer(device, &bd);

        WGPUBindGroupEntry bg[2] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        bg[0].binding = 0;
        bg[0].buffer = r.params;
        bg[0].size = sizeof(params);
        bg[1].binding = 1;
        bg[1].buffer = r.output;
        bg[1].size = outBytes;
        WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgDesc.layout = s.testLayout;
        bgDesc.entryCount = 2;
        bgDesc.entries = bg;
        r.group = wgpuDeviceCreateBindGroup(device, &bgDesc);

        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, r.pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, r.scene.bindGroup(), 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(pass, 1, r.group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, (kGaussians + 63) / 64, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, r.output, 0, r.readback, 0, outBytes);
        s.runs.push_back(std::move(run));
    }
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    for (auto& run : s.runs) {
        if (!run->readback) continue;
        ++s.pendingMaps;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = &Impl::onMapped;
        cb.userdata1 = run.get();
        wgpuBufferMapAsync(run->readback, WGPUMapMode_Read, 0, WGPU_WHOLE_MAP_SIZE, cb);
    }
    if (s.pendingMaps == 0) s.compare();
}

void WgpuDecodeSelftest::Impl::onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*) {
    auto* r = static_cast<Run*>(userdata1);
    if (status == WGPUMapAsyncStatus_Success) {
        const size_t floats = size_t(kGaussians) * kVec4PerGaussian * 4;
        const auto* p = static_cast<const float*>(wgpuBufferGetConstMappedRange(r->readback, 0, floats * 4));
        r->data.assign(p, p + floats);
        wgpuBufferUnmap(r->readback);
        r->mapped = true;
    } else {
        r->mapFailed = true;
        r->error = "map failed: " + toString(message);
    }
    Impl* s = r->owner;
    if (--s->pendingMaps == 0) s->compare();
}

void WgpuDecodeSelftest::Impl::compare() {
    nlohmann::json cases = nlohmann::json::array();
    bool pass = true;
    for (const auto& run : runs) {
        const Run& r = *run;
        nlohmann::json j = {{"name", r.c.name}, {"format", sceneFormatName(r.scene.format)},
                            {"bytesPerGaussian", r.encoded.bytesPerGaussian()},
                            {"variableSh", uint32_t(r.encoded.variableSh)}};
        if (!r.mapped) {
            j["pass"] = false;
            j["error"] = r.error.empty() ? "not run" : r.error;
            pass = false;
            cases.push_back(j);
            continue;
        }
        // Exact except the quaternion of the fp16 records, which the shader renormalizes with
        // inverseSqrt (an approximate instruction on GPUs) where the CPU divides by sqrt.
        uint64_t mismatches = 0;
        float maxQuatErr = 0.0f;
        nlohmann::json first;
        DecodedGaussian d;
        auto check = [&](uint32_t i, const char* what, float got, float want) {
            if (std::memcmp(&got, &want, 4) == 0) return;
            if (++mismatches == 1) first = {{"gaussian", i}, {"field", what}, {"got", got}, {"expected", want}};
        };
        for (uint32_t i = 0; i < kGaussians; ++i) {
            decodeGaussian(r.encoded, i, d);
            const float* o = r.data.data() + size_t(i) * kVec4PerGaussian * 4;
            for (int c = 0; c < 4; ++c) check(i, "posAlpha", o[c], d.posAlpha[c]);
            for (int c = 0; c < 3; ++c) check(i, "sigma", o[4 + c], d.sigma[c]);
            uint32_t degree = 3;
            if (r.encoded.variableSh != VariableSh::Off) {
                const uint32_t meta = r.encoded.geometry == GeometryEncoding::SoaF32
                                          ? r.encoded.buffers[1][size_t(i) * 4 + 3]
                                          : r.encoded.buffers[0][size_t(i) * r.encoded.geometryStride * 4 + 7];
                degree = meta >> 30;
            }
            check(i, "degree", o[7], float(degree));
            for (int c = 0; c < 4; ++c) {
                if (r.encoded.geometry == GeometryEncoding::RecordF16)
                    maxQuatErr = std::max(maxQuatErr, std::abs(o[8 + c] - d.rotation[c]));
                else
                    check(i, "rotation", o[8 + c], d.rotation[c]);
            }
            for (int c = 0; c < 3; ++c) check(i, "dc", o[12 + c], d.sh[0][c]);
            for (uint32_t k = 0; k < 16; ++k)
                for (int c = 0; c < 3; ++c)
                    check(i, "sh", o[16 + k * 4 + c], k < d.sh.size() ? d.sh[k][c] : 0.0f);
        }
        const bool ok = mismatches == 0 && maxQuatErr <= 1e-6f;
        pass = pass && ok;
        j["pass"] = ok;
        j["mismatches"] = mismatches;
        j["maxQuaternionError"] = maxQuatErr;
        if (!first.is_null()) j["firstMismatch"] = first;
        cases.push_back(j);
    }
    result = {{"pass", pass}, {"gaussians", kGaussians}, {"cases", cases}};
    finished = true;
}

WgpuDecodeSelftest::Impl::~Impl() {
    for (auto& run : runs) {
        Run& r = *run;
        if (r.group) wgpuBindGroupRelease(r.group);
        if (r.pipeline) wgpuComputePipelineRelease(r.pipeline);
        for (WGPUBuffer b : {r.params, r.output, r.readback}) {
            if (b) {
                wgpuBufferDestroy(b);
                wgpuBufferRelease(b);
            }
        }
    }
    if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
    if (testLayout) wgpuBindGroupLayoutRelease(testLayout);
}

}  // namespace gsr
