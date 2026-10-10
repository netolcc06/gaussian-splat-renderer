#include "webgpu/wgpu_context.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <utility>

namespace gsr {
namespace {

constexpr size_t kMaxRecentErrors = 16;

const char* featureName(WGPUFeatureName f) {
    switch (f) {
    case WGPUFeatureName_CoreFeaturesAndLimits: return "core-features-and-limits";
    case WGPUFeatureName_DepthClipControl: return "depth-clip-control";
    case WGPUFeatureName_Depth32FloatStencil8: return "depth32float-stencil8";
    case WGPUFeatureName_TextureCompressionBC: return "texture-compression-bc";
    case WGPUFeatureName_TextureCompressionBCSliced3D: return "texture-compression-bc-sliced-3d";
    case WGPUFeatureName_TextureCompressionETC2: return "texture-compression-etc2";
    case WGPUFeatureName_TextureCompressionASTC: return "texture-compression-astc";
    case WGPUFeatureName_TextureCompressionASTCSliced3D: return "texture-compression-astc-sliced-3d";
    case WGPUFeatureName_TimestampQuery: return "timestamp-query";
    case WGPUFeatureName_IndirectFirstInstance: return "indirect-first-instance";
    case WGPUFeatureName_ShaderF16: return "shader-f16";
    case WGPUFeatureName_RG11B10UfloatRenderable: return "rg11b10ufloat-renderable";
    case WGPUFeatureName_BGRA8UnormStorage: return "bgra8unorm-storage";
    case WGPUFeatureName_Float32Filterable: return "float32-filterable";
    case WGPUFeatureName_Float32Blendable: return "float32-blendable";
    case WGPUFeatureName_ClipDistances: return "clip-distances";
    case WGPUFeatureName_DualSourceBlending: return "dual-source-blending";
    case WGPUFeatureName_Subgroups: return "subgroups";
    case WGPUFeatureName_TextureFormatsTier1: return "texture-formats-tier1";
    case WGPUFeatureName_TextureFormatsTier2: return "texture-formats-tier2";
    case WGPUFeatureName_PrimitiveIndex: return "primitive-index";
    case WGPUFeatureName_TextureComponentSwizzle: return "texture-component-swizzle";
    case WGPUFeatureName_Unorm16TextureFormats: return "unorm16-texture-formats";
    case WGPUFeatureName_MultiDrawIndirect: return "multi-draw-indirect";
    default: return nullptr;
    }
}

const char* wgslFeatureName(WGPUWGSLLanguageFeatureName f) {
    switch (f) {
    case WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures: return "readonly_and_readwrite_storage_textures";
    case WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct: return "packed_4x8_integer_dot_product";
    case WGPUWGSLLanguageFeatureName_UnrestrictedPointerParameters: return "unrestricted_pointer_parameters";
    case WGPUWGSLLanguageFeatureName_PointerCompositeAccess: return "pointer_composite_access";
    case WGPUWGSLLanguageFeatureName_UniformBufferStandardLayout: return "uniform_buffer_standard_layout";
    case WGPUWGSLLanguageFeatureName_SubgroupId: return "subgroup_id";
    case WGPUWGSLLanguageFeatureName_TextureAndSamplerLet: return "texture_and_sampler_let";
    case WGPUWGSLLanguageFeatureName_SubgroupUniformity: return "subgroup_uniformity";
    case WGPUWGSLLanguageFeatureName_TextureFormatsTier1: return "texture_formats_tier1";
    case WGPUWGSLLanguageFeatureName_LinearIndexing: return "linear_indexing";
    default: return nullptr;
    }
}

std::string unknownName(uint32_t value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "unknown-0x%08x", value);
    return buf;
}

std::vector<std::string> featureList(const WGPUSupportedFeatures& features) {
    std::vector<std::string> names;
    for (size_t i = 0; i < features.featureCount; ++i) {
        const char* n = featureName(features.features[i]);
        names.push_back(n ? n : unknownName(features.features[i]));
    }
    return names;
}

const char* adapterTypeName(WGPUAdapterType t) {
    switch (t) {
    case WGPUAdapterType_DiscreteGPU: return "discrete GPU";
    case WGPUAdapterType_IntegratedGPU: return "integrated GPU";
    case WGPUAdapterType_CPU: return "CPU";
    default: return "unknown";
    }
}

const char* backendName(WGPUBackendType b) {
    switch (b) {
    case WGPUBackendType_Null: return "Null";
    case WGPUBackendType_WebGPU: return "WebGPU";
    case WGPUBackendType_D3D11: return "D3D11";
    case WGPUBackendType_D3D12: return "D3D12";
    case WGPUBackendType_Metal: return "Metal";
    case WGPUBackendType_Vulkan: return "Vulkan";
    case WGPUBackendType_OpenGL: return "OpenGL";
    case WGPUBackendType_OpenGLES: return "OpenGLES";
    default: return "undefined";
    }
}

const char* deviceLostReasonName(WGPUDeviceLostReason r) {
    switch (r) {
    case WGPUDeviceLostReason_Destroyed: return "destroyed";
    case WGPUDeviceLostReason_CallbackCancelled: return "callback-cancelled";
    case WGPUDeviceLostReason_FailedCreation: return "failed-creation";
    default: return "unknown";
    }
}

nlohmann::json limitsJson(const WGPULimits& l) {
    // The limits that matter for this renderer (DESIGN.md section 2.2) plus a few related ones.
    return {
        {"maxStorageBuffersPerShaderStage", l.maxStorageBuffersPerShaderStage},
        {"maxStorageTexturesPerShaderStage", l.maxStorageTexturesPerShaderStage},
        {"maxUniformBuffersPerShaderStage", l.maxUniformBuffersPerShaderStage},
        {"maxComputeWorkgroupStorageSize", l.maxComputeWorkgroupStorageSize},
        {"maxComputeInvocationsPerWorkgroup", l.maxComputeInvocationsPerWorkgroup},
        {"maxComputeWorkgroupSizeX", l.maxComputeWorkgroupSizeX},
        {"maxComputeWorkgroupSizeY", l.maxComputeWorkgroupSizeY},
        {"maxComputeWorkgroupsPerDimension", l.maxComputeWorkgroupsPerDimension},
        {"maxStorageBufferBindingSize", l.maxStorageBufferBindingSize},
        {"maxUniformBufferBindingSize", l.maxUniformBufferBindingSize},
        {"maxBufferSize", l.maxBufferSize},
        {"maxTextureDimension2D", l.maxTextureDimension2D},
        {"maxBindGroups", l.maxBindGroups},
        {"maxDynamicUniformBuffersPerPipelineLayout", l.maxDynamicUniformBuffersPerPipelineLayout},
        {"minUniformBufferOffsetAlignment", l.minUniformBufferOffsetAlignment},
        {"minStorageBufferOffsetAlignment", l.minStorageBufferOffsetAlignment},
        {"maxImmediateSize", l.maxImmediateSize},
    };
}

}  // namespace

std::string toString(WGPUStringView view) {
    if (!view.data) return {};
    if (view.length == WGPU_STRLEN) return std::string(view.data);
    return std::string(view.data, view.length);
}

WGPUStringView toStringView(std::string_view s) {
    return WGPUStringView{s.data(), s.size()};
}

WgpuContext::~WgpuContext() {
    if (queue_) wgpuQueueRelease(queue_);
    if (device_) wgpuDeviceRelease(device_);
    if (adapter_) wgpuAdapterRelease(adapter_);
    if (surface_) wgpuSurfaceRelease(surface_);
    if (instance_) wgpuInstanceRelease(instance_);
}

void WgpuContext::start(std::string canvasSelector, DoneCallback done) {
    canvasSelector_ = std::move(canvasSelector);
    done_ = std::move(done);
    state_ = WgpuContextState::Requesting;

    instance_ = wgpuCreateInstance(nullptr);
    if (!instance_) return finish(false, "wgpuCreateInstance failed");

    // The WGSL language extensions are a property of the browser, known before any adapter.
    WGPUSupportedWGSLLanguageFeatures wgsl = {};
    wgpuInstanceGetWGSLLanguageFeatures(instance_, &wgsl);
    for (size_t i = 0; i < wgsl.featureCount; ++i) {
        const char* n = wgslFeatureName(wgsl.features[i]);
        wgslFeatures_.push_back(n ? n : unknownName(wgsl.features[i]));
    }
    wgpuSupportedWGSLLanguageFeaturesFreeMembers(wgsl);

    // The canvas surface (the swapchain equivalent: the browser presents after each frame).
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasSource = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    canvasSource.selector = toStringView(canvasSelector_);
    WGPUSurfaceDescriptor surfaceDesc = WGPU_SURFACE_DESCRIPTOR_INIT;
    surfaceDesc.nextInChain = &canvasSource.chain;
    surface_ = wgpuInstanceCreateSurface(instance_, &surfaceDesc);
    if (!surface_) return finish(false, "cannot create a WebGPU surface for canvas " + canvasSelector_);

    // Prefer the discrete GPU, like native (Chrome also needs --force_high_performance_gpu in
    // headless mode on a hybrid laptop, DESIGN.md section 9).
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.featureLevel = WGPUFeatureLevel_Core;
    options.powerPreference = WGPUPowerPreference_HighPerformance;
    options.compatibleSurface = surface_;
    WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = &WgpuContext::onAdapter;
    cb.userdata1 = this;
    wgpuInstanceRequestAdapter(instance_, &options, cb);
}

void WgpuContext::onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
                            void* userdata1, void*) {
    auto* self = static_cast<WgpuContext*>(userdata1);
    if (status != WGPURequestAdapterStatus_Success || !adapter) {
        return self->finish(false, "no WebGPU adapter: " + toString(message));
    }
    self->adapter_ = adapter;

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
        WgpuAdapterDescription& d = self->adapterDescription_;
        d.vendor = toString(info.vendor);
        d.architecture = toString(info.architecture);
        d.device = toString(info.device);
        d.description = toString(info.description);
        d.backend = backendName(info.backendType);
        d.adapterType = adapterTypeName(info.adapterType);
        d.vendorId = info.vendorID;
        d.deviceId = info.deviceID;
        d.subgroupMinSize = info.subgroupMinSize;
        d.subgroupMaxSize = info.subgroupMaxSize;
        wgpuAdapterInfoFreeMembers(info);
    }
    wgpuAdapterGetLimits(adapter, &self->adapterLimits_);
    WGPUSupportedFeatures features = {};
    wgpuAdapterGetFeatures(adapter, &features);
    self->adapterFeatures_ = featureList(features);
    wgpuSupportedFeaturesFreeMembers(features);

    self->requestDevice();
}

void WgpuContext::requestDevice() {
    // Request the adapter's own values for the limits the splat renderers depend on (never
    // assume them, DESIGN.md section 2.2); everything else stays at the WebGPU defaults.
    WGPULimits required = WGPU_LIMITS_INIT;
    required.maxStorageBuffersPerShaderStage = adapterLimits_.maxStorageBuffersPerShaderStage;
    required.maxComputeWorkgroupStorageSize = adapterLimits_.maxComputeWorkgroupStorageSize;
    required.maxComputeInvocationsPerWorkgroup = adapterLimits_.maxComputeInvocationsPerWorkgroup;
    required.maxComputeWorkgroupSizeX = adapterLimits_.maxComputeWorkgroupSizeX;
    required.maxComputeWorkgroupSizeY = adapterLimits_.maxComputeWorkgroupSizeY;
    required.maxComputeWorkgroupsPerDimension = adapterLimits_.maxComputeWorkgroupsPerDimension;
    required.maxStorageBufferBindingSize = adapterLimits_.maxStorageBufferBindingSize;
    required.maxBufferSize = adapterLimits_.maxBufferSize;

    // Optional features: GPU timestamps and subgroup operations, when the adapter has them.
    std::vector<WGPUFeatureName> wanted;
    for (WGPUFeatureName f : {WGPUFeatureName_TimestampQuery, WGPUFeatureName_Subgroups}) {
        if (wgpuAdapterHasFeature(adapter_, f)) wanted.push_back(f);
    }

    WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
    desc.label = toStringView("gsplat device");
    desc.requiredFeatureCount = wanted.size();
    desc.requiredFeatures = wanted.data();
    desc.requiredLimits = &required;
    desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    desc.deviceLostCallbackInfo.callback = &WgpuContext::onDeviceLost;
    desc.deviceLostCallbackInfo.userdata1 = this;
    desc.uncapturedErrorCallbackInfo.callback = &WgpuContext::onUncapturedError;
    desc.uncapturedErrorCallbackInfo.userdata1 = this;

    WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = &WgpuContext::onDevice;
    cb.userdata1 = this;
    wgpuAdapterRequestDevice(adapter_, &desc, cb);
}

void WgpuContext::onDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message,
                           void* userdata1, void*) {
    auto* self = static_cast<WgpuContext*>(userdata1);
    if (status != WGPURequestDeviceStatus_Success || !device) {
        return self->finish(false, "requestDevice failed: " + toString(message));
    }
    self->device_ = device;
    self->queue_ = wgpuDeviceGetQueue(device);
    wgpuDeviceGetLimits(device, &self->deviceLimits_);
    WGPUSupportedFeatures features = {};
    wgpuDeviceGetFeatures(device, &features);
    self->deviceFeatures_ = featureList(features);
    wgpuSupportedFeaturesFreeMembers(features);

    // The canvas' preferred format comes first (navigator.gpu.getPreferredCanvasFormat(),
    // bgra8unorm on Windows).
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    if (wgpuSurfaceGetCapabilities(self->surface_, self->adapter_, &caps) != WGPUStatus_Success ||
        caps.formatCount == 0) {
        return self->finish(false, "cannot query the canvas surface capabilities");
    }
    self->surfaceFormat_ = caps.formats[0];
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    self->finish(true);
}

void WgpuContext::onUncapturedError(WGPUDevice const*, WGPUErrorType type, WGPUStringView message,
                                    void* userdata1, void*) {
    auto* self = static_cast<WgpuContext*>(userdata1);
    WgpuErrorCounters& e = self->errors_;
    const char* kind = "unknown";
    switch (type) {
    case WGPUErrorType_Validation: ++e.validation; kind = "validation"; break;
    case WGPUErrorType_OutOfMemory: ++e.outOfMemory; kind = "out-of-memory"; break;
    case WGPUErrorType_Internal: ++e.internal; kind = "internal"; break;
    default: ++e.unknown; break;
    }
    const std::string text = std::string("[WebGPU ") + kind + " error] " + toString(message);
    if (e.recent.size() < kMaxRecentErrors) e.recent.push_back(text);
    std::fprintf(stderr, "%s\n", text.c_str());
}

void WgpuContext::onDeviceLost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message,
                               void* userdata1, void*) {
    auto* self = static_cast<WgpuContext*>(userdata1);
    // CallbackCancelled = the context is being torn down (not a loss of a working device).
    if (reason == WGPUDeviceLostReason_CallbackCancelled) return;
    self->errors_.deviceLost = true;
    self->errors_.deviceLostReason = deviceLostReasonName(reason);
    self->errors_.deviceLostMessage = toString(message);
    std::fprintf(stderr, "[WebGPU device lost: %s] %s\n", self->errors_.deviceLostReason.c_str(),
                 self->errors_.deviceLostMessage.c_str());
    if (self->state_ == WgpuContextState::Requesting) self->finish(false, "device lost during creation");
}

void WgpuContext::finish(bool ok, std::string failure) {
    state_ = ok ? WgpuContextState::Ready : WgpuContextState::Failed;
    failure_ = std::move(failure);
    if (!ok) std::fprintf(stderr, "WebGPU initialization failed: %s\n", failure_.c_str());
    if (done_) {
        DoneCallback done = std::move(done_);
        done_ = nullptr;
        done(ok);
    }
}

void WgpuContext::configureSurface(uint32_t width, uint32_t height) {
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = device_;
    config.format = surfaceFormat_;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = width;
    config.height = height;
    config.alphaMode = WGPUCompositeAlphaMode_Opaque;
    config.presentMode = WGPUPresentMode_Fifo;  // the only mode a canvas has (rAF-paced)
    wgpuSurfaceConfigure(surface_, &config);
    surfaceWidth_ = width;
    surfaceHeight_ = height;
}

std::string WgpuContext::describeJson() const {
    const WgpuAdapterDescription& d = adapterDescription_;
    nlohmann::json j;
    j["adapter"] = {
        {"vendor", d.vendor},
        {"architecture", d.architecture},
        {"device", d.device},
        {"description", d.description},
        {"backend", d.backend},
        {"adapterType", d.adapterType},
        {"vendorId", d.vendorId},
        {"deviceId", d.deviceId},
        {"subgroupMinSize", d.subgroupMinSize},
        {"subgroupMaxSize", d.subgroupMaxSize},
    };
    j["adapterFeatures"] = adapterFeatures_;
    j["deviceFeatures"] = deviceFeatures_;
    j["wgslLanguageFeatures"] = wgslFeatures_;
    j["adapterLimits"] = limitsJson(adapterLimits_);
    j["deviceLimits"] = limitsJson(deviceLimits_);
    j["surfaceFormat"] = surfaceFormat_ == WGPUTextureFormat_BGRA8Unorm   ? "bgra8unorm"
                         : surfaceFormat_ == WGPUTextureFormat_RGBA8Unorm ? "rgba8unorm"
                                                                          : "other";
    j["errors"] = {
        {"validation", errors_.validation},
        {"outOfMemory", errors_.outOfMemory},
        {"internal", errors_.internal},
        {"unknown", errors_.unknown},
        {"total", errors_.total()},
        {"recent", errors_.recent},
    };
    j["deviceLost"] = errors_.deviceLost;
    if (errors_.deviceLost) {
        j["deviceLostReason"] = errors_.deviceLostReason;
        j["deviceLostMessage"] = errors_.deviceLostMessage;
    }
    return j.dump();
}

}  // namespace gsr
