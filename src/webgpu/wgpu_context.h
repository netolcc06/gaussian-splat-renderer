// WebGPU context: instance, canvas surface, adapter, device and queue (the web counterpart of
// src/core/context.*). Creation is asynchronous in a browser: start() requests the adapter,
// its callback requests the device, and the device callback reports success or failure. No
// ASYNCIFY or JSPI is needed; the callbacks run from the browser's event loop
// (WGPUCallbackMode_AllowSpontaneous).
//
// WebGPU always validates. Validation, out-of-memory and internal errors that no error scope
// catches arrive in the "uncaptured error" callback; they are counted here (the web test
// asserts 0, like the native rule "validation messages => exit code 3"), together with
// device loss.
#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace gsr {

// WGPUStringView -> std::string (null data = empty; WGPU_STRLEN = NUL-terminated).
std::string toString(WGPUStringView view);
// std::string_view -> WGPUStringView (no copy: the view must outlive the call).
WGPUStringView toStringView(std::string_view s);

struct WgpuAdapterDescription {
    std::string vendor, architecture, device, description;
    std::string backend;      // "WebGPU" in a browser (the browser's own backend is hidden)
    std::string adapterType;  // "discrete GPU", "integrated GPU", "CPU", "unknown"
    uint32_t vendorId = 0, deviceId = 0;
    uint32_t subgroupMinSize = 0, subgroupMaxSize = 0;
};

struct WgpuErrorCounters {
    uint32_t validation = 0;
    uint32_t outOfMemory = 0;
    uint32_t internal = 0;
    uint32_t unknown = 0;
    std::vector<std::string> recent;  // the first few messages, for logs and tests
    bool deviceLost = false;
    std::string deviceLostReason;
    std::string deviceLostMessage;
    uint32_t total() const { return validation + outOfMemory + internal + unknown; }
};

enum class WgpuContextState { Idle, Requesting, Ready, Failed };

class WgpuContext {
public:
    using DoneCallback = std::function<void(bool ok)>;

    WgpuContext() = default;
    WgpuContext(const WgpuContext&) = delete;
    WgpuContext& operator=(const WgpuContext&) = delete;
    ~WgpuContext();

    // Starts the asynchronous creation for the canvas matching `canvasSelector` (CSS selector,
    // e.g. "#canvas"). `done` is called once, with the outcome. The object must stay alive.
    void start(std::string canvasSelector, DoneCallback done);

    WgpuContextState state() const { return state_; }
    const std::string& failure() const { return failure_; }

    // (Re)configures the canvas surface; also sets the canvas' backing size in pixels.
    void configureSurface(uint32_t width, uint32_t height);
    uint32_t surfaceWidth() const { return surfaceWidth_; }
    uint32_t surfaceHeight() const { return surfaceHeight_; }

    WGPUInstance instance() const { return instance_; }
    WGPUAdapter adapter() const { return adapter_; }
    WGPUDevice device() const { return device_; }
    WGPUQueue queue() const { return queue_; }
    WGPUSurface surface() const { return surface_; }
    WGPUTextureFormat surfaceFormat() const { return surfaceFormat_; }

    const WgpuAdapterDescription& adapterDescription() const { return adapterDescription_; }
    const WGPULimits& adapterLimits() const { return adapterLimits_; }
    const WGPULimits& deviceLimits() const { return deviceLimits_; }
    const std::vector<std::string>& adapterFeatures() const { return adapterFeatures_; }
    const std::vector<std::string>& deviceFeatures() const { return deviceFeatures_; }
    const std::vector<std::string>& wgslLanguageFeatures() const { return wgslFeatures_; }
    bool hasFeature(WGPUFeatureName f) const { return device_ && wgpuDeviceHasFeature(device_, f); }

    const WgpuErrorCounters& errors() const { return errors_; }

    // Everything above as a JSON object (used by gsr_stats_json and the web test).
    std::string describeJson() const;

private:
    static void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
                          void* userdata1, void* userdata2);
    static void onDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message,
                         void* userdata1, void* userdata2);
    static void onUncapturedError(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message,
                                  void* userdata1, void* userdata2);
    static void onDeviceLost(WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message,
                             void* userdata1, void* userdata2);

    void requestDevice();
    void finish(bool ok, std::string failure = {});

    WgpuContextState state_ = WgpuContextState::Idle;
    std::string failure_;
    std::string canvasSelector_;
    DoneCallback done_;

    WGPUInstance instance_ = nullptr;
    WGPUSurface surface_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WGPUTextureFormat surfaceFormat_ = WGPUTextureFormat_Undefined;
    uint32_t surfaceWidth_ = 0, surfaceHeight_ = 0;

    WgpuAdapterDescription adapterDescription_;
    WGPULimits adapterLimits_ = WGPU_LIMITS_INIT;
    WGPULimits deviceLimits_ = WGPU_LIMITS_INIT;
    std::vector<std::string> adapterFeatures_, deviceFeatures_, wgslFeatures_;
    WgpuErrorCounters errors_;
};

}  // namespace gsr
