#include "webgpu/wgpu_shaders.h"

#include "webgpu/wgpu_context.h"

#include <nlohmann/json.hpp>

#include <cstdio>

namespace gsr {

std::string WgslVariant::label() const {
    std::string s = file;
    if (!defines.empty()) {
        s += "[";
        for (size_t i = 0; i < defines.size(); ++i) s += (i ? "," : "") + defines[i];
        s += "]";
    }
    return s;
}

std::vector<WgslVariant> webShaderVariants(bool subgroups) {
    std::vector<WgslVariant> v = {
        {"gradient.wgsl", {}},     // W0 test pattern (wgpu_gradient.*)
        {"resolve.wgsl", {}},      // rgba16float -> canvas / rgba8 capture (wgpu_resolve.*)
        {"points.wgsl", {}},       // W1 points renderer (wgpu_points.*), includes scene_decode.wgsl
        {"decode_test.wgsl", {}},  // W1 scene decoder self-test (wgpu_decode_selftest.*)
        // W2 splat-tiles (wgpu_tile_renderer.*, wgpu_scan.*, wgpu_radix_sort.*): the portable builds ...
        {"scan.wgsl", {}},
        {"radix_sort.wgsl", {}},
        {"splat_preprocess.wgsl", {}},
        {"tile_compact.wgsl", {}},
        {"tile_setup.wgsl", {}},
        {"tile_duplicate.wgsl", {}},
        {"tile_ranges.wgsl", {}},
        {"tile_render.wgsl", {}},
        {"tile_render.wgsl", {"GSR_OUT_RGBA32F"}},
    };
    if (subgroups) {  // ... and the subgroup builds (only compiled when the device has the feature)
        for (const char* f : {"scan.wgsl", "radix_sort.wgsl", "splat_preprocess.wgsl", "tile_render.wgsl"})
            v.push_back({f, {"GSR_SUBGROUPS"}});
        v.push_back({"tile_render.wgsl", {"GSR_SUBGROUPS", "GSR_OUT_RGBA32F"}});
    }
    return v;
}

WGPUShaderModule WgslShaderCache::get(WGPUDevice device, const WgslVariant& variant) {
    const std::string label = variant.label();
    for (const Entry& e : entries_) {
        if (e.label == label) return e.module;
    }

    Entry& entry = entries_.emplace_back();
    entry.label = label;
    WgslModuleReport& report = reports_.emplace_back();
    report.label = label;
    entry.report = &report;

    entry.source = preprocessWgsl(variant.file, variant.defines, embeddedWgslLoader());
    if (!entry.source.ok) {
        report.errors = 1;
        report.infoReceived = true;
        report.messages.push_back("preprocessor error " + entry.source.error);
        std::fprintf(stderr, "WGSL %s: preprocessor error %s\n", label.c_str(), entry.source.error.c_str());
        return nullptr;
    }
    report.preprocessed = true;

    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = toStringView(entry.source.code);
    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &wgsl.chain;
    desc.label = toStringView(entry.label);
    entry.module = wgpuDeviceCreateShaderModule(device, &desc);

    // The compiler's messages arrive asynchronously; errors also reach the uncaptured-error
    // callback (counted by WgpuContext), warnings only arrive here.
    WGPUCompilationInfoCallbackInfo cb = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = &WgslShaderCache::onCompilationInfo;
    cb.userdata1 = this;
    cb.userdata2 = &entry;
    ++pending_;
    wgpuShaderModuleGetCompilationInfo(entry.module, cb);
    return entry.module;
}

void WgslShaderCache::onCompilationInfo(WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* info,
                                        void* userdata1, void* userdata2) {
    auto* self = static_cast<WgslShaderCache*>(userdata1);
    auto* entry = static_cast<Entry*>(userdata2);
    if (self->pending_ > 0) --self->pending_;
    WgslModuleReport& r = *entry->report;
    r.infoReceived = true;
    if (status != WGPUCompilationInfoRequestStatus_Success || !info) {
        r.messages.push_back("compilation info unavailable");
        return;
    }
    for (size_t i = 0; i < info->messageCount; ++i) {
        const WGPUCompilationMessage& m = info->messages[i];
        const char* kind = "info";
        switch (m.type) {
        case WGPUCompilationMessageType_Error: ++r.errors; kind = "error"; break;
        case WGPUCompilationMessageType_Warning: ++r.warnings; kind = "warning"; break;
        default: ++r.infos; break;
        }
        // Map the line of the preprocessed text back to the original file.
        const std::string where = entry->source.describeLine(m.lineNum) + ":" + std::to_string(m.linePos);
        const std::string text = std::string(kind) + " " + where + ": " + toString(m.message);
        r.messages.push_back(text);
        std::fprintf(stderr, "WGSL %s: %s\n", r.label.c_str(), text.c_str());
    }
}

uint32_t WgslShaderCache::errorCount() const {
    uint32_t n = 0;
    for (const WgslModuleReport& r : reports_) n += r.errors;
    return n;
}

uint32_t WgslShaderCache::warningCount() const {
    uint32_t n = 0;
    for (const WgslModuleReport& r : reports_) n += r.warnings;
    return n;
}

std::string WgslShaderCache::reportJson() const {
    nlohmann::json modules = nlohmann::json::array();
    bool pass = pending_ == 0;
    for (const WgslModuleReport& r : reports_) {
        modules.push_back({{"label", r.label},
                           {"preprocessed", r.preprocessed},
                           {"infoReceived", r.infoReceived},
                           {"errors", r.errors},
                           {"warnings", r.warnings},
                           {"infos", r.infos},
                           {"messages", r.messages}});
        pass = pass && r.preprocessed && r.infoReceived && r.errors == 0 && r.warnings == 0;
    }
    return nlohmann::json{{"pass", pass}, {"pending", pending_}, {"modules", modules}}.dump();
}

void WgslShaderCache::release() {
    for (Entry& e : entries_) {
        if (e.module) wgpuShaderModuleRelease(e.module);
        e.module = nullptr;
    }
    // Keep entries_/reports_: a callback may still point at them (they only go away with the cache).
}

}  // namespace gsr
