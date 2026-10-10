// WGSL shader modules: preprocess (wgsl_preprocess.h) the embedded sources, create the
// modules, and collect the compiler's messages (getCompilationInfo) per variant. The web
// counterpart of loading .spv files; there is no offline compile step on the web, the browser
// compiles WGSL at module creation.
//
// Every shader variant the web build uses is listed in webShaderVariants(). At start-up all of
// them are compiled once and their messages collected: that is the shader self-test
// (gsr_selftest("shaders")), which fails on any error or warning (DESIGN.md section 9.1).
#pragma once

#include "webgpu/wgsl_preprocess.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <vector>

namespace gsr {

struct WgslVariant {
    std::string file;                  // root file in shaders/wgsl/
    std::vector<std::string> defines;  // #ifdef flags
    std::string label() const;         // "file[DEF1,DEF2]"
};

// Every variant the web build creates pipelines from (with the subgroup builds when the device
// supports them).
std::vector<WgslVariant> webShaderVariants(bool subgroups);

struct WgslModuleReport {
    std::string label;
    bool preprocessed = false;  // false: preprocessor error (in `messages`), no module created
    bool infoReceived = false;  // the getCompilationInfo callback has run
    uint32_t errors = 0, warnings = 0, infos = 0;
    std::vector<std::string> messages;  // "error resolve.wgsl:12:5: ..."
};

class WgslShaderCache {
public:
    WgslShaderCache() = default;
    WgslShaderCache(const WgslShaderCache&) = delete;
    WgslShaderCache& operator=(const WgslShaderCache&) = delete;
    ~WgslShaderCache() { release(); }

    // The module for a variant, created on first use (nullptr if preprocessing failed).
    // Cached: the same variant returns the same module (not add-ref'ed; the cache owns it).
    WGPUShaderModule get(WGPUDevice device, const WgslVariant& variant);

    // Compilation-info callbacks still outstanding.
    uint32_t pendingInfo() const { return pending_; }

    // Totals over all modules created so far.
    uint32_t errorCount() const;
    uint32_t warningCount() const;
    const std::deque<WgslModuleReport>& reports() const { return reports_; }
    std::string reportJson() const;  // {"pass":..., "modules":[...]}

    void release();

private:
    struct Entry {
        std::string label;
        WGPUShaderModule module = nullptr;
        WgslPreprocessResult source;
        WgslModuleReport* report = nullptr;
    };
    static void onCompilationInfo(WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* info,
                                  void* userdata1, void* userdata2);

    std::deque<Entry> entries_;           // deque: stable addresses for the callbacks
    std::deque<WgslModuleReport> reports_;
    uint32_t pending_ = 0;
};

}  // namespace gsr
