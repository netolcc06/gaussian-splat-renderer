// The WGSL shader files (shaders/wgsl/*.wgsl), embedded into the wasm at build time by
// cmake/embed_wgsl.cmake. There is no file system on the web side: the preprocessor's
// #include and the shader cache look files up by name here.
#pragma once

#include <span>
#include <string_view>

namespace gsr {

struct WgslSourceFile {
    std::string_view name;  // file name relative to shaders/wgsl/, e.g. "resolve.wgsl"
    std::string_view text;
};

// Every embedded file, sorted by name (generated: build-web/generated/wgsl_embedded.cpp).
std::span<const WgslSourceFile> embeddedWgslFiles();

inline const WgslSourceFile* findEmbeddedWgsl(std::string_view name) {
    for (const WgslSourceFile& f : embeddedWgslFiles()) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

}  // namespace gsr
