// A minimal preprocessor for WGSL, which has none (docs/web/DESIGN.md section 2.1, D5).
// It replaces the GLSL variants built with -D defines (GSR_TILES, GSR_PACKED_SPLAT, ...).
//
// Directives (a line whose first non-blank character is '#'; '#' never appears in WGSL itself):
//   #include "file.wgsl"   inserts another file; every file is inserted at most once per
//                          result (an implicit include guard, so diamonds and cycles are fine)
//   #define NAME           defines a flag (flags only: there is no macro substitution;
//                          numeric parameters are WGSL `override` constants instead)
//   #ifdef NAME / #ifndef NAME / #else / #endif   conditional blocks, nestable
// Anything else starting with '#' is an error. Directive lines and lines in inactive blocks
// are dropped; the line map records the source file and line of every output line so that
// compiler messages can be reported against the original files.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gsr {

struct WgslLineOrigin {
    std::string file;
    uint32_t line = 0;  // 1-based
};

struct WgslPreprocessResult {
    bool ok = false;
    std::string code;                     // the preprocessed WGSL (valid only if ok)
    std::string error;                    // "file:line: message" (only if !ok)
    std::vector<WgslLineOrigin> lineMap;  // lineMap[i] = origin of output line i + 1

    // "file:line" for a 1-based output line number (as reported by WGSL compilers).
    std::string describeLine(uint64_t outputLine) const;
};

// Returns the text of a file by name, or nullopt if it does not exist.
using WgslFileLoader = std::function<std::optional<std::string_view>(std::string_view name)>;

WgslPreprocessResult preprocessWgsl(std::string_view rootFile, std::span<const std::string> defines,
                                    const WgslFileLoader& loader);

// Loader over the shaders embedded into the wasm (wgsl_sources.h).
WgslFileLoader embeddedWgslLoader();

}  // namespace gsr
