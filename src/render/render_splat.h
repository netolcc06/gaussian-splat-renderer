// CPU twin of shaders/render_splat.glsl (Phase 5): encode/decode of the packed render
// splats of splat-tiles (`--render-splat p32|p24`). Used by the tests to check the GPU's
// packed records against the 48-byte Splat, and to document the layouts in C++.
//
//   p32 (8 words): cx, cy, conic.a, conic.b | conic.c, opacity, h(r, g), h(b, 0) | radius << 16
//   p24 (6 words): cx, cy | h(a * 2^12, b * 2^12), h(c * 2^12, opacity) | h(r, g), h(b, 0) | radius << 16
#pragma once

#include "scene/half.h"
#include "splat/splat_math.h"

#include <algorithm>
#include <cstdint>

namespace gsr {

inline constexpr float kConicScaleP24 = 4096.0f;  // kConicScaleP24 in render_splat.glsl

// The fields the render pass uses (radius = the 16-bit integer radius).
struct RenderSplatFields {
    glm::vec2 center{0.0f};
    glm::vec4 conicOpacity{0.0f};
    glm::vec3 color{0.0f};
    float radius = 0.0f;
};

inline uint32_t packColorRadius(float b, float radius) {
    return uint32_t(floatToHalf(b)) | (uint32_t(std::min(radius, 65535.0f)) << 16);
}

// words: 8 (p32) or 6 (p24) uint32.
inline void encodeRenderSplat(bool p24, const ProjectedSplat& s, uint32_t* w) {
    if (!p24) {
        w[0] = floatToBits(s.center.x);
        w[1] = floatToBits(s.center.y);
        w[2] = floatToBits(s.conicOpacity.x);
        w[3] = floatToBits(s.conicOpacity.y);
        w[4] = floatToBits(s.conicOpacity.z);
        w[5] = floatToBits(s.conicOpacity.w);
        w[6] = packHalf2(s.color.r, s.color.g);
        w[7] = packColorRadius(s.color.b, s.radius);
    } else {
        w[0] = floatToBits(s.center.x);
        w[1] = floatToBits(s.center.y);
        w[2] = packHalf2(s.conicOpacity.x * kConicScaleP24, s.conicOpacity.y * kConicScaleP24);
        w[3] = packHalf2(s.conicOpacity.z * kConicScaleP24, s.conicOpacity.w);
        w[4] = packHalf2(s.color.r, s.color.g);
        w[5] = packColorRadius(s.color.b, s.radius);
    }
}

inline RenderSplatFields decodeRenderSplat(bool p24, const uint32_t* w) {
    RenderSplatFields f;
    f.center = {bitsToFloat(w[0]), bitsToFloat(w[1])};
    if (!p24) {
        f.conicOpacity = {bitsToFloat(w[2]), bitsToFloat(w[3]), bitsToFloat(w[4]), bitsToFloat(w[5])};
        f.color = {unpackHalfLo(w[6]), unpackHalfHi(w[6]), unpackHalfLo(w[7])};
        f.radius = float(w[7] >> 16);
    } else {
        const float inv = 1.0f / kConicScaleP24;
        f.conicOpacity = {unpackHalfLo(w[2]) * inv, unpackHalfHi(w[2]) * inv, unpackHalfLo(w[3]) * inv,
                          unpackHalfHi(w[3])};
        f.color = {unpackHalfLo(w[4]), unpackHalfHi(w[4]), unpackHalfLo(w[5])};
        f.radius = float(w[5] >> 16);
    }
    return f;
}

}  // namespace gsr
