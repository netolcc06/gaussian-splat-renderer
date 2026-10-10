// Phase 5: the RENDER SPLAT of the tile rasterizer (`--render-splat`), i.e. what the
// duplicate and render passes of splat-tiles read per (tile, splat) entry.
// C++ side: RenderSplatMode in src/render/splat_tile_renderer.h, CPU encode/decode twins in
// src/render/render_splat.h (tests).
//
//   full  (0)  the 48-byte Splat of splat_common.glsl (Phase 4): 2 x 32-byte sectors per
//              random fetch, 40 of the 48 bytes used by the render pass.
//   p32   (1)  32 bytes, one sector:   (cx, cy, conic.a, conic.b) fp32
//                                      (conic.c fp32, opacity fp32, h(r, g), h(b, 0) | radius << 16)
//              Lossless except the colour (fp16, relative error <= 2^-11).
//   p24   (2)  24 bytes (1.5 sectors on average for a random fetch):
//                                      (cx, cy) fp32
//                                      (h(a * S, b * S), h(c * S, opacity))
//                                      (h(r, g), h(b, 0) | radius << 16)
//              conic in fp16, pre-multiplied by S = 2^12 so that the conic entries of large
//              splats (1/sigma^2 down to ~1e-8 px^-2) stay in fp16's normal range (>= 2^-14).
//
// In both packed modes the tile rectangle and the view depth (needed by the duplicate,
// compact and global-sort passes) live in their own per-gaussian arrays (rects, depths),
// so those passes read 8 + 4 bytes instead of a 48-byte Splat. radius is the integer
// ceil(3 sqrt(lambda_max)) clamped to 65535 (only the radius-bound render test uses it).

#ifndef RENDER_SPLAT_GLSL
#define RENDER_SPLAT_GLSL

const uint kRenderSplatFull = 0u;
const uint kRenderSplatP32 = 1u;
const uint kRenderSplatP24 = 2u;
layout(constant_id = 10) const uint kRenderSplatMode = 0u;

const float kConicScaleP24 = 4096.0;  // 2^12, exact in fp32 and fp16

uint packColorRadius(float b, float radius) {
    return (packHalf2x16(vec2(b, 0.0)) & 0xFFFFu) | (uint(min(radius, 65535.0)) << 16);
}

// Tile rectangle [lo, hi) in tile units, 16 bits per coordinate.
uvec2 packRect(ivec4 rect) { return uvec2(uint(rect.x) | (uint(rect.y) << 16), uint(rect.z) | (uint(rect.w) << 16)); }
ivec4 unpackRect(uvec2 r) { return ivec4(int(r.x & 0xFFFFu), int(r.x >> 16), int(r.y & 0xFFFFu), int(r.y >> 16)); }

// The decoded form the render pass keeps in shared memory.
struct RenderSplatDecoded {
    vec2 center;
    vec4 conicOpacity;
    uvec2 colorRadius;  // h(r, g), h(b, 0) | radius << 16  (decoded only when blending)
};

vec3 colorOf(uvec2 cr) { return vec3(unpackHalf2x16(cr.x), unpackHalf2x16(cr.y).x); }

// Exact half -> float of the 16-bit pattern h, with integer operations only. Same value as
// unpackHalf2x16 for the stored bits, but a compiler cannot fold it with a preceding
// packHalf2x16 (Phase 6 pitfall: in the preprocess, unpackHalf2x16(packHalf2x16(x)) did NOT
// give the values later read back from the stored words; the count then disagreed with the
// duplicate pass on ~0.4% of the p24 splats).
float halfBitsToFloat(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) return uintBitsToFloat(sign) + (sign != 0u ? -1.0 : 1.0) * float(m) * (1.0 / 16777216.0);  // 2^-24
    if (e == 31u) return uintBitsToFloat(sign | 0x7F800000u | (m << 13));
    return uintBitsToFloat(sign | ((e + 112u) << 23) | (m << 13));
}

// p24: the conic + opacity exactly as the render pass decodes them from the packed words.
vec4 decodeConicOpacityP24(uvec2 w) {
    vec2 ab = vec2(halfBitsToFloat(w.x & 0xFFFFu), halfBitsToFloat(w.x >> 16));
    vec2 co = vec2(halfBitsToFloat(w.y & 0xFFFFu), halfBitsToFloat(w.y >> 16));
    return vec4(ab * (1.0 / kConicScaleP24), co.x * (1.0 / kConicScaleP24), co.y);
}
uvec2 encodeConicOpacityP24(vec4 conicOpacity) {
    return uvec2(packHalf2x16(conicOpacity.xy * kConicScaleP24),
                 packHalf2x16(vec2(conicOpacity.z * kConicScaleP24, conicOpacity.w)));
}
float radiusOf(uvec2 cr) { return float(cr.y >> 16); }

#endif  // RENDER_SPLAT_GLSL
