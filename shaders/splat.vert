#version 460
// Phase 2 draw: one screen-space quad per visible splat, drawn BACK TO FRONT.
//
// vkCmdDraw(6, visibleCount): instance = position in the depth order, vertex = one of
// the 6 corners of two triangles. The CPU sort gives the indices NEAREST FIRST (the
// order the reference renderer blends in), so instance i reads entry
// visibleCount - 1 - i: the farthest splat is drawn first.
//
// Quad extent (must match the reference renderer's coverage, see splat_math.h):
//   bound 0 = tile:   the splat's INRIA tile rectangle, i.e. exactly the pixels INRIA
//                     evaluates it on. Edges lie on integer pixel boundaries.
//   bound 1 = radius: the square [c - r, c + r] around the centre (fewer fragments).
// Phase 6 bounds (`--bounds`, bits 8.. of the push constant `bound`, tile_coverage.glsl):
//   tight        the quad is clipped to the pixel bounding box of the cutoff ellipse
//                (alpha >= 1/255): only fragments the fragment shader discards anyway are
//                removed, the image is identical
//   cutoff-only  the quad is that bounding box alone (not clipped to the 3-sigma bound)
#extension GL_GOOGLE_include_directive : require
#include "splat_common.glsl"
#include "tile_coverage.glsl"

layout(std430, set = 0, binding = 0) readonly buffer Splats { Splat splats[]; };
layout(std430, set = 0, binding = 1) readonly buffer SortedIndices { uint sortedIndices[]; };

// C++ twin: DrawPushConstants in src/render/splat_renderer.cpp / splat_gpusort_renderer.cpp.
layout(push_constant) uniform PushConstants {
    vec2 viewportSize;  // pixels
    uint visibleCount;  // instances drawn = entries in sortedIndices (unused with GSR_INDIRECT_COUNT)
    uint bound;         // bits 0-7: 0 = tile, 1 = radius; bits 8-15: Phase 6 bounds (kBounds*)
} pc;

#ifdef GSR_INDIRECT_COUNT
// Phase 3 build (splat_indirect.vert.spv): the instance count is only known on the GPU.
// It is the instanceCount of the VkDrawIndirectCommand this draw was launched with
// (vkCmdDrawIndirect), read back here to reverse the nearest-first list.
layout(std430, set = 0, binding = 2) readonly buffer DrawArgs {
    uint vertexCount;
    uint instanceCount;
} drawArgs;
uint visibleCount() { return drawArgs.instanceCount; }
#else
uint visibleCount() { return pc.visibleCount; }
#endif

layout(location = 0) flat out vec2 outCenter;        // pixel coordinates
layout(location = 1) flat out vec4 outConicOpacity;
layout(location = 2) flat out vec3 outColor;

// Two triangles covering [0, 1]^2 (lo -> hi corner interpolation).
const vec2 kCorners[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));

void main() {
    uint id = sortedIndices[visibleCount() - 1u - uint(gl_InstanceIndex)];
    Splat s = splats[id];

    // Culled by the preprocess (singular covariance, off screen; near-plane culled
    // gaussians are not in the sorted list at all): a point outside the clip volume
    // for all 6 vertices makes the rasterizer drop the quad.
    if (s.cullReason != kCullVisible) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outCenter = vec2(0.0);
        outConicOpacity = vec4(0.0);
        outColor = vec3(0.0);
        return;
    }

    // Quad corners in framebuffer coordinates (pixel (i, j) covers [i, i+1) x [j, j+1)).
    vec2 lo, hi;
    uint coverage = pc.bound & 0xFFu;
    uint bounds = pc.bound >> 8;
    if (coverage == 0u) {
        uvec2 tiles = (uvec2(pc.viewportSize) + uint(kTileSize - 1)) / uint(kTileSize);
        ivec4 rect = computeTileRect(s.center, s.radius, tiles);
        lo = vec2(rect.xy * kTileSize);
        hi = vec2(rect.zw * kTileSize);
    } else {
        // Pixel coordinate c is framebuffer coordinate c + 0.5 (the pixel centre).
        lo = s.center + 0.5 - s.radius;
        hi = s.center + 0.5 + s.radius;
    }
    if (bounds != kBoundsRect) {
        // Pixels i with |i - c| <= the ellipse's padded half extent: framebuffer [ceil(c - e), floor(c + e) + 1).
        CutoffEllipse e = makeCutoffEllipse(s.center, s.conicOpacity.xyz, s.conicOpacity.w);
        if (e.valid) {
            vec2 elo = ceil(s.center - vec2(e.exPad, e.eyPad));
            vec2 ehi = floor(s.center + vec2(e.exPad, e.eyPad)) + 1.0;
            if (e.t < 0.0) ehi = elo;  // no pixel reaches alpha 1/255
            if (bounds == kBoundsCutoff) {
                lo = elo;
                hi = ehi;
            } else {
                lo = max(lo, elo);
                hi = min(hi, ehi);
            }
            hi = max(hi, lo);  // empty: a zero-area quad, no fragment
        }
    }
    vec2 fb = mix(lo, hi, kCorners[gl_VertexIndex]);
    // Framebuffer -> NDC with the standard viewport (y down in both): no projection
    // matrix here, the preprocess already projected the splat.
    gl_Position = vec4(fb / pc.viewportSize * 2.0 - 1.0, 0.0, 1.0);

    outCenter = s.center;
    outConicOpacity = s.conicOpacity;
    outColor = s.color;
}
