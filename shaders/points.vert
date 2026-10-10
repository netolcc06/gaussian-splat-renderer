#version 460
// Phase 1 point-cloud debug view: one small screen-space square per gaussian,
// coloured by its degree-0 SH colour. No covariance, no blending: this only checks
// that positions, colours, opacities and the scene transform are loaded correctly.
//
// Drawn as vkCmdDraw(6, count): instance = gaussian, vertex = one of 6 quad corners.
// Scene buffers (bindings 0-3, any Phase 5 scene format): see src/render/gpu_scene.h and
// scene_decode.glsl.
#extension GL_GOOGLE_include_directive : require
#include "scene_decode.glsl"

layout(push_constant) uniform PushConstants {
    mat4 mvp;               // proj * view * model (model = scene transform)
    vec2 viewportSize;      // pixels
    float pointSizePx;      // square side in pixels
    float opacityThreshold; // gaussians with alpha below this are skipped
    uint shCoeffs;          // unused since Phase 5 (the decoder knows the stored layout)
} pc;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outCorner;  // -1..1 inside the square (for round points)

const float SH_C0 = 0.28209479177387814;

// Two CCW triangles covering [-1, 1]^2.
const vec2 kCorners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    const uint id = gl_InstanceIndex;
    const vec4 pa = loadPosAlpha(id);
    const vec4 clip = pc.mvp * vec4(pa.xyz, 1.0);

    // Culled: below the opacity threshold or behind the camera. A position outside the
    // clip volume (x > w) for all 6 vertices makes the whole quad disappear.
    if (pa.w < pc.opacityThreshold || clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outColor = vec3(0.0);
        outCorner = vec2(0.0);
        return;
    }

    // Offset in NDC: a full NDC range (2 units) spans viewportSize pixels, so a half
    // size of pointSizePx/2 pixels is pointSizePx / viewportSize NDC units. Multiply
    // by w because the rasterizer divides by it.
    const vec2 corner = kCorners[gl_VertexIndex];
    gl_Position = clip + vec4(corner * pc.pointSizePx / pc.viewportSize * clip.w, 0.0, 0.0);

    // Degree-0 colour, as the 3DGS reference computes it (no upper clamp per gaussian
    // there; we clamp because this is the final displayed colour).
    const vec3 dc = loadDC(id);
    outColor = clamp(0.5 + SH_C0 * dc, 0.0, 1.0);
    outCorner = corner;
}
