// Port of shaders/points.vert + shaders/points.frag (web phase W1): the point-cloud debug view.
// One small screen-space square per gaussian, cut to its inscribed disc, coloured by its
// degree-0 SH colour; depth test, no blending. Checks that positions, colours, opacities, the
// scene transform and the camera arrive correctly.
//
// Drawn as draw(6, count): instance = gaussian, vertex = one of 6 quad corners.
// Scene buffers: group 0, bindings 0-3 (scene_decode.wgsl, src/webgpu/wgpu_scene.h).
// Differences from the GLSL: push constants are a uniform buffer (group 1); clip.y is negated
// (the matrices are the native Vulkan ones, whose clip space has +y down; WebGPU's has +y up,
// DESIGN.md 2.1); the colour is passed flat (it is constant over the quad anyway).

#include "scene_decode.wgsl"

// C++ twin: PointParams in src/webgpu/wgpu_points.cpp.
struct PointParams {
    mvp: mat4x4<f32>,            // proj * view * model (model = scene transform)
    viewportSize: vec2<f32>,     // pixels
    pointSizePx: f32,            // square side in pixels
    opacityThreshold: f32,       // gaussians with alpha below this are skipped
}
@group(1) @binding(0) var<uniform> pc: PointParams;

const SH_C0: f32 = 0.28209479177387814;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>,
    @location(1) corner: vec2<f32>,  // -1..1 inside the square (for round points)
}

@vertex
fn vs_points(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) id: u32) -> VertexOut {
    // Two triangles covering [-1, 1]^2 (var: indexed at run time).
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let pa = loadPosAlpha(id);
    let clip = pc.mvp * vec4<f32>(pa.xyz, 1.0);

    var out: VertexOut;
    // Culled: below the opacity threshold or behind the camera. A position outside the clip
    // volume (x > w) for all 6 vertices makes the whole quad disappear.
    if (pa.w < pc.opacityThreshold || clip.w <= 0.0) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.corner = vec2<f32>(0.0);
        return out;
    }

    // Offset in NDC: a full NDC range (2 units) spans viewportSize pixels, so a half size of
    // pointSizePx/2 pixels is pointSizePx / viewportSize NDC units. Times w: the rasterizer
    // divides by it.
    let corner = corners[vertexIndex];
    let p = clip + vec4<f32>(corner * pc.pointSizePx / pc.viewportSize * clip.w, 0.0, 0.0);
    out.position = vec4<f32>(p.x, -p.y, p.z, p.w);  // Vulkan clip space (+y down) -> WebGPU (+y up)

    // Degree-0 colour, clamped because it is the final displayed colour.
    out.color = clamp(0.5 + SH_C0 * loadDC(id), vec3<f32>(0.0), vec3<f32>(1.0));
    out.corner = corner;
    return out;
}

@fragment
fn fs_points(in: VertexOut) -> @location(0) vec4<f32> {
    if (dot(in.corner, in.corner) > 1.0) {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
