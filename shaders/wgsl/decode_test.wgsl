// Self-test of scene_decode.wgsl (gsr_selftest("decode"), web phase W1): decodes every gaussian
// of an uploaded scene into a float buffer that the C++ side compares with decodeGaussian(), the
// CPU decoder (src/scene/scene_compression.cpp). The web counterpart of the decoder check in
// tests/test_scene_formats.cpp. Output per gaussian: 20 vec4 =
//   posAlpha | sigma, degree | q (w, x, y, z) | DC, 0 | SH coefficient 0..15, 0.

#include "scene_decode.wgsl"

struct DecodeParams {
    count: u32,
    used: u32,  // SH coefficients requested from loadSH
    pad0: u32,
    pad1: u32,
}
@group(1) @binding(0) var<uniform> params: DecodeParams;
@group(1) @binding(1) var<storage, read_write> outData: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let base = i * 20u;
    outData[base] = loadPosAlpha(i);
    let shape = loadShape(i);
    outData[base + 1u] = vec4<f32>(shape.sigma, f32(gaussianShDegree(i)));
    outData[base + 2u] = shape.q;
    outData[base + 3u] = vec4<f32>(loadDC(i), 0.0);
    var sh = loadSH(i, params.used);  // var: indexed at run time
    for (var k = 0u; k < 16u; k++) {
        outData[base + 4u + k] = vec4<f32>(sh[k], 0.0);
    }
}
