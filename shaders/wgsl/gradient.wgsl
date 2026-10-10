// W0 test pattern (web counterpart of shaders/gradient.comp, not a port of it): fills the
// rgba16float render target with an integer gradient that the automated web test checks
// exactly (tools/web/run_web_checks.mjs, expectedGradient()). Every channel is k / 255 for an
// integer k, which survives the fp16 store and the unorm8 resolve without rounding changes:
// the fp16 error of k / 255 is at most 2^-12 relative, i.e. < 0.07 of an 8-bit step.

@group(0) @binding(0) var outImage: texture_storage_2d<rgba16float, write>;

// The 8-bit value of each channel at `pixel` for a target of `size` pixels.
fn gradientValue(pixel: vec2<u32>, size: vec2<u32>) -> vec4<u32> {
    let denom = max(size - vec2<u32>(1u), vec2<u32>(1u));
    let r = (pixel.x * 255u) / denom.x;      // 0 at the left edge, 255 at the right edge
    let g = (pixel.y * 255u) / denom.y;      // 0 at the top row, 255 at the bottom row
    let b = (pixel.x + pixel.y) & 255u;      // diagonal ramp: catches a 1-pixel shift
    return vec4<u32>(r, g, b, 255u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(outImage);
    if (id.x >= size.x || id.y >= size.y) {
        return;  // partial edge workgroups
    }
    let value = gradientValue(id.xy, size);
    textureStore(outImage, vec2<i32>(id.xy), vec4<f32>(value) / 255.0);
}
