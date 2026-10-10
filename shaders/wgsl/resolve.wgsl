// Resolve pass: copies the rgba16float render target to an 8-bit color target, pixel for
// pixel. It replaces the native vkCmdBlitImage (RGBA16F -> swapchain): WebGPU has no blit and
// copyTextureToTexture needs matching formats. Two pipelines use this module: one writes the
// canvas texture (bgra8unorm on Windows), one writes the rgba8unorm capture texture read back
// by the tests. Both targets are non-sRGB views, as on native: blending happens in
// sRGB-encoded space, so the values are stored as they are (clamped to [0, 1] by the unorm
// conversion).

#include "fullscreen.wgsl"

@group(0) @binding(0) var source: texture_2d<f32>;

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // position.xy is the pixel center (x + 0.5, y + 0.5), top-left origin, so the
    // conversion to integer pixel coordinates truncates to the pixel itself.
    return textureLoad(source, vec2<i32>(position.xy), 0);
}
