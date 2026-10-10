#version 460
// Phase 2 draw: evaluate one gaussian at this pixel (the INRIA renderCUDA inner loop)
// and output a PREMULTIPLIED colour. The blend state is "over" for premultiplied
// alpha: dst = src + (1 - src.a) * dst, applied back to front (see splat.vert).

layout(location = 0) flat in vec2 inCenter;
layout(location = 1) flat in vec4 inConicOpacity;
layout(location = 2) flat in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main() {
    // INRIA pixel coordinates: pixel (i, j) is at (i, j); gl_FragCoord is (i + 0.5, j + 0.5).
    vec2 d = inCenter - (gl_FragCoord.xy - 0.5);
    vec3 conic = inConicOpacity.xyz;
    float power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.0) discard;
    float alpha = min(0.99, inConicOpacity.w * exp(power));
    if (alpha < 1.0 / 255.0) discard;
    outColor = vec4(inColor * alpha, alpha);
}
