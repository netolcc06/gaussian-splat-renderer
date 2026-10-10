#version 460
// Phase 0 graphics pass: a colored unit cube transformed by one matrix.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;  // projection * view (Vulkan clip space: y down, z in [0, 1])
} pc;

layout(location = 0) out vec3 outColor;

void main() {
    outColor = inColor;
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
}
