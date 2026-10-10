#version 460
// Point-cloud debug view: opaque round points (the square is cut to its inscribed disc).

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inCorner;
layout(location = 0) out vec4 outColor;

void main() {
    if (dot(inCorner, inCorner) > 1.0) discard;
    outColor = vec4(inColor, 1.0);
}
