#include "scene/scene_transform.h"

#include <vector>

namespace gsr {

std::optional<UpAxis> parseUpAxis(std::string_view text) {
    if (text == "+y" || text == "y") return UpAxis::PosY;
    if (text == "-y") return UpAxis::NegY;
    if (text == "+z" || text == "z") return UpAxis::PosZ;
    if (text == "-z") return UpAxis::NegZ;
    if (text == "cameras") return UpAxis::Cameras;
    return std::nullopt;
}

const char* upAxisName(UpAxis up) {
    switch (up) {
        case UpAxis::PosY: return "+y";
        case UpAxis::NegY: return "-y";
        case UpAxis::PosZ: return "+z";
        case UpAxis::NegZ: return "-z";
        case UpAxis::Cameras: return "cameras";
    }
    return "?";
}

glm::mat4 sceneToWorld(UpAxis up) {
    // Columns are the images of the data axes e_x, e_y, e_z in world space.
    // Every matrix is a rotation (det = +1), never a mirror.
    glm::mat4 m(1.0f);
    switch (up) {
        case UpAxis::PosY:  // identity
            break;
        case UpAxis::NegY:  // 180 deg around X: (x, y, z) -> (x, -y, -z)
        case UpAxis::Cameras:
            m[1] = glm::vec4(0, -1, 0, 0);
            m[2] = glm::vec4(0, 0, -1, 0);
            break;
        case UpAxis::PosZ:  // -90 deg around X: (x, y, z) -> (x, z, -y)
            m[1] = glm::vec4(0, 0, -1, 0);
            m[2] = glm::vec4(0, 1, 0, 0);
            break;
        case UpAxis::NegZ:  // +90 deg around X: (x, y, z) -> (x, -z, y)
            m[1] = glm::vec4(0, 0, 1, 0);
            m[2] = glm::vec4(0, -1, 0, 0);
            break;
    }
    return m;
}

glm::mat4 sceneToWorldAligned(const glm::vec3& dataUp) {
    const glm::mat4 flip = sceneToWorld(UpAxis::NegY);
    const glm::vec3 u = glm::normalize(glm::mat3(flip) * dataUp);  // ~ +Y for a level capture
    const glm::vec3 y(0.0f, 1.0f, 0.0f);
    const float c = glm::dot(u, y);
    const glm::vec3 axis = glm::cross(u, y);
    const float s = glm::length(axis);
    if (s < 1e-6f) return c > 0.0f ? flip : glm::mat4(1.0f);  // already aligned / exactly opposite
    // Rodrigues: rotation by angle atan2(s, c) around axis/s.
    const glm::vec3 k = axis / s;
    const glm::mat3 K(glm::vec3(0, k.z, -k.y), glm::vec3(-k.z, 0, k.x), glm::vec3(k.y, -k.x, 0));  // columns
    const glm::mat3 R = glm::mat3(1.0f) + s * K + (1.0f - c) * (K * K);
    return glm::mat4(R) * flip;
}

UpAxis defaultUpAxis(const std::vector<std::string>& plyComments, bool hasTrainingCameras) {
    for (const std::string& c : plyComments)
        if (c.find("generator=tools/gen_synthetic_ply.py") != std::string::npos) return UpAxis::PosY;
    return hasTrainingCameras ? UpAxis::Cameras : UpAxis::NegY;
}

}  // namespace gsr
