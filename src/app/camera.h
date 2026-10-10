// Camera with two interaction modes sharing one state.
//
// State: position, yaw, pitch, distance (to the orbit target), fov, near, far.
// Conventions: right-handed world, +Y up. yaw = 0, pitch = 0 looks down -Z.
// Positive yaw turns right (toward +X), positive pitch looks up.
//
//   Orbit: the target (= position + forward * distance) stays fixed; dragging changes
//          yaw/pitch and moves the camera around it; scroll changes the distance.
//   Fly:   the position moves (WASD/QE) and mouse look changes yaw/pitch in place.
//
// Because the target is always derived the same way, switching modes never jumps.
#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json_fwd.hpp>

#include <string>

namespace gsr {

enum class CameraMode { Orbit, Fly };

class Camera {
public:
    CameraMode mode = CameraMode::Orbit;
    glm::vec3 position{0.0f, 0.0f, 3.0f};
    float yawDeg = 0.0f;
    float pitchDeg = 0.0f;
    float distance = 3.0f;  // orbit radius
    float fovYDeg = 60.0f;  // vertical field of view
    float nearPlane = 0.01f;
    float farPlane = 1000.0f;

    // The default view of the Phase 0 test scene (cube at the origin seen from +X+Y+Z).
    static Camera defaultCamera();

    glm::vec3 forward() const;
    glm::vec3 right() const;
    glm::vec3 up() const;
    glm::vec3 target() const { return position + forward() * distance; }

    glm::mat4 view() const;
    // Perspective projection for Vulkan: depth in [0, 1] and y flipped (Vulkan's
    // framebuffer y axis points down, unlike OpenGL's NDC).
    glm::mat4 projection(float aspect) const;

    // Orbit mode: rotate around the target / move the target / change the radius.
    void orbit(float dYawDeg, float dPitchDeg);
    void pan(float dRight, float dUp);  // in world units
    void zoom(float scrollSteps);
    // Fly mode: rotate in place / move along camera axes (world units).
    void look(float dYawDeg, float dPitchDeg);
    void move(const glm::vec3& localDelta);  // x = right, y = up, z = forward

    // Points the camera from `eye` at `target` (sets yaw/pitch/distance).
    void lookAt(const glm::vec3& eye, const glm::vec3& target);

    // JSON (see README for the format). fromJson throws on malformed input.
    nlohmann::json toJson() const;
    static Camera fromJson(const nlohmann::json& j);
    static Camera load(const std::string& path);
    void save(const std::string& path) const;
};

}  // namespace gsr
