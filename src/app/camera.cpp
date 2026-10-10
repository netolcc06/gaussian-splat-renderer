#include "app/camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace gsr {

namespace {
constexpr float kMaxPitch = 89.0f;  // avoid the singularity when looking straight up/down
const glm::vec3 kWorldUp(0.0f, 1.0f, 0.0f);

glm::vec3 vec3FromJson(const nlohmann::json& j) {
    if (!j.is_array() || j.size() != 3) throw std::runtime_error("camera: expected [x, y, z]");
    return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
}
}  // namespace

Camera Camera::defaultCamera() {
    Camera cam;
    cam.lookAt({2.0f, 1.6f, 2.4f}, {0.0f, 0.0f, 0.0f});
    return cam;
}

glm::vec3 Camera::forward() const {
    const float yaw = glm::radians(yawDeg);
    const float pitch = glm::radians(pitchDeg);
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
}

glm::vec3 Camera::right() const { return glm::normalize(glm::cross(forward(), kWorldUp)); }
glm::vec3 Camera::up() const { return glm::cross(right(), forward()); }

glm::mat4 Camera::view() const { return glm::lookAt(position, position + forward(), kWorldUp); }

glm::mat4 Camera::projection(float aspect) const {
    // GLM_FORCE_DEPTH_ZERO_TO_ONE (set in CMake) gives Vulkan's [0, 1] depth range.
    glm::mat4 proj = glm::perspective(glm::radians(fovYDeg), aspect, nearPlane, farPlane);
    proj[1][1] *= -1.0f;  // flip y: Vulkan clip space has +y pointing down
    return proj;
}

void Camera::orbit(float dYawDeg, float dPitchDeg) {
    const glm::vec3 pivot = target();
    yawDeg = std::fmod(yawDeg + dYawDeg, 360.0f);
    pitchDeg = std::clamp(pitchDeg + dPitchDeg, -kMaxPitch, kMaxPitch);
    position = pivot - forward() * distance;
}

void Camera::pan(float dRight, float dUp) { position += right() * dRight + up() * dUp; }

void Camera::zoom(float scrollSteps) {
    const glm::vec3 pivot = target();
    distance = std::clamp(distance * std::pow(0.9f, scrollSteps), 1e-3f, 1e5f);
    position = pivot - forward() * distance;
}

void Camera::look(float dYawDeg, float dPitchDeg) {
    yawDeg = std::fmod(yawDeg + dYawDeg, 360.0f);
    pitchDeg = std::clamp(pitchDeg + dPitchDeg, -kMaxPitch, kMaxPitch);
}

void Camera::move(const glm::vec3& localDelta) {
    position += right() * localDelta.x + kWorldUp * localDelta.y + forward() * localDelta.z;
}

void Camera::lookAt(const glm::vec3& eye, const glm::vec3& targetPoint) {
    const glm::vec3 d = targetPoint - eye;
    const float len = glm::length(d);
    if (len < 1e-6f) throw std::runtime_error("camera: position and target coincide");
    const glm::vec3 f = d / len;
    position = eye;
    distance = len;
    pitchDeg = std::clamp(glm::degrees(std::asin(std::clamp(f.y, -1.0f, 1.0f))), -kMaxPitch, kMaxPitch);
    yawDeg = glm::degrees(std::atan2(f.x, -f.z));
}

nlohmann::json Camera::toJson() const {
    const glm::vec3 t = target();
    return {
        {"mode", mode == CameraMode::Orbit ? "orbit" : "fly"},
        {"position", {position.x, position.y, position.z}},
        {"target", {t.x, t.y, t.z}},
        {"yaw_deg", yawDeg},
        {"pitch_deg", pitchDeg},
        {"fov_y_deg", fovYDeg},
        {"near", nearPlane},
        {"far", farPlane},
    };
}

Camera Camera::fromJson(const nlohmann::json& j) {
    Camera cam = defaultCamera();
    if (j.contains("mode")) {
        const std::string m = j["mode"].get<std::string>();
        if (m == "orbit") cam.mode = CameraMode::Orbit;
        else if (m == "fly") cam.mode = CameraMode::Fly;
        else throw std::runtime_error("camera: mode must be \"orbit\" or \"fly\"");
    }
    // Orientation: "target" wins if present (look-at); otherwise yaw/pitch are used.
    if (j.contains("position")) cam.position = vec3FromJson(j["position"]);
    if (j.contains("target")) {
        cam.lookAt(cam.position, vec3FromJson(j["target"]));
    } else {
        if (j.contains("yaw_deg")) cam.yawDeg = j["yaw_deg"].get<float>();
        if (j.contains("pitch_deg")) cam.pitchDeg = std::clamp(j["pitch_deg"].get<float>(), -kMaxPitch, kMaxPitch);
        if (j.contains("distance")) cam.distance = j["distance"].get<float>();
    }
    if (j.contains("fov_y_deg")) cam.fovYDeg = j["fov_y_deg"].get<float>();
    if (j.contains("near")) cam.nearPlane = j["near"].get<float>();
    if (j.contains("far")) cam.farPlane = j["far"].get<float>();
    if (!(cam.nearPlane > 0.0f && cam.farPlane > cam.nearPlane && cam.fovYDeg > 0.0f && cam.fovYDeg < 180.0f))
        throw std::runtime_error("camera: invalid fov/near/far");
    return cam;
}

Camera Camera::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open camera file: " + path);
    try {
        return fromJson(nlohmann::json::parse(file));
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Invalid camera file " + path + ": " + e.what());
    }
}

void Camera::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write camera file: " + path);
    file << toJson().dump(2) << "\n";
}

}  // namespace gsr
