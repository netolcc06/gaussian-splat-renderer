#include "app/scene_camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace gsr {

void transformBounds(const glm::mat4& model, const glm::vec3& dataMin, const glm::vec3& dataMax,
                     glm::vec3& worldMin, glm::vec3& worldMax) {
    worldMin = glm::vec3(std::numeric_limits<float>::infinity());
    worldMax = -worldMin;
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 p((corner & 1) ? dataMax.x : dataMin.x, (corner & 2) ? dataMax.y : dataMin.y,
                          (corner & 4) ? dataMax.z : dataMin.z);
        const glm::vec3 w(model * glm::vec4(p, 1.0f));
        worldMin = glm::min(worldMin, w);
        worldMax = glm::max(worldMax, w);
    }
}

Camera frameBounds(const glm::vec3& worldMin, const glm::vec3& worldMax, float fovYDeg) {
    Camera cam;
    cam.mode = CameraMode::Orbit;
    cam.fovYDeg = fovYDeg;
    const glm::vec3 center = 0.5f * (worldMin + worldMax);
    const float radius = std::max(0.5f * glm::length(worldMax - worldMin), 1e-3f);
    // The bounding sphere touches the top/bottom of the view at this distance.
    const float dist = radius / std::sin(glm::radians(fovYDeg) * 0.5f);
    cam.yawDeg = 30.0f;
    cam.pitchDeg = -20.0f;
    cam.distance = dist;
    cam.position = center - cam.forward() * dist;
    cam.nearPlane = std::max(dist * 1e-3f, 1e-3f);
    cam.farPlane = std::max(dist * 100.0f, 100.0f);
    return cam;
}

Camera cameraFromTraining(const TrainingCamera& tc, const glm::mat4& model, const glm::vec3& pivot,
                          float* rollDegOut) {
    const glm::mat3 m3(model);
    const glm::vec3 eye = glm::vec3(model * glm::vec4(tc.position, 1.0f));
    const glm::vec3 forward = glm::normalize(m3 * tc.forward());
    // Orbit distance: depth of the pivot along the view direction (at least a little).
    const float depth = glm::dot(pivot - eye, forward);
    const float dist = std::max(depth, 0.1f * std::max(glm::length(pivot - eye), 1e-2f));

    Camera cam;
    cam.mode = CameraMode::Orbit;
    cam.lookAt(eye, eye + forward * dist);
    cam.fovYDeg = tc.fovYDeg();
    cam.nearPlane = 0.01f;
    cam.farPlane = 1000.0f;

    if (rollDegOut) {
        // Angle between the training camera's up (-y axis) and ours, around forward.
        const glm::vec3 upTrain = glm::normalize(-(m3 * tc.down()));
        const glm::vec3 upOurs = cam.up();
        const float s = glm::dot(glm::cross(upOurs, upTrain), cam.forward());
        const float c = glm::dot(upOurs, upTrain);
        *rollDegOut = glm::degrees(std::atan2(s, c));
    }
    return cam;
}

SceneSetup resolveSceneSetup(const GaussianScene& scene, const std::filesystem::path& scenePath,
                             const std::string& camerasPath, const std::string& upAxisText) {
    SceneSetup setup;
    // Training cameras: explicit path, else cameras.json next to the .ply.
    std::filesystem::path path = camerasPath;
    if (path.empty()) {
        const std::filesystem::path sibling = scenePath.parent_path() / "cameras.json";
        if (std::filesystem::exists(sibling)) path = sibling;
    }
    if (!path.empty()) {
        setup.trainingCameras = loadTrainingCameras(path);
        setup.trainingCamerasPath = std::string(reinterpret_cast<const char*>(path.u8string().c_str()));  // UTF-8
    }

    setup.upAxisAuto = upAxisText.empty();
    if (setup.upAxisAuto) {
        setup.upAxis = defaultUpAxis(scene.comments, !setup.trainingCameras.empty());
    } else {
        const std::optional<UpAxis> up = parseUpAxis(upAxisText);
        if (!up) throw std::runtime_error("invalid up axis '" + upAxisText + "'");
        setup.upAxis = *up;
    }
    if (setup.upAxis == UpAxis::Cameras && setup.trainingCameras.empty())
        throw std::runtime_error("--up cameras needs training cameras (cameras.json next to the scene or --cameras)");
    setup.model = sceneModelMatrix(setup.upAxis, setup.trainingCameras);
    return setup;
}

glm::mat4 sceneModelMatrix(UpAxis up, const std::vector<TrainingCamera>& cameras) {
    return up == UpAxis::Cameras ? sceneToWorldAligned(meanCameraUp(cameras)) : sceneToWorld(up);
}

Camera framedSceneCamera(const GaussianScene& scene, const glm::mat4& model) {
    glm::vec3 lo, hi;
    transformBounds(model, scene.stats.robustMin, scene.stats.robustMax, lo, hi);
    return frameBounds(lo, hi);
}

Camera trainingCameraView(const SceneSetup& setup, uint32_t index) {
    const std::vector<TrainingCamera>& cams = setup.trainingCameras;
    if (index >= cams.size())
        throw std::runtime_error("--camera-index " + std::to_string(index) + " out of range (" +
                                 std::to_string(cams.size()) + " training cameras" +
                                 (cams.empty() ? ", no cameras.json found; use --cameras" : "") + ")");
    // Orbit pivot: the point all training cameras look at (data space -> world).
    const glm::vec3 pivot = glm::vec3(setup.model * glm::vec4(cameraFocusPoint(cams), 1.0f));
    float roll = 0.0f;
    Camera cam = cameraFromTraining(cams[index], setup.model, pivot, &roll);
    const TrainingCamera& tc = cams[index];
    std::printf("Camera: training camera %u ('%s', %ux%u, fov_y %.2f deg, dropped roll %.2f deg)\n", index,
                tc.imageName.c_str(), tc.width, tc.height, tc.fovYDeg(), roll);
    return cam;
}

Camera initialSceneCamera(const GaussianScene& scene, const SceneSetup& setup, const std::string& cameraPath,
                          std::optional<uint32_t> cameraIndex) {
    if (!cameraPath.empty()) return Camera::load(cameraPath);
    if (cameraIndex) return trainingCameraView(setup, *cameraIndex);
    return framedSceneCamera(scene, setup.model);
}

}  // namespace gsr
