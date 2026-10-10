// Training cameras from an INRIA `cameras.json` (utils/camera_utils.py, camera_to_JSON):
//   [{ "id", "img_name", "width", "height",
//      "position": [3]        camera centre in data (COLMAP world) space
//      "rotation": [3][3]     camera-to-world rotation, row-major; its COLUMNS are the
//                             camera axes in world space (OpenCV: x right, y down, z forward)
//      "fx", "fy" }]          focal lengths in pixels for width x height
// Values stay in data space; app/scene_camera.cpp maps them through the scene transform.
#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace gsr {

struct TrainingCamera {
    int id = 0;
    std::string imageName;
    uint32_t width = 0, height = 0;
    glm::vec3 position{0.0f};
    glm::mat3 cameraToWorld{1.0f};  // GLM is column-major: cameraToWorld[1] = camera y (down) axis
    float fx = 0.0f, fy = 0.0f;

    glm::vec3 right() const { return cameraToWorld[0]; }
    glm::vec3 down() const { return cameraToWorld[1]; }
    glm::vec3 forward() const { return cameraToWorld[2]; }
    float fovYDeg() const;  // 2 * atan(height / (2 * fy))
};

// Throws std::runtime_error on I/O or format errors.
std::vector<TrainingCamera> loadTrainingCameras(const std::filesystem::path& path);

// Normalized average "up" (-y camera axis) of all cameras, in data space. For a
// hand-held or tripod capture this is a good estimate of the scene's true up.
glm::vec3 meanCameraUp(const std::vector<TrainingCamera>& cameras);

// The point closest (least squares) to all optical axes, in data space: what the
// cameras look at. For an object-centric capture this is the object; used as the
// orbit pivot of camera presets. Falls back to the camera centroid if degenerate.
glm::vec3 cameraFocusPoint(const std::vector<TrainingCamera>& cameras);

}  // namespace gsr
