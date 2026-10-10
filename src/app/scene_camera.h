// Cameras derived from a loaded scene: training-camera presets and auto-framing.
// Both work in WORLD space, i.e. after the scene transform (scene/scene_transform.h).
#pragma once

#include "app/camera.h"
#include "scene/gaussian_scene.h"
#include "scene/scene_transform.h"
#include "scene/training_cameras.h"

#include <glm/glm.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gsr {

// World-space axis-aligned box of a data-space box under `model`. Exact for our
// axis-permuting rotations; conservative for any other affine transform.
void transformBounds(const glm::mat4& model, const glm::vec3& dataMin, const glm::vec3& dataMax,
                     glm::vec3& worldMin, glm::vec3& worldMax);

// A camera that sees the whole world-space box: looks at its centre from a fixed
// three-quarter direction (yaw 30, pitch -20), at the distance where the bounding
// sphere fits the vertical field of view.
Camera frameBounds(const glm::vec3& worldMin, const glm::vec3& worldMax, float fovYDeg = 60.0f);

// The view of a training camera (cameras.json), mapped through `model`.
// Position and viewing direction are exact; the vertical FOV comes from fy/height.
// Our Camera has no roll (world up is always +Y), so any roll of the training camera
// around its viewing axis is dropped; `rollDegOut` reports how much that was.
// `pivot` (world space) sets the orbit distance: its depth along the view direction.
Camera cameraFromTraining(const TrainingCamera& tc, const glm::mat4& model, const glm::vec3& pivot,
                          float* rollDegOut = nullptr);

// ---------------------------------------------------------------------------------
// Scene setup shared by gsplat and tools/gsplat_ref, so that the same command-line
// flags (--cameras, --up, --camera, --camera-index) give both programs the exact same
// scene transform and camera. That is what makes GPU-vs-reference diffs meaningful.
// ---------------------------------------------------------------------------------
struct SceneSetup {
    std::vector<TrainingCamera> trainingCameras;
    std::string trainingCamerasPath;  // empty if none
    UpAxis upAxis = UpAxis::PosY;
    bool upAxisAuto = true;           // chosen by defaultUpAxis (no --up)
    glm::mat4 model{1.0f};            // scene -> world
};

// Loads the training cameras (`camerasPath`, else cameras.json next to the scene, else
// none) and resolves the up axis (`upAxis` text, empty = automatic). Throws on errors.
SceneSetup resolveSceneSetup(const GaussianScene& scene, const std::filesystem::path& scenePath,
                             const std::string& camerasPath, const std::string& upAxis);

// The scene transform for an up axis (Cameras uses the mean training-camera up).
glm::mat4 sceneModelMatrix(UpAxis up, const std::vector<TrainingCamera>& cameras);

// Auto-framing of the robust bounding box (see frameBounds).
Camera framedSceneCamera(const GaussianScene& scene, const glm::mat4& model);

// Training camera `index` as a viewer Camera (orbit pivot: the cameras' focus point).
// Prints one line describing it. Throws if the index is out of range.
Camera trainingCameraView(const SceneSetup& setup, uint32_t index);

// The start camera: `cameraPath` (JSON file) if given, else training camera
// `cameraIndex` if given, else auto-framing.
Camera initialSceneCamera(const GaussianScene& scene, const SceneSetup& setup, const std::string& cameraPath,
                          std::optional<uint32_t> cameraIndex);

}  // namespace gsr
