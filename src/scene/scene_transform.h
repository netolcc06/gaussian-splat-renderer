// Scene transform: how the scene's own coordinates map into the viewer's world.
//
// The viewer (camera, UI, benchmark orbit) is right-handed with +Y up (see
// app/camera.h). 3DGS scenes are stored in the COLMAP/SfM frame of the capture, whose
// cameras use the OpenCV convention (x right, y DOWN, z forward). For a typical
// capture the photographer holds the camera upright, so the scene's "up" is roughly
// -Y in data space and the scene looks upside down in a +Y-up viewer.
//
// We fix this with a model matrix M (scene -> world) instead of rewriting the data:
//   world = M * data
// The default for real 3DGS scenes is a 180 degree rotation around X:
//   M = diag(1, -1, -1)      (x, y, z) -> (x, -y, -z)
// which maps data -Y to world +Y and keeps the frame right-handed (det = +1, a proper
// rotation: no mirroring, triangle winding and quaternion handedness are preserved).
//
// Why a matrix and not baked data:
//   - The PLY data stays bit-identical to the file, so the GPU path, the Phase 2 CPU
//     reference renderer and tests all see the same numbers.
//   - Renderers apply it as `proj * view * model`. It is a pure rotation, so for
//     splatting it can equally be folded into the view matrix: view' = view * M.
//   - Training cameras (cameras.json) live in data space too; they are mapped with the
//     same M (app/scene_camera.cpp), so `--camera-index` and the transform always agree.
//   - Pitfall for Phase 2: view-dependent colour (SH) must be evaluated with the
//     direction in DATA space (dir = mean - M^-1 * camPosWorld), because the SH
//     coefficients were fitted in data space. With view' = view * M this falls out
//     naturally (the camera position in data space is inverse(view')[3]).
#pragma once

#include <glm/glm.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gsr {

// Which data-space direction points "up" in the viewer.
//   PosY..NegZ: a coordinate axis.
//   Cameras:    the average up vector of the training cameras (cameras.json). The
//               transform is the NegY flip followed by the smallest rotation that takes
//               the (flipped) camera up onto +Y, so it equals NegY for a perfectly
//               level capture and only corrects the tilt otherwise (train_7k: 4.7 deg,
//               bonsai_30k: 40 deg, see docs/notes/phase1.md).
enum class UpAxis { PosY, NegY, PosZ, NegZ, Cameras };

// "+y", "-y", "+z", "-z", "cameras" (also "y" = "+y", "z" = "+z").
std::optional<UpAxis> parseUpAxis(std::string_view text);
const char* upAxisName(UpAxis up);

// Rotation (det +1) that maps the chosen data axis onto world +Y.
// For UpAxis::Cameras use sceneToWorldAligned (this returns the NegY flip).
glm::mat4 sceneToWorld(UpAxis up);

// NegY flip, then the minimal rotation taking `dataUp` (data space) onto world +Y.
glm::mat4 sceneToWorldAligned(const glm::vec3& dataUp);

// Default when the user does not pass --up: files written by our synthetic generator
// are already +Y up; a scene with training cameras uses their average up; anything
// else is assumed to be a COLMAP-frame 3DGS scene (-Y up).
UpAxis defaultUpAxis(const std::vector<std::string>& plyComments, bool hasTrainingCameras = false);

}  // namespace gsr
