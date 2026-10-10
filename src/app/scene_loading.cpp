#include "app/scene_loading.h"

#include "app/file_dialog.h"
#include "render/gpu_scene.h"
#include "scene/ply_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <new>

namespace gsr {

LoadedScene loadSceneFile(const SceneLoadRequest& request) {
    const std::string name = pathToUtf8(request.path);
    std::printf("Loading %s ...\n", name.c_str());
    std::fflush(stdout);
    LoadedScene loaded;
    loaded.path = request.path;
    loaded.scene = std::make_unique<GaussianScene>(loadGaussianPly(request.path));
    const GaussianScene& sc = *loaded.scene;
    const SceneStats& st = sc.stats;
    const double mb = double(std::filesystem::file_size(request.path)) / (1024.0 * 1024.0);
    std::printf("Scene: %u gaussians, SH degree %d, %.1f MB file\n", sc.count, sc.shDegree, mb);
    std::printf("  load %.1f ms (map %.1f, header %.2f, decode %.1f, stats %.1f) = %.0f MB/s\n", sc.timings.totalMs,
                sc.timings.mapMs, sc.timings.headerMs, sc.timings.decodeMs, sc.timings.statsMs,
                mb / (sc.timings.totalMs * 1e-3));
    std::printf("  bbox   [%.3f %.3f %.3f] .. [%.3f %.3f %.3f]\n", st.bboxMin.x, st.bboxMin.y, st.bboxMin.z,
                st.bboxMax.x, st.bboxMax.y, st.bboxMax.z);
    std::printf("  p1-p99 [%.3f %.3f %.3f] .. [%.3f %.3f %.3f]\n", st.robustMin.x, st.robustMin.y, st.robustMin.z,
                st.robustMax.x, st.robustMax.y, st.robustMax.z);
    std::printf("  alpha %.4g .. %.4g (mean %.3f, %u below 1/255), sigma %.3g .. %.3g (median max-axis %.3g)\n",
                st.alphaMin, st.alphaMax, st.alphaMean, st.alphaBelowVisible, st.sigmaMin, st.sigmaMax,
                st.sigmaMaxMedian);
    for (const std::string& w : sc.warnings) std::printf("  warning: %s\n", w.c_str());

    // Training cameras + up axis: the same resolution as tools/gsplat_ref (scene_camera.h).
    loaded.setup = resolveSceneSetup(sc, request.path, request.camerasPath, request.upAxis);
    const SceneSetup& setup = loaded.setup;
    if (!setup.trainingCameras.empty())
        std::printf("  %zu training cameras from %s\n", setup.trainingCameras.size(), setup.trainingCamerasPath.c_str());
    std::printf("  up axis %s%s", upAxisName(setup.upAxis), setup.upAxisAuto ? " (auto)" : "");
    if (setup.upAxis == UpAxis::Cameras) {
        const glm::vec3 u = meanCameraUp(setup.trainingCameras);
        std::printf(": mean camera up (%.3f %.3f %.3f), %.1f deg from -y", u.x, u.y, u.z,
                    glm::degrees(std::acos(std::clamp(-u.y, -1.0f, 1.0f))));
    }
    std::printf("\n");
    std::fflush(stdout);
    return loaded;
}

void encodeLoadedScene(LoadedScene& loaded, const CompressionOptions& options) {
    loaded.encoded.reset();
    loaded.encodedWith = options;
    if (!loaded.scene || GpuScene::zeroCopy(*loaded.scene, options)) return;
    loaded.encoded = encodeScene(*loaded.scene, options);
}

// ---------------------------------------------------------------------------
// SceneLoadJob
// ---------------------------------------------------------------------------

void SceneLoadJob::start(SceneLoadRequest request, const CompressionOptions& encode) {
    wait();  // a previous, already taken job may still own a joinable handle (never in practice)
    m_request = std::move(request);
    m_encode = encode;
    m_result = Result{};
    m_cancel.store(false);
    m_stage.store(int(Stage::Parsing), std::memory_order_release);
    m_start = std::chrono::steady_clock::now();
    m_active = true;
    m_thread = std::thread([this] { run(); });
}

void SceneLoadJob::run() {
    // Everything that can go wrong becomes a message: an exception escaping a thread
    // function would call std::terminate.
    try {
        LoadedScene loaded = loadSceneFile(m_request);
        if (!m_cancel.load()) {
            m_stage.store(int(Stage::Encoding), std::memory_order_release);
            encodeLoadedScene(loaded, m_encode);
        }
        m_result.loaded = std::move(loaded);
    } catch (const PlyError& e) {
        m_result.error = std::string("not a supported 3DGS .ply: ") + e.what();
    } catch (const std::bad_alloc&) {
        m_result.error = "out of memory while loading (scene too large for this machine?)";
    } catch (const std::exception& e) {
        m_result.error = e.what();
    } catch (...) {
        m_result.error = "unknown error";
    }
    m_result.cancelled = m_cancel.load();
    if (m_result.cancelled) m_result.loaded = LoadedScene{};  // free it here, not on the main thread
    m_stage.store(int(Stage::Finished), std::memory_order_release);
}

std::optional<SceneLoadJob::Result> SceneLoadJob::take() {
    if (!m_active || !finished()) return std::nullopt;
    wait();
    m_active = false;
    m_stage.store(int(Stage::Idle), std::memory_order_release);
    return std::move(m_result);
}

void SceneLoadJob::wait() {
    if (m_thread.joinable()) m_thread.join();
}

double SceneLoadJob::elapsedSeconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
}

}  // namespace gsr
