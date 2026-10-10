// Loading a scene file, shared by both ways the app gets a scene:
//   - the command line (--scene): loadSceneFile() on the main thread, before the window opens;
//   - the UI (Open scene..., Ctrl+O, drag and drop, recent scenes, and the --load-after
//     automation hook): the same loadSceneFile() on a worker thread (SceneLoadJob), while
//     the main thread keeps rendering the current scene. The Application then swaps the
//     result in between two frames (Application::installScene).
// Everything here is CPU-only: parsing, per-scene setup (cameras.json, up axis) and,
// on the worker, the GPU encoding of the compressed scene formats.
#pragma once

#include "app/scene_camera.h"
#include "scene/gaussian_scene.h"
#include "scene/scene_compression.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>

namespace gsr {

struct SceneLoadRequest {
    std::filesystem::path path;
    std::string camerasPath;  // --cameras (empty: cameras.json next to the scene, if present)
    std::string upAxis;       // --up (empty: automatic, see defaultUpAxis)
};

struct LoadedScene {
    std::filesystem::path path;
    std::unique_ptr<GaussianScene> scene;
    SceneSetup setup;                     // training cameras, up axis, model matrix
    std::optional<EncodedScene> encoded;  // pre-encoded GPU arrays (see encodeLoadedScene)
    CompressionOptions encodedWith;       // the options `encoded` was made with
};

// Parses the .ply, prints the scene summary and resolves the scene setup exactly like
// the command line always did (resolveSceneSetup: cameras.json, up axis). Throws
// PlyError (bad or unsupported file), std::runtime_error (I/O, cameras.json) or
// std::bad_alloc.
LoadedScene loadSceneFile(const SceneLoadRequest& request);

// Runs the CPU half of GpuScene::upload ahead of time (encodeScene, e.g. the k-means
// codebook of --scene-format cb), unless the format needs none (zero-copy f32 soa).
void encodeLoadedScene(LoadedScene& loaded, const CompressionOptions& options);

// One background load at a time. Cancel-safe: cancel() never blocks (the parser has no
// interruption points, so the thread runs to the end of its current step and its result
// is discarded); the destructor waits for the thread.
class SceneLoadJob {
public:
    enum class Stage : int { Idle, Parsing, Encoding, Finished };

    struct Result {
        bool cancelled = false;
        std::string error;  // empty = success
        LoadedScene loaded;
    };

    SceneLoadJob() = default;
    ~SceneLoadJob() { wait(); }
    SceneLoadJob(const SceneLoadJob&) = delete;
    SceneLoadJob& operator=(const SceneLoadJob&) = delete;

    // Starts loadSceneFile (+ encodeLoadedScene with `encode`) on a new thread.
    // Precondition: !busy().
    void start(SceneLoadRequest request, const CompressionOptions& encode);
    // A load was started and its result has not been taken yet.
    bool busy() const { return m_active; }
    bool finished() const { return m_stage.load(std::memory_order_acquire) == int(Stage::Finished); }
    void cancel() { m_cancel.store(true); }
    bool cancelling() const { return m_cancel.load(); }
    // The result once finished() (joins the thread); std::nullopt while still running.
    std::optional<Result> take();
    void wait();  // blocks until the thread is done (result kept for take())

    Stage stage() const { return Stage(m_stage.load(std::memory_order_acquire)); }
    const std::filesystem::path& path() const { return m_request.path; }
    double elapsedSeconds() const;

private:
    void run();

    SceneLoadRequest m_request;
    CompressionOptions m_encode;
    std::thread m_thread;
    bool m_active = false;  // main thread only
    std::atomic<bool> m_cancel{false};
    std::atomic<int> m_stage{int(Stage::Idle)};
    std::chrono::steady_clock::time_point m_start{};
    Result m_result;  // written by the worker before m_stage becomes Finished (release)
};

}  // namespace gsr
