// Application: window, frame loop, input, UI and the automation modes.
//
// Frame structure (all on one queue, one command buffer per frame):
//
//   renderer->render()          -> offscreen color target (RGBA8) + depth
//   [screenshot] copy target    -> host-visible readback buffer
//   blit target                 -> swapchain image
//   ImGui                       -> swapchain image (on top)
//   present
//
// Rendering into an offscreen target instead of the swapchain keeps the renderers
// independent of the swapchain format/usage (storage writes, fixed size for
// benchmarks) and lets screenshots exclude the UI.
#pragma once

#include "app/camera.h"
#include "app/cli.h"
#include "app/scene_camera.h"
#include "app/scene_loading.h"
#include "core/context.h"
#include "core/profiler.h"
#include "core/resources.h"
#include "core/swapchain.h"
#include "render/gpu_scene.h"
#include "render/renderer.h"
#include "scene/gaussian_scene.h"
#include "scene/scene_transform.h"
#include "scene/training_cameras.h"

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

namespace gsr {

class SplatTileRenderer;

class Application {
public:
    explicit Application(Options options);
    ~Application();
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Runs until the window closes (interactive) or the requested frames are done
    // (automation). Returns the process exit code.
    int run();

private:
    // Two frames in flight: the CPU records frame N+1 while the GPU executes frame N.
    static constexpr uint32_t kFramesInFlight = 2;

    struct FrameData {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAcquired = VK_NULL_HANDLE;  // binary: signaled by vkAcquireNextImageKHR
        uint64_t timelineValue = 0;  // value m_frameTimeline reaches when this frame's work is done
    };

    void initWindow();
    void initVulkan();
    void initImGui();
    void shutdown();

    // Scene (CPU load before any Vulkan work, so a bad file fails fast).
    void loadScene();
    CompressionOptions compressionOptions() const;  // the --scene-* options (Phase 5)
    // Encodes (unless `preEncoded`) + uploads m_scene with compressionOptions().
    void uploadScene(EncodedScene* preEncoded = nullptr);

    // Runtime scene loading (UI: Open scene..., Ctrl+O, drag and drop, recent scenes;
    // automation: --load-after). requestSceneLoad() is the single entry point of all of
    // them: it starts the worker thread (scene_loading.h); pollSceneLoad(), called between
    // frames, installs the result.
    void requestSceneLoad(const std::filesystem::path& path);
    void pollSceneLoad();
    bool sceneLoadPending() const { return m_loadJob.busy() || !m_queuedScenePath.empty(); }
    // Swaps a loaded scene in; on any failure restores the previous scene and returns false.
    bool installScene(LoadedScene& loaded);
    std::string rendererForNewScene() const;
    void waitForSubmittedFrames();  // host wait until every submitted frame has finished
    void openSceneDialog();
    void addRecentScene(const std::filesystem::path& path);
    void loadRecentScenes();
    void saveRecentScenes();
    void buildCompressionUi();
    void setUpAxis(UpAxis up);
    Camera initialSceneCamera() const;
    // Creates m_renderer by name ("test" / "points"). Waits for the GPU if one exists.
    void createRenderer(const std::string& name);

    void createRenderTargets(VkExtent2D extent);
    void destroyRenderTargets();
    void recreateSwapchain();

    // Records, submits and presents one frame. Returns false if the frame was skipped
    // (swapchain out of date); the caller simply tries again.
    bool drawFrame(bool captureScreenshot);
    void recordFrame(VkCommandBuffer cmd, uint32_t imageIndex, bool captureScreenshot);

    void processInput(float dt);
    void buildUi();
    void buildSceneUi();
    void buildSceneLoadUi();

    int runInteractive();
    int runAutomated();
    void writeScreenshot();
    void writePipelineStats();  // Phase 6 --pipeline-stats
    void writeBenchmark(const std::vector<double>& cpuFrameMs, uint32_t warmupFrames);
    void writeTileStats(const SplatTileRenderer& tiles);  // --tile-stats

    Options m_options;
    GLFWwindow* m_window = nullptr;
    bool m_framebufferResized = false;
    bool m_swapchainDirty = false;  // e.g. vsync toggled in the UI

    VulkanContext m_ctx;
    Swapchain m_swapchain;
    std::array<FrameData, kFramesInFlight> m_frames;
    // Timeline semaphore: a 64-bit counter the GPU increments as frames finish.
    // Waiting for "counter >= value" replaces one VkFence per frame in flight.
    VkSemaphore m_frameTimeline = VK_NULL_HANDLE;
    uint64_t m_timelineValue = 0;
    uint64_t m_frameIndex = 0;  // frames submitted so far

    // Offscreen render targets (see the file comment) and their last-use state.
    static constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
    Image m_color;
    Image m_depth;
    ImageState m_colorState{};
    ImageState m_depthState{};
    Buffer m_readback;  // screenshot destination (host-visible)

    std::unique_ptr<Renderer> m_renderer;
    std::string m_pendingRenderer;  // UI request, applied between frames
    bool m_pendingSceneUpload = false;  // UI changed the scene format: re-encode + re-upload between frames

    // Scene: CPU copy (stats, future CPU reference), GPU buffers, transform, cameras.
    std::unique_ptr<GaussianScene> m_scene;
    std::filesystem::path m_scenePath;
    GpuScene m_gpuScene;
    SceneSetup m_setup;  // training cameras, up axis, model matrix (scene -> world)
    int m_uiCameraIndex = 0;
    GpuProfiler m_profiler;
    CpuProfiler m_cpuProfiler;  // CPU stages inside renderers (e.g. depth sort)
    Camera m_camera;
    Camera m_initialCamera;

    // UI / input state
    bool m_imguiInitialized = false;
    VkFormat m_swapchainFormatForImGui = VK_FORMAT_UNDEFINED;
    double m_lastMouseX = 0.0, m_lastMouseY = 0.0;
    bool m_tabWasDown = false;
    float m_scrollDelta = 0.0f;  // accumulated by the GLFW scroll callback
    float m_flySpeed = 2.0f;  // world units per second
    std::string m_statusMessage;

    // Runtime scene loading state.
    SceneLoadJob m_loadJob;                   // the background load (one at a time)
    std::filesystem::path m_queuedScenePath;  // requested while a load was running: started next
    std::string m_sceneMessage;               // last load result for the UI ("Loaded ..." / error)
    bool m_sceneMessageIsError = false;
    bool m_openDialogRequested = false;       // button / Ctrl+O: the dialog opens between frames
    std::vector<std::filesystem::path> m_droppedPaths;  // GLFW drop callback -> next frame
    std::vector<std::filesystem::path> m_recentScenes;  // most recent first
    static constexpr size_t kMaxRecentScenes = 8;
};

}  // namespace gsr
