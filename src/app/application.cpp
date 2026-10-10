#include "app/application.h"

#include "app/file_dialog.h"
#include "app/image_io.h"
#include "app/scene_camera.h"
#include "core/memory_report.h"
#include "core/pipelines.h"
#include "render/point_cloud_renderer.h"
#include "render/splat_gpusort_renderer.h"
#include "render/splat_renderer.h"
#include "render/splat_tile_renderer.h"
#include "render/test_renderer.h"
#include "scene/ply_loader.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <utility>

namespace gsr {

namespace {

void checkImGuiVk(VkResult result) {
    if (result != VK_SUCCESS) throwVkError(result, "ImGui Vulkan backend", __FILE__, __LINE__);
}

const char* presentModeName(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "IMMEDIATE";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_KHR: return "FIFO";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
        default: return "OTHER";
    }
}

#ifdef NDEBUG
constexpr bool kDebugBuild = false;
#else
constexpr bool kDebugBuild = true;
#endif

// The Phase 5 memory report as JSON: per buffer, per owner, totals, VMA heaps.
nlohmann::json memoryReportJson(const MemoryReport& r) {
    using nlohmann::json;
    json buffers = json::array();
    for (const MemoryRecord& m : r.records)
        buffers.push_back({{"owner", m.owner},
                           {"name", m.name},
                           {"mb", toMiB(m.allocated)},
                           {"requested_bytes", m.requested},
                           {"allocated_bytes", m.allocated},
                           {"usage", m.image ? imageUsageString(m.usage) : bufferUsageString(m.usage)},
                           {"memory", memoryKindName(m.memoryKind)}});
    json owners = json::object();
    for (const MemoryOwnerTotal& o : r.owners)
        owners[o.owner] = {{"mb", toMiB(o.allocated)}, {"buffers", o.count}};
    json heaps = json::array();
    for (const MemoryHeapInfo& h : r.heaps)
        heaps.push_back({{"heap", h.index},
                         {"device_local", h.deviceLocal},
                         {"size_mb", toMiB(h.size)},
                         {"budget_mb", toMiB(h.budget)},
                         {"process_usage_mb", toMiB(h.usage)},
                         {"vma_blocks_mb", toMiB(h.vmaBlocks)},
                         {"vma_allocations_mb", toMiB(h.vmaAllocations)}});
    return {{"total_registered_mb", toMiB(r.totalAllocated)},
            {"device_local_registered_mb", toMiB(r.deviceLocalAllocated)},
            {"by_owner", owners},
            {"buffers", buffers},
            {"vma_heaps", heaps},
            {"memory_budget_extension", r.budgetExtension}};
}

void printMemorySummary(const MemoryReport& r) {
    std::printf("GPU memory (registered): %.1f MB total", toMiB(r.totalAllocated));
    for (const MemoryOwnerTotal& o : r.owners) std::printf(", %s %.1f", o.owner.c_str(), toMiB(o.allocated));
    std::printf("\n");
    for (const MemoryHeapInfo& h : r.heaps)
        if (h.deviceLocal)
            std::printf("  VMA heap %u (device-local): process usage %.1f MB of budget %.1f MB (VMA blocks %.1f MB)\n",
                        h.index, toMiB(h.usage), toMiB(h.budget), toMiB(h.vmaBlocks));
    std::fflush(stdout);
}

bool sameCompression(const CompressionOptions& a, const CompressionOptions& b) {
    return a.format == b.format && a.layout == b.layout && a.maxShDegree == b.maxShDegree &&
           a.codebookSize == b.codebookSize && a.codebookIterations == b.codebookIterations &&
           a.codebookSamples == b.codebookSamples && a.threads == b.threads && a.fp16Positions == b.fp16Positions &&
           a.shThreshold == b.shThreshold;
}

// Creates renderer R and runs `init` on it. If init throws (e.g. out of GPU memory for a
// large scene), whatever it created so far is destroyed before the exception propagates.
template <typename R, typename Init>
std::unique_ptr<Renderer> makeRenderer(VulkanContext& ctx, Init&& init) {
    auto renderer = std::make_unique<R>();
    try {
        init(*renderer);
    } catch (...) {
        renderer->destroy(ctx);
        throw;
    }
    return renderer;
}

}  // namespace

Application::Application(Options options) : m_options(std::move(options)) {}

Application::~Application() { shutdown(); }

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void Application::initWindow() {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    if (!glfwVulkanSupported()) throw std::runtime_error("GLFW: Vulkan is not supported");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // no OpenGL context: we use Vulkan
    if (m_options.automated()) {
        // Automation never shows the window; the size is fixed so outputs are reproducible.
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    }
    m_window = glfwCreateWindow(int(m_options.width), int(m_options.height), "Gaussian Splat Renderer",
                                nullptr, nullptr);
    if (!m_window) throw std::runtime_error("glfwCreateWindow failed");

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* w, int, int) {
        static_cast<Application*>(glfwGetWindowUserPointer(w))->m_framebufferResized = true;
    });
    // Installed before ImGui so the ImGui backend chains to it.
    glfwSetScrollCallback(m_window, [](GLFWwindow* w, double, double dy) {
        static_cast<Application*>(glfwGetWindowUserPointer(w))->m_scrollDelta += float(dy);
    });
    // Files dropped onto the window (UTF-8 paths): loaded between frames (pollSceneLoad).
    glfwSetDropCallback(m_window, [](GLFWwindow* w, int count, const char** paths) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        for (int i = 0; i < count; ++i) app->m_droppedPaths.push_back(pathFromUtf8(paths[i]));
    });
}

void Application::initVulkan() {
    m_ctx.init(m_window, m_options.validation || kDebugBuild);

    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);
    // Automation uses FIFO-free presentation so the GPU is never throttled by vsync.
    m_swapchain.create(m_ctx, uint32_t(fbWidth), uint32_t(fbHeight), m_options.vsync && !m_options.automated());

    // Per-frame command pools/buffers and acquire semaphores.
    for (FrameData& f : m_frames) {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;  // reset every frame
        poolInfo.queueFamilyIndex = m_ctx.queueFamily;
        VK_CHECK(vkCreateCommandPool(m_ctx.device, &poolInfo, nullptr, &f.commandPool));

        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = f.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(m_ctx.device, &allocInfo, &f.commandBuffer));

        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(m_ctx.device, &semInfo, nullptr, &f.imageAcquired));
        m_ctx.setDebugName(f.imageAcquired, VK_OBJECT_TYPE_SEMAPHORE, "image acquired");
    }

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semInfo.pNext = &timelineInfo;
    VK_CHECK(vkCreateSemaphore(m_ctx.device, &semInfo, nullptr, &m_frameTimeline));
    m_ctx.setDebugName(m_frameTimeline, VK_OBJECT_TYPE_SEMAPHORE, "frame timeline");

    m_profiler.init(m_ctx, kFramesInFlight);

    // Render targets: in automation they have exactly the requested size (independent
    // of what the OS gives the window); interactively they follow the swapchain.
    const VkExtent2D targetExtent = m_options.automated() ? VkExtent2D{m_options.width, m_options.height}
                                                          : m_swapchain.extent;
    createRenderTargets(targetExtent);

    if (m_scene) uploadScene();
    // Phase 6 --pipeline-stats: statistics must be requested when the pipelines are created.
    if (!m_options.pipelineStatsPath.empty()) {
        if (!m_ctx.pipelineExecutableInfo)
            std::printf("warning: VK_KHR_pipeline_executable_properties is not available: no statistics\n");
        setPipelineCaptureStatistics(m_ctx.pipelineExecutableInfo);
    }
    // Without a scene only "test" can run; the requested technique starts with the first scene.
    createRenderer(m_scene ? m_options.renderer : "test");

    if (!m_options.screenshotPath.empty()) {
        MemoryScope memoryScope(m_ctx, "app");
        m_readback = createBuffer(m_ctx, VkDeviceSize(targetExtent.width) * targetExtent.height * 4,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Readback, "screenshot readback");
    }
}

CompressionOptions Application::compressionOptions() const {
    CompressionOptions co;
    co.format = *parseSceneFormat(m_options.sceneFormat);
    co.layout = m_options.sceneLayout.empty()
                    ? (co.format == SceneFormat::F32 ? SceneLayout::Soa : SceneLayout::Split)
                    : *parseSceneLayout(m_options.sceneLayout);
    co.maxShDegree = m_options.sceneShDegree ? int(*m_options.sceneShDegree) : 3;
    co.codebookSize = m_options.codebookSize;
    co.codebookIterations = m_options.codebookIterations;
    co.fp16Positions = m_options.debugFp16Positions;
    co.shThreshold = m_options.shThreshold;
    return co;
}

void Application::uploadScene(EncodedScene* preEncoded) {
    m_gpuScene.upload(m_ctx, *m_scene, compressionOptions(), preEncoded);
    const GpuScene& g = m_gpuScene;
    std::printf("Scene uploaded: format %s, layout %s, SH degree %d stored, %.1f MB device-local (%.1f B/gaussian), "
                "encode %.1f ms, upload %.1f ms\n",
                sceneFormatName(g.format), sceneLayoutName(g.layout), g.shDegree, toMiB(g.totalBytes),
                g.bytesPerGaussian(), g.encodeMs, g.uploadMs);
    if (g.color == ColorEncoding::Codebook)
        std::printf("  SH codebook: %u entries x %u dims, %u samples, %u iterations (%u re-seeded), train %.1f ms, "
                    "assign %.1f ms, total %.1f ms, MSE %.3g (sample %.3g)%s\n",
                    g.codebook.entries, g.codebook.dimensions, g.codebook.samples, g.codebook.iterations,
                    g.codebook.emptyReseeded, g.codebook.trainMs, g.codebook.assignMs, g.codebook.totalMs,
                    g.codebook.fullMse, g.codebook.sampleMse, g.codebook.simd ? ", AVX2" : ", scalar");
    if (m_options.debugFp16Positions) std::printf("  DEBUG: positions rounded through fp16\n");
    if (g.variableSh != VariableSh::Off) {
        const auto& h = g.degreeHistogram;
        const double n = std::max(1.0, double(g.count));
        std::printf("  per-gaussian SH degree (threshold %.3g/255, %s): degree 0 %.1f%%, 1 %.1f%%, 2 %.1f%%, 3 %.1f%%, "
                    "mean SH coefficients %.2f\n",
                    m_options.shThreshold, g.variableSh == VariableSh::Compact ? "compacted" : "degree only",
                    100.0 * h[0] / n, 100.0 * h[1] / n, 100.0 * h[2] / n, 100.0 * h[3] / n,
                    (h[0] * 1.0 + h[1] * 4.0 + h[2] * 9.0 + h[3] * 16.0) / n);
    }
}

void Application::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // do not write imgui.ini into the working directory
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(m_window, true);

    m_swapchainFormatForImGui = m_swapchain.format;
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = m_ctx.instance;
    info.PhysicalDevice = m_ctx.physicalDevice;
    info.Device = m_ctx.device;
    info.QueueFamily = m_ctx.queueFamily;
    info.Queue = m_ctx.queue;
    info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + 16;
    info.MinImageCount = m_swapchain.minImageCount;
    info.ImageCount = uint32_t(m_swapchain.images.size());
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormatForImGui;
    info.CheckVkResultFn = checkImGuiVk;
    if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("ImGui_ImplVulkan_Init failed");
    m_imguiInitialized = true;
}

void Application::createRenderTargets(VkExtent2D extent) {
    MemoryScope memoryScope(m_ctx, "app");
    // STORAGE: written by compute shaders. COLOR_ATTACHMENT: drawn into by raster
    // passes. TRANSFER_SRC: blitted to the swapchain / copied for screenshots.
    // TRANSFER_DST: renderers with their own intermediate target (e.g. the splat
    // renderer's float target) blit their result into it. SAMPLED: lets later phases
    // read it in a shader (e.g. post-processing).
    m_color = createImage2D(m_ctx, extent.width, extent.height, kColorFormat,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT,
                            "color target");
    m_depth = createImage2D(m_ctx, extent.width, extent.height, kDepthFormat,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth target");
    // Fresh images have no previous use to wait for.
    m_colorState = {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
    m_depthState = {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};
}

void Application::createRenderer(const std::string& name) {
    if (m_renderer) {
        VK_CHECK(vkDeviceWaitIdle(m_ctx.device));  // the old renderer's resources may be in use
        m_renderer->destroy(m_ctx);
        m_renderer.reset();
    }
    MemoryScope memoryScope(m_ctx, name);  // the renderer's buffers are reported under its name
    if (name == "splat") {
        if (!m_gpuScene.valid()) throw std::runtime_error("renderer 'splat' needs a scene (--scene)");
        SplatSettings settings;
        settings.format = *parseSplatTargetFormat(m_options.splatFormat);
        settings.bound = *parseCoverageBound(m_options.splatBound);
        settings.maxShDegree = m_options.shDegree ? int(*m_options.shDegree) : 3;
        settings.bounds = *parseTileBounds(m_options.bounds);
        m_renderer = makeRenderer<SplatRenderer>(
            m_ctx, [&](SplatRenderer& r) { r.init(m_ctx, kColorFormat, m_gpuScene, *m_scene, kFramesInFlight, settings); });
    } else if (name == "splat-gpusort") {
        if (!m_gpuScene.valid()) throw std::runtime_error("renderer 'splat-gpusort' needs a scene (--scene)");
        SplatSettings settings;
        settings.format = *parseSplatTargetFormat(m_options.splatFormat);
        settings.bound = *parseCoverageBound(m_options.splatBound);
        settings.maxShDegree = m_options.shDegree ? int(*m_options.shDegree) : 3;
        settings.bounds = *parseTileBounds(m_options.bounds);
        GpuSortSettings sortSettings;
        sortSettings.sortEvery = m_options.sortEvery;
        sortSettings.sortThresholdDeg = m_options.sortThresholdDeg;
        sortSettings.radixVariant = m_options.radixVariant == "reorder" ? GpuRadixSort::Variant::Reorder
                                                                         : GpuRadixSort::Variant::Classic;
        m_renderer = makeRenderer<SplatGpuSortRenderer>(m_ctx, [&](SplatGpuSortRenderer& r) {
            r.init(m_ctx, kColorFormat, m_gpuScene, *m_scene, kFramesInFlight, settings, sortSettings);
        });
    } else if (name == "splat-tiles") {
        if (!m_gpuScene.valid()) throw std::runtime_error("renderer 'splat-tiles' needs a scene (--scene)");
        SplatSettings settings;
        settings.format = *parseSplatTargetFormat(m_options.splatFormat);
        settings.bound = *parseCoverageBound(m_options.splatBound);
        settings.maxShDegree = m_options.shDegree ? int(*m_options.shDegree) : 3;
        settings.bounds = *parseTileBounds(m_options.bounds);
        TileSettings tileSettings;
        tileSettings.sortMode = m_options.tileSort == "global" ? TileSortMode::Global : TileSortMode::Binning;
        tileSettings.duplicateMode = m_options.tileDuplicate == "per-splat" ? TileDuplicateMode::PerSplat
                                                                             : TileDuplicateMode::PerEntry;
        tileSettings.radixVariant = m_options.radixVariant == "classic" ? GpuRadixSort::Variant::Classic
                                                                         : GpuRadixSort::Variant::Reorder;
        tileSettings.budgetMB = m_options.tileBudgetMB;
        tileSettings.stats = !m_options.tileStatsPrefix.empty();
        tileSettings.renderSplat = m_options.renderSplat == "p32"   ? RenderSplatMode::Packed32
                                   : m_options.renderSplat == "p24" ? RenderSplatMode::Packed24
                                                                    : RenderSplatMode::Full;
        tileSettings.adaptiveList = m_options.tileList == "adaptive";
        tileSettings.tileSize = m_options.tileSize;
        tileSettings.sortWindow = m_options.sortWindow;
        tileSettings.renderBatch = m_options.renderBatch;
        tileSettings.voteEvery = m_options.renderVoteEvery;
        tileSettings.sharedAoS = m_options.renderShared == "aos";
        tileSettings.fp16Math = m_options.renderFp16;
        m_renderer = makeRenderer<SplatTileRenderer>(
            m_ctx, [&](SplatTileRenderer& r) { r.init(m_ctx, m_gpuScene, kFramesInFlight, settings, tileSettings); });
    } else if (name == "points") {
        if (!m_gpuScene.valid()) throw std::runtime_error("renderer 'points' needs a scene (--scene)");
        m_renderer = makeRenderer<PointCloudRenderer>(
            m_ctx, [&](PointCloudRenderer& r) { r.init(m_ctx, kColorFormat, kDepthFormat, m_gpuScene); });
    } else if (name == "test") {
        m_renderer = makeRenderer<TestRenderer>(m_ctx, [&](TestRenderer& r) { r.init(m_ctx, kColorFormat, kDepthFormat); });
    } else {
        throw std::runtime_error("unknown renderer '" + name + "'");
    }
    m_renderer->onRenderTargetsChanged(m_ctx, m_color, m_depth);
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

void Application::loadScene() {
    if (m_options.scenePath.empty()) return;
    // The same function the UI's scene loading runs on its worker thread (scene_loading.h).
    LoadedScene loaded = loadSceneFile({m_options.scenePath, m_options.camerasPath, m_options.upAxis});
    m_scene = std::move(loaded.scene);
    m_scenePath = loaded.path;
    m_setup = std::move(loaded.setup);
}

void Application::setUpAxis(UpAxis up) {
    if (up == UpAxis::Cameras && m_setup.trainingCameras.empty()) return;  // UI: not available
    m_setup.upAxis = up;
    m_setup.model = sceneModelMatrix(up, m_setup.trainingCameras);
}

Camera Application::initialSceneCamera() const {
    if (!m_scene) return m_options.cameraPath.empty() ? Camera::defaultCamera() : Camera::load(m_options.cameraPath);
    return gsr::initialSceneCamera(*m_scene, m_setup, m_options.cameraPath, m_options.cameraIndex);
}

void Application::destroyRenderTargets() {
    destroyImage(m_ctx, m_color);
    destroyImage(m_ctx, m_depth);
}

void Application::recreateSwapchain() {
    // A minimized window has a 0x0 framebuffer: wait until it is restored.
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while ((w == 0 || h == 0) && !glfwWindowShouldClose(m_window)) {
        glfwWaitEvents();
        glfwGetFramebufferSize(m_window, &w, &h);
    }
    // Simplest correct approach: wait until the GPU is idle, then rebuild everything
    // that depends on the window size.
    VK_CHECK(vkDeviceWaitIdle(m_ctx.device));
    m_swapchain.create(m_ctx, uint32_t(w), uint32_t(h), m_options.vsync && !m_options.automated());
    if (m_imguiInitialized) ImGui_ImplVulkan_SetMinImageCount(m_swapchain.minImageCount);

    if (!m_options.automated()) {
        destroyRenderTargets();
        createRenderTargets(m_swapchain.extent);
        m_renderer->onRenderTargetsChanged(m_ctx, m_color, m_depth);
    }
    m_framebufferResized = false;
    m_swapchainDirty = false;
}

void Application::shutdown() {
    // A scene still loading in the background: let its thread finish (result discarded).
    m_loadJob.cancel();
    m_loadJob.wait();
    if (m_ctx.device) vkDeviceWaitIdle(m_ctx.device);
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_imguiInitialized = false;
    }
    if (m_ctx.device) {
        if (m_renderer) m_renderer->destroy(m_ctx);
        m_renderer.reset();
        m_gpuScene.destroy(m_ctx);
        destroyBuffer(m_ctx, m_readback);
        destroyRenderTargets();
        m_profiler.destroy();
        for (FrameData& f : m_frames) {
            if (f.commandPool) vkDestroyCommandPool(m_ctx.device, f.commandPool, nullptr);
            if (f.imageAcquired) vkDestroySemaphore(m_ctx.device, f.imageAcquired, nullptr);
            f = FrameData{};
        }
        if (m_frameTimeline) vkDestroySemaphore(m_ctx.device, m_frameTimeline, nullptr);
        m_frameTimeline = VK_NULL_HANDLE;
        m_swapchain.destroy(m_ctx);
    }
    m_ctx.destroy();
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
    }
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

bool Application::drawFrame(bool captureScreenshot) {
    if (m_framebufferResized || m_swapchainDirty) recreateSwapchain();
    pollSceneLoad();  // a scene loaded in the background is swapped in here, between frames
    if (m_pendingSceneUpload) {
        // New scene format: the renderer's descriptor sets and pipelines depend on the scene
        // buffers and their format, so it is recreated too.
        VK_CHECK(vkDeviceWaitIdle(m_ctx.device));
        const std::string name = m_renderer->name();
        m_renderer->destroy(m_ctx);
        m_renderer.reset();
        m_gpuScene.destroy(m_ctx);
        uploadScene();
        createRenderer(m_pendingRenderer.empty() ? name : m_pendingRenderer);
        m_pendingRenderer.clear();
        m_pendingSceneUpload = false;
    }
    if (!m_pendingRenderer.empty()) {
        createRenderer(m_pendingRenderer);  // waits for the GPU; the new renderer starts clean
        m_pendingRenderer.clear();
    }

    const uint32_t slot = uint32_t(m_frameIndex % kFramesInFlight);
    FrameData& frame = m_frames[slot];

    // 1) CPU waits until the GPU finished the frame that last used this slot, so its
    //    command buffer, acquire semaphore and timestamp queries can be reused.
    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_frameTimeline;
    waitInfo.pValues = &frame.timelineValue;
    VK_CHECK(vkWaitSemaphores(m_ctx.device, &waitInfo, UINT64_MAX));
    m_profiler.beginFrame(slot, m_frameIndex);

    // 2) Get a swapchain image. The semaphore is signaled when the image is really free.
    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(m_ctx.device, m_swapchain.handle, UINT64_MAX,
                                             frame.imageAcquired, VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return false;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) VK_CHECK(acquire);

    // 3) Record this frame's commands.
    VK_CHECK(vkResetCommandPool(m_ctx.device, frame.commandPool, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &begin));
    recordFrame(frame.commandBuffer, imageIndex, captureScreenshot);
    VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));

    // 4) Submit. Wait for the acquire semaphore only at the stage that first touches
    //    the swapchain image (the blit); everything before it can start immediately.
    frame.timelineValue = ++m_timelineValue;

    VkSemaphoreSubmitInfo waitSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitSemaphore.semaphore = frame.imageAcquired;
    waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

    VkSemaphoreSubmitInfo signalSemaphores[2]{};
    signalSemaphores[0] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalSemaphores[0].semaphore = m_swapchain.renderFinished[imageIndex];  // for present
    signalSemaphores[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signalSemaphores[1] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalSemaphores[1].semaphore = m_frameTimeline;  // for the CPU (step 1, readbacks)
    signalSemaphores[1].value = frame.timelineValue;
    signalSemaphores[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = frame.commandBuffer;

    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitSemaphore;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.signalSemaphoreInfoCount = 2;
    submit.pSignalSemaphoreInfos = signalSemaphores;
    VK_CHECK(vkQueueSubmit2(m_ctx.queue, 1, &submit, VK_NULL_HANDLE));

    // 5) Present once rendering is done (binary semaphore; present cannot wait on timelines).
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &m_swapchain.renderFinished[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &m_swapchain.handle;
    present.pImageIndices = &imageIndex;
    VkResult presentResult = vkQueuePresentKHR(m_ctx.queue, &present);
    m_frameIndex++;
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        m_framebufferResized = true;  // recreate at the start of the next frame
    else
        VK_CHECK(presentResult);
    return true;
}

void Application::recordFrame(VkCommandBuffer cmd, uint32_t imageIndex, bool captureScreenshot) {
    auto frameTimer = m_profiler.scope(cmd, "frame (total)");
    const VkExtent2D targetExtent = m_color.extent;

    // --- Scene ---------------------------------------------------------------
    const float aspect = float(targetExtent.width) / float(targetExtent.height);
    const FrameInput input{cmd,
                           m_profiler,
                           m_cpuProfiler,
                           m_color,
                           m_depth,
                           m_colorState,
                           m_depthState,
                           m_camera.view(),
                           m_camera.projection(aspect),
                           m_setup.model,
                           m_frameIndex,
                           uint32_t(m_frameIndex % kFramesInFlight)};
    const RenderOutput out = m_renderer->render(input);
    m_depthState = out.depth;

    // The color target is read by transfer commands from here on.
    imageBarrier(cmd, m_color.image, VK_IMAGE_ASPECT_COLOR_BIT, out.color.layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.color.stage, out.color.access,
                 VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    // Next frame's first write must wait for these reads (execution dependency only).
    m_colorState = {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_NONE};

    // --- Screenshot readback ---------------------------------------------------
    if (captureScreenshot) {
        auto timer = m_profiler.scope(cmd, "screenshot copy");
        VkBufferImageCopy region{};
        region.bufferRowLength = 0;  // tightly packed
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {targetExtent.width, targetExtent.height, 1};
        vkCmdCopyImageToBuffer(cmd, m_color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_readback.buffer, 1, &region);
        // Make the copy visible to the host once the frame's timeline value is reached.
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    }

    // --- Copy to the swapchain image -------------------------------------------
    VkImage swapImage = m_swapchain.images[imageIndex];
    {
        auto timer = m_profiler.scope(cmd, "blit to swapchain");
        // srcStage = ALL_TRANSFER matches the acquire semaphore's wait stage, which
        // chains this layout transition after the presentation engine released the image.
        imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                     VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        // A blit (not a copy) because it converts RGBA8 -> the swapchain's BGRA8 and
        // scales if the window size differs from the render size.
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {int32_t(targetExtent.width), int32_t(targetExtent.height), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {int32_t(m_swapchain.extent.width), int32_t(m_swapchain.extent.height), 1};
        const bool sameSize = targetExtent.width == m_swapchain.extent.width &&
                              targetExtent.height == m_swapchain.extent.height;
        vkCmdBlitImage(cmd, m_color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       sameSize ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    }

    // --- UI on top --------------------------------------------------------------
    {
        auto timer = m_profiler.scope(cmd, "imgui");
        imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = m_swapchain.views[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // keep the blitted scene
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, m_swapchain.extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &colorAttachment;
        vkCmdBeginRendering(cmd, &rendering);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }

    // Presentation requires PRESENT_SRC layout. dst = NONE: the renderFinished
    // semaphore signal (ALL_COMMANDS) orders presentation after this barrier.
    imageBarrier(cmd, swapImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
}

// ---------------------------------------------------------------------------
// Input and UI
// ---------------------------------------------------------------------------

void Application::processInput(float dt) {
    const ImGuiIO& io = ImGui::GetIO();
    double mx = 0.0, my = 0.0;
    glfwGetCursorPos(m_window, &mx, &my);
    const float dx = float(mx - m_lastMouseX);
    const float dy = float(my - m_lastMouseY);
    m_lastMouseX = mx;
    m_lastMouseY = my;
    const float scroll = m_scrollDelta;
    m_scrollDelta = 0.0f;

    // Tab toggles orbit/fly (edge-triggered), R resets the camera.
    const bool tabDown = glfwGetKey(m_window, GLFW_KEY_TAB) == GLFW_PRESS;
    if (!io.WantCaptureKeyboard) {
        if (tabDown && !m_tabWasDown)
            m_camera.mode = m_camera.mode == CameraMode::Orbit ? CameraMode::Fly : CameraMode::Orbit;
        if (glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS) m_camera = m_initialCamera;
    }
    // Ctrl+O = the Scene panel's "Open scene..." button (the dialog opens between frames).
    if (!io.WantTextInput && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O)) m_openDialogRequested = true;
    m_tabWasDown = tabDown;

    const bool lmb = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool rmb = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool mmb = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const float rotateSpeed = 0.25f;  // degrees per pixel

    if (m_camera.mode == CameraMode::Orbit) {
        if (!io.WantCaptureMouse) {
            if (lmb) m_camera.orbit(dx * rotateSpeed, -dy * rotateSpeed);
            if (rmb || mmb) {
                const float panScale = m_camera.distance * 0.0015f;
                m_camera.pan(-dx * panScale, dy * panScale);
            }
            if (scroll != 0.0f) m_camera.zoom(scroll);
        }
    } else {
        if (!io.WantCaptureMouse) {
            if (rmb || lmb) m_camera.look(dx * rotateSpeed, -dy * rotateSpeed);
            if (scroll != 0.0f) m_flySpeed = std::clamp(m_flySpeed * std::pow(1.2f, scroll), 0.01f, 1000.0f);
        }
        if (!io.WantCaptureKeyboard) {
            glm::vec3 move(0.0f);
            if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) move.z += 1.0f;
            if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) move.z -= 1.0f;
            if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) move.x += 1.0f;
            if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) move.x -= 1.0f;
            if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) move.y += 1.0f;
            if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) move.y -= 1.0f;
            float speed = m_flySpeed * dt;
            if (glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;
            if (move != glm::vec3(0.0f)) m_camera.move(glm::normalize(move) * speed);
        }
    }
}

void Application::buildSceneUi() {
    ImGui::SeparatorText("Scene");
    buildSceneLoadUi();
    if (!m_scene) {
        ImGui::TextUnformatted("No scene: open a .ply (button, Ctrl+O, drag and drop or --scene)");
        return;
    }
    const GaussianScene& sc = *m_scene;
    const SceneStats& st = sc.stats;
    ImGui::TextUnformatted(pathToUtf8(m_scenePath.filename()).c_str());
    ImGui::Text("%u gaussians, SH degree %d, loaded in %.0f ms", sc.count, sc.shDegree, sc.timings.totalMs);
    ImGui::Text("GPU %.1f MB = %.1f B/gaussian (format %s, layout %s, SH %d stored)", toMiB(m_gpuScene.totalBytes),
                m_gpuScene.bytesPerGaussian(), sceneFormatName(m_gpuScene.format), sceneLayoutName(m_gpuScene.layout),
                m_gpuScene.shDegree);
    ImGui::Text("encode %.0f ms, upload %.0f ms", m_gpuScene.encodeMs, m_gpuScene.uploadMs);
    if (m_gpuScene.color == ColorEncoding::Codebook)
        ImGui::Text("SH codebook %u entries, k-means %.0f ms, MSE %.2e", m_gpuScene.codebook.entries,
                    m_gpuScene.codebook.totalMs, m_gpuScene.codebook.fullMse);
    if (ImGui::TreeNode("Stats (data space)")) {
        ImGui::Text("bbox   %.2f %.2f %.2f .. %.2f %.2f %.2f", st.bboxMin.x, st.bboxMin.y, st.bboxMin.z, st.bboxMax.x,
                    st.bboxMax.y, st.bboxMax.z);
        ImGui::Text("p1-p99 %.2f %.2f %.2f .. %.2f %.2f %.2f", st.robustMin.x, st.robustMin.y, st.robustMin.z,
                    st.robustMax.x, st.robustMax.y, st.robustMax.z);
        ImGui::Text("alpha  %.4f .. %.4f, mean %.3f", st.alphaMin, st.alphaMax, st.alphaMean);
        ImGui::Text("alpha < 1/255: %u", st.alphaBelowVisible);
        ImGui::Text("sigma  %.2e .. %.2e (median max %.2e)", st.sigmaMin, st.sigmaMax, st.sigmaMaxMedian);
        ImGui::Text("|q| raw %.3f .. %.3f, zero: %u", st.quatNormMin, st.quatNormMax, st.zeroQuaternions);
        ImGui::Text("NaN/Inf values: %llu", static_cast<unsigned long long>(st.nonFiniteValues));
        ImGui::TreePop();
    }

    const char* ups[] = {"+y", "-y", "+z", "-z", "cameras"};
    int up = int(m_setup.upAxis);
    if (ImGui::Combo("Up axis", &up, ups, m_setup.trainingCameras.empty() ? 4 : 5)) setUpAxis(UpAxis(up));
    if (ImGui::Button("Frame scene")) m_camera = framedSceneCamera(*m_scene, m_setup.model);
    if (!m_setup.trainingCameras.empty()) {
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputInt("##camidx", &m_uiCameraIndex);
        m_uiCameraIndex = std::clamp(m_uiCameraIndex, 0, int(m_setup.trainingCameras.size()) - 1);
        ImGui::SameLine();
        if (ImGui::Button("Go to training camera")) m_camera = trainingCameraView(m_setup, uint32_t(m_uiCameraIndex));
        ImGui::Text("%zu training cameras", m_setup.trainingCameras.size());
    }
}

void Application::buildCompressionUi() {
    // Phase 5 options that need a re-upload (scene) or a new renderer (splat-tiles buffers).
    if (!m_scene || !ImGui::TreeNode("Compression (Phase 5) + quality vs cost (Phase 6)")) return;
    const char* formats[] = {"f32", "f16", "q8", "cb"};
    int format = int(*parseSceneFormat(m_options.sceneFormat));
    const char* layouts[] = {"default", "soa", "split", "aos"};
    int layout = m_options.sceneLayout.empty() ? 0 : int(*parseSceneLayout(m_options.sceneLayout)) + 1;
    int stored = m_options.sceneShDegree ? int(*m_options.sceneShDegree) : 3;
    int codebook = int(m_options.codebookSize);
    bool reupload = false;
    reupload |= ImGui::Combo("Scene format", &format, formats, 4);
    reupload |= ImGui::Combo("Scene layout", &layout, layouts, 4);
    reupload |= ImGui::SliderInt("Stored SH degree", &stored, 0, 3);
    if (format == int(SceneFormat::Codebook)) {
        ImGui::InputInt("Codebook size", &codebook, 256, 1024);
        ImGui::SameLine();
        reupload |= ImGui::Button("Rebuild");
    }
    if (reupload) {
        if (format != int(SceneFormat::F32) && layout == 1) layout = 2;  // soa is f32 only
        m_options.sceneFormat = formats[format];
        m_options.sceneLayout = layout == 0 ? "" : layouts[layout];
        m_options.sceneShDegree = uint32_t(stored);
        m_options.codebookSize = uint32_t(std::clamp(codebook, 1, 65536));
        m_pendingSceneUpload = true;
    }
    // Phase 6: binning bounds, tile size, StopThePop window (renderer recreation), SH threshold (re-upload).
    {
        const char* boundsNames[] = {"rect (INRIA 3-sigma)", "tight (exact)", "cutoff-only (not INRIA)"};
        int b = int(*parseTileBounds(m_options.bounds));
        int tile = m_options.tileSize == 8 ? 1 : 0;
        int window = int(m_options.sortWindow);
        float sht = m_options.shThreshold;
        bool useSh = sht >= 0.0f;
        bool re = ImGui::Combo("Bounds (Phase 6)", &b, boundsNames, 3);
        const char* tiles[] = {"16x16", "8x8"};
        re |= ImGui::Combo("Tile size (splat-tiles)", &tile, tiles, 2);
        re |= ImGui::SliderInt("StopThePop window (splat-tiles, 0 = off)", &window, 0, 32);
        if (re) {
            const char* names[] = {"rect", "tight", "cutoff-only"};
            m_options.bounds = names[b];
            if (b == 2) m_options.splatBound = "tile";
            m_options.tileSize = tile == 1 ? 8u : 16u;
            m_options.sortWindow = uint32_t(window);
            m_pendingRenderer = m_renderer->name();
        }
        bool reSh = ImGui::Checkbox("Per-gaussian SH degree", &useSh);
        if (useSh) reSh |= ImGui::SliderFloat("SH threshold (/255)", &sht, 0.0f, 8.0f, "%.2f");
        if (reSh) {
            m_options.shThreshold = useSh ? std::max(sht, 0.0f) : -1.0f;
            m_pendingSceneUpload = true;
        }
    }
    const char* splats[] = {"full", "p32", "p24"};
    int splat = m_options.renderSplat == "p32" ? 1 : m_options.renderSplat == "p24" ? 2 : 0;
    bool adaptive = m_options.tileList == "adaptive";
    bool recreate = ImGui::Combo("Render splat (splat-tiles)", &splat, splats, 3);
    recreate |= ImGui::Checkbox("Adaptive list capacity (splat-tiles)", &adaptive);
    if (recreate) {
        m_options.renderSplat = splats[splat];
        m_options.tileList = adaptive ? "adaptive" : "fixed";
        if (std::string(m_renderer->name()) == "splat-tiles") m_pendingRenderer = "splat-tiles";
    }
    ImGui::TreePop();
}

void Application::buildUi() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Renderer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%s", m_ctx.deviceName.c_str());
    ImGui::Text("%.1f FPS (%.2f ms)", io.Framerate, 1000.0f / std::max(io.Framerate, 1e-3f));
    ImGui::Text("Render %ux%u  |  Window %ux%u  |  %s", m_color.extent.width, m_color.extent.height,
                m_swapchain.extent.width, m_swapchain.extent.height, presentModeName(m_swapchain.presentMode));
    ImGui::Text("Validation: %s", m_ctx.validationEnabled ? "on" : "off");
    if (!m_options.automated()) {
        bool vsync = m_options.vsync;
        if (ImGui::Checkbox("VSync", &vsync)) {
            m_options.vsync = vsync;
            m_swapchainDirty = true;
        }
    }

    ImGui::SeparatorText("Renderer");
    {
        const char* names[] = {"splat", "splat-gpusort", "splat-tiles", "points", "test"};
        int current = 0;
        for (int i = 0; i < 5; ++i)
            if (names[i] == std::string(m_renderer->name())) current = i;
        ImGui::BeginDisabled(!m_gpuScene.valid());  // all but "test" need a scene
        if (ImGui::Combo("Technique", &current, names, 5) && names[current] != std::string(m_renderer->name()))
            m_pendingRenderer = names[current];
        ImGui::EndDisabled();
        m_renderer->drawUi();
    }
    buildSceneUi();
    buildCompressionUi();

    // GPU memory (Phase 5): totals per owner, every buffer on demand, VMA heap budgets.
    ImGui::SeparatorText("GPU memory");
    {
        const MemoryReport report = buildMemoryReport(m_ctx);
        ImGui::Text("Registered %.1f MB in %zu buffers/images", toMiB(report.totalAllocated), report.records.size());
        for (const MemoryOwnerTotal& o : report.owners)
            ImGui::BulletText("%s: %.1f MB (%u)", o.owner.c_str(), toMiB(o.allocated), o.count);
        for (const MemoryHeapInfo& h : report.heaps)
            if (h.deviceLocal)
                ImGui::Text("Heap %u (device-local): usage %.0f / budget %.0f MB%s", h.index, toMiB(h.usage),
                            toMiB(h.budget), report.budgetExtension ? "" : " (no VK_EXT_memory_budget)");
        if (ImGui::TreeNode("Buffers")) {
            if (ImGui::BeginTable("mem", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Owner");
                ImGui::TableSetupColumn("Name");
                ImGui::TableSetupColumn("MB");
                ImGui::TableSetupColumn("Usage");
                ImGui::TableHeadersRow();
                for (const MemoryRecord& m : report.records) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(m.owner.c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(m.name.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f", toMiB(m.allocated));
                    ImGui::TableNextColumn();
                    ImGui::Text("%s, %s", (m.image ? imageUsageString(m.usage) : bufferUsageString(m.usage)).c_str(),
                                memoryKindName(m.memoryKind));
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
    }

    ImGui::SeparatorText("GPU timings (ms, avg of last 120 frames)");
    if (ImGui::BeginTable("gpu", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Avg");
        ImGui::TableSetupColumn("Last");
        ImGui::TableHeadersRow();
        for (const GpuProfiler::Stats& s : m_profiler.stats()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(s.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", s.avgMs);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", s.lastMs);
        }
        ImGui::EndTable();
    }

    if (!m_cpuProfiler.stats().empty()) {
        ImGui::SeparatorText("CPU timings (ms, avg of last 120 frames)");
        if (ImGui::BeginTable("cpu", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn("Avg");
            ImGui::TableSetupColumn("Last");
            ImGui::TableHeadersRow();
            for (const TimingStats& s : m_cpuProfiler.stats()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(s.name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", s.avgMs);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", s.lastMs);
            }
            ImGui::EndTable();
        }
    }

    ImGui::SeparatorText("Camera");
    int mode = m_camera.mode == CameraMode::Orbit ? 0 : 1;
    ImGui::RadioButton("Orbit", &mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Fly", &mode, 1);
    m_camera.mode = mode == 0 ? CameraMode::Orbit : CameraMode::Fly;
    const glm::vec3 t = m_camera.target();
    ImGui::Text("Position %.3f %.3f %.3f", m_camera.position.x, m_camera.position.y, m_camera.position.z);
    ImGui::Text("Target   %.3f %.3f %.3f", t.x, t.y, t.z);
    ImGui::Text("Yaw %.1f  Pitch %.1f  Dist %.3f", m_camera.yawDeg, m_camera.pitchDeg, m_camera.distance);
    ImGui::SliderFloat("FOV (deg)", &m_camera.fovYDeg, 10.0f, 120.0f);
    if (m_camera.mode == CameraMode::Fly) ImGui::SliderFloat("Fly speed", &m_flySpeed, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::Button("Save camera.json")) {
        try {
            m_camera.save("camera.json");
            m_statusMessage = "Saved " + std::filesystem::absolute("camera.json").string();
        } catch (const std::exception& e) {
            m_statusMessage = e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset (R)")) m_camera = m_initialCamera;
    if (!m_statusMessage.empty()) ImGui::TextWrapped("%s", m_statusMessage.c_str());

    ImGui::SeparatorText("Controls");
    ImGui::TextUnformatted(
        "Tab: toggle orbit / fly\n"
        "Orbit: LMB rotate, RMB/MMB pan, wheel zoom\n"
        "Fly: RMB/LMB look, WASD move, Q/E down/up,\n"
        "     Shift faster, wheel changes speed\n"
        "Ctrl+O or drop a .ply on the window: open a scene");
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Run loops
// ---------------------------------------------------------------------------

int Application::run() {
    loadScene();
    initWindow();
    initVulkan();
    initImGui();

    m_camera = initialSceneCamera();
    m_initialCamera = m_camera;
    if (m_options.cameraIndex) m_uiCameraIndex = int(*m_options.cameraIndex);
    if (!m_options.automated()) {  // automated runs never read or write the recent-scenes file
        loadRecentScenes();
        if (m_scene) addRecentScene(m_scenePath);
    }

    std::printf("Device: %s (%s)\n", m_ctx.deviceName.c_str(), m_ctx.driverVersion.c_str());
    std::printf("Validation: %s\n", m_ctx.validationEnabled ? "enabled" : "disabled");
    std::fflush(stdout);

    if (!m_options.pipelineStatsPath.empty()) {
        writePipelineStats();
        shutdown();
        const auto& vc = VulkanContext::validationCounters();
        std::printf("Validation messages: %u errors, %u warnings\n", vc.errors.load(), vc.warnings.load());
        return (vc.errors > 0 || vc.warnings > 0) ? 3 : 0;
    }
    const int code = m_options.automated() ? runAutomated() : runInteractive();
    shutdown();

    const auto& counters = VulkanContext::validationCounters();
    std::printf("Validation messages: %u errors, %u warnings\n", counters.errors.load(), counters.warnings.load());
    // Automated runs fail loudly if validation complained, so scripts cannot miss it.
    if (code == 0 && m_options.automated() && (counters.errors > 0 || counters.warnings > 0)) return 3;
    return code;
}

int Application::runInteractive() {
    glfwGetCursorPos(m_window, &m_lastMouseX, &m_lastMouseY);
    auto last = std::chrono::steady_clock::now();
    uint64_t frames = 0;
    bool loadRequested = m_options.loadAfterPath.empty();
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        if (!loadRequested && frames >= m_options.loadAfterFrames) {  // --load-after (automation hook)
            requestSceneLoad(m_options.loadAfterPath);
            loadRequested = true;
        }
        if (m_openDialogRequested) {
            // Outside the ImGui frame: the native dialog is modal and blocks this thread.
            m_openDialogRequested = false;
            openSceneDialog();
            last = std::chrono::steady_clock::now();  // no giant dt for the fly camera
        }
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        processInput(dt);
        buildUi();
        ImGui::Render();
        if (drawFrame(false)) ++frames;
    }
    return 0;
}

int Application::runAutomated() {
    const uint32_t frameCount = m_options.frameCount();
    const bool benchmark = !m_options.benchmarkPath.empty();
    // Benchmarks discard the first frames (pipeline warm-up, clocks ramping up).
    const uint32_t warmup = benchmark ? std::min(frameCount / 10 + 5, frameCount - 1) : 0;
    std::vector<double> cpuFrameMs;
    cpuFrameMs.reserve(frameCount);

    auto last = std::chrono::steady_clock::now();
    bool loadRequested = m_options.loadAfterPath.empty();
    uint32_t framesWhileLoading = 0;
    for (uint32_t i = 0; i < frameCount;) {
        glfwPollEvents();
        // --load-after: exactly what the UI's Open scene... / drag and drop do.
        if (!loadRequested && i == m_options.loadAfterFrames) {
            requestSceneLoad(m_options.loadAfterPath);
            loadRequested = true;
        }
        // Benchmark camera path: one full orbit around the target over the run.
        if (benchmark && i > 0) m_camera.orbit(360.0f / float(frameCount), 0.0f);
        // Screenshots of a moving camera (e.g. to measure reused sort orders).
        if (!benchmark && i > 0 && m_options.orbitStepDeg != 0.0f) m_camera.orbit(m_options.orbitStepDeg, 0.0f);
        const bool sequence = !m_options.sequencePrefix.empty();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildUi();
        ImGui::Render();

        // From the first measured frame on, GPU timings go into the benchmark stats.
        if (benchmark && i == warmup) {
            m_profiler.resetAccumulators(m_frameIndex);
            m_cpuProfiler.resetAccumulators();  // CPU stages run inside drawFrame(i)
        }
        const bool lastFrame = i + 1 == frameCount;
        if (!drawFrame(!m_options.screenshotPath.empty() && (lastFrame || sequence))) continue;  // retry
        if (sequence) {
            // Phase 6 --sequence: every frame to its own PNG (waits for the GPU: not for timing).
            VK_CHECK(vkDeviceWaitIdle(m_ctx.device));
            char name[32];
            std::snprintf(name, sizeof(name), "_%04u.png", i);
            VK_CHECK(vmaInvalidateAllocation(m_ctx.allocator, m_readback.allocation, 0, VK_WHOLE_SIZE));
            writePngRgb(m_options.sequencePrefix + name, static_cast<const uint8_t*>(m_readback.mapped),
                        m_color.extent.width, m_color.extent.height);
        }

        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        // Frames rendered while a scene loads in the background are not counted: the frames
        // after the swap are then exactly those of a run that loaded it with --scene.
        if (sceneLoadPending()) {
            ++framesWhileLoading;
            continue;
        }
        if (i >= warmup) cpuFrameMs.push_back(ms);
        ++i;
    }
    if (!m_options.loadAfterPath.empty())
        std::printf("--load-after: %u frames rendered while the scene loaded (not counted)\n", framesWhileLoading);

    VK_CHECK(vkDeviceWaitIdle(m_ctx.device));
    m_profiler.collectAll();
    if (auto* tiles = dynamic_cast<SplatTileRenderer*>(m_renderer.get())) {
        tiles->collectPending();  // the last frames' statistics (GPU idle)
        if (!m_options.tileStatsPrefix.empty()) writeTileStats(*tiles);
    }
    if (!m_options.screenshotPath.empty()) writeScreenshot();
    if (benchmark) writeBenchmark(cpuFrameMs, warmup);
    printMemorySummary(buildMemoryReport(m_ctx));
    return 0;
}

namespace {

// A perceptual "inferno"-like colour ramp for heatmaps: t in [0, 1] -> RGB8.
void heatColor(double t, uint8_t* rgb) {
    static const double stops[6][3] = {{0, 0, 4},     {40, 11, 84},  {101, 21, 110},
                                       {159, 42, 99}, {237, 105, 37}, {252, 255, 164}};
    t = std::clamp(t, 0.0, 1.0) * 5.0;
    const int i = std::min(int(t), 4);
    const double f = t - i;
    for (int c = 0; c < 3; ++c) rgb[c] = uint8_t(std::lround(stops[i][c] + f * (stops[i + 1][c] - stops[i][c])));
}

}  // namespace

void Application::writeTileStats(const SplatTileRenderer& tiles) {
    using nlohmann::json;
    const auto& st = tiles.lastStats();
    if (!st.valid || st.tiles.empty()) {
        std::printf("warning: no tile statistics available\n");
        return;
    }
    const uint32_t tx = st.tilesX, ty = st.tilesY, tileCount = tx * ty;
    uint64_t evaluated = 0, blended = 0, terminated = 0, slots = 0, pixels = 0, full = 0, listed = 0, fetched = 0;
    uint32_t maxList = 0, maxFetched = 0, nonEmpty = 0;
    std::vector<uint32_t> lists;
    for (uint32_t t = 0; t < tileCount; ++t) {
        const uint32_t* s = &st.tiles[size_t(t) * kTileStatCount];
        evaluated += s[kTileStatEvaluated];
        blended += s[kTileStatBlended];
        terminated += s[kTileStatTerminated];
        slots += s[kTileStatLaneSlots];
        pixels += s[kTileStatPixels];
        full += uint64_t(s[kTileStatListLength]) * s[kTileStatPixels];
        listed += s[kTileStatListLength];
        fetched += s[kTileStatFetched];
        maxList = std::max(maxList, s[kTileStatListLength]);
        maxFetched = std::max(maxFetched, s[kTileStatFetched]);
        nonEmpty += s[kTileStatListLength] > 0;
        lists.push_back(s[kTileStatListLength]);
    }
    std::sort(lists.begin(), lists.end());
    auto pct = [&](double q) { return lists[std::min(lists.size() - 1, size_t(q * double(lists.size())))]; };
    const TileControl& c = st.control;
    json j;
    j["scene"] = m_options.scenePath;
    j["renderer"] = "splat-tiles";
    j["tile_sort"] = tileSortModeName(tiles.tileSettings().sortMode);
    j["coverage_bound"] = coverageBoundName(tiles.settings().bound);
    j["bounds"] = tileBoundsName(tiles.settings().bounds);
    j["tile_size"] = tiles.tileSize();
    j["frame_index"] = st.frameIndex;
    j["resolution"] = {m_color.extent.width, m_color.extent.height};
    j["tiles"] = {tx, ty};
    j["gaussians"] = c.gaussianCount;
    j["visible"] = c.cullCounts[0];
    j["duplicates"] = c.duplicates;
    j["entries"] = c.entries;
    j["overflow"] = c.overflow;
    j["avg_splats_per_tile"] = double(c.duplicates) / double(tileCount);
    j["avg_splats_per_nonempty_tile"] = nonEmpty ? double(listed) / double(nonEmpty) : 0.0;
    j["nonempty_tiles"] = nonEmpty;
    j["list_length_percentiles"] = {{"p50", pct(0.5)}, {"p90", pct(0.9)}, {"p99", pct(0.99)}, {"max", maxList}};
    j["pairs_without_early_stop"] = full;
    j["pairs_evaluated"] = evaluated;
    j["pairs_blended"] = blended;
    j["fraction_pairs_skipped_by_early_stop"] = full ? 1.0 - double(evaluated) / double(full) : 0.0;
    j["pixels"] = pixels;
    j["pixels_terminated"] = terminated;
    j["fraction_pixels_terminated"] = pixels ? double(terminated) / double(pixels) : 0.0;
    j["lane_slots"] = slots;
    j["simd_efficiency_evaluated_over_lane_slots"] = slots ? double(evaluated) / double(slots) : 0.0;
    j["fraction_lane_slots_saved_vs_no_stop"] = full ? 1.0 - double(slots) / double(full) : 0.0;
    j["list_entries_fetched"] = fetched;
    j["fraction_list_entries_fetched"] = listed ? double(fetched) / double(listed) : 0.0;
    j["heatmap"] = {{"quantity", "list entries fetched (splats processed) per tile"}, {"scale_max", maxFetched}};

    const std::string jsonPath = m_options.tileStatsPrefix + ".json";
    const std::filesystem::path path(jsonPath);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << j.dump(2) << "\n";

    // Heatmap: every pixel coloured by its tile's "splats processed" (linear, 0 .. max).
    const uint32_t w = m_color.extent.width, h = m_color.extent.height;
    std::vector<uint8_t> img(size_t(w) * h * 4, 255);  // RGBA8 (writePngRgb drops alpha)
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t t = (y / tiles.tileSize()) * tx + x / tiles.tileSize();
            const double v = double(st.tiles[size_t(t) * kTileStatCount + kTileStatFetched]);
            heatColor(maxFetched ? v / double(maxFetched) : 0.0, &img[(size_t(y) * w + x) * 4]);
        }
    const std::string png = m_options.tileStatsPrefix + "_heatmap.png";
    writePngRgb(png, img.data(), w, h);
    // Phase 6: list entries per PIXEL (tile list length / tile area, so 8- and 16-px tiles are
    // comparable), linear 0 .. --tile-stats-max (default: this frame's maximum).
    {
        const double area = double(tiles.tileSize()) * tiles.tileSize();
        double maxDensity = m_options.tileStatsMax > 0.0f ? double(m_options.tileStatsMax) : 0.0;
        if (maxDensity == 0.0)
            for (uint32_t t = 0; t < tileCount; ++t)
                maxDensity = std::max(maxDensity, double(st.tiles[size_t(t) * kTileStatCount + kTileStatListLength]) / area);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const uint32_t t = (y / tiles.tileSize()) * tx + x / tiles.tileSize();
                const double v = double(st.tiles[size_t(t) * kTileStatCount + kTileStatListLength]) / area;
                heatColor(maxDensity > 0.0 ? v / maxDensity : 0.0, &img[(size_t(y) * w + x) * 4]);
            }
        writePngRgb(m_options.tileStatsPrefix + "_entries.png", img.data(), w, h);
        std::printf("Entries heatmap: %s_entries.png (list entries per pixel, scale 0 .. %.2f)\n",
                    m_options.tileStatsPrefix.c_str(), maxDensity);
    }
    std::printf("Tile stats written: %s, %s (pairs skipped by early stop %.1f%%, %.1f splats per tile)\n",
                jsonPath.c_str(), png.c_str(), 100.0 * j["fraction_pairs_skipped_by_early_stop"].get<double>(),
                j["avg_splats_per_tile"].get<double>());
}

void Application::writePipelineStats() {
    using nlohmann::json;
    json pipes = json::array();
    std::printf("Pipeline statistics (%s, renderer %s):\n", m_ctx.deviceName.c_str(), m_renderer->name());
    for (const auto& [name, pipeline] : m_renderer->pipelines()) {
        json execs = json::array();
        for (const PipelineExecutableStats& e : queryPipelineStatistics(m_ctx.device, pipeline)) {
            json stats = json::object();
            std::string line;
            for (const PipelineStatistic& s : e.statistics) {
                stats[s.name] = s.number;
                line += "  " + s.name + "=" + s.value;
            }
            execs.push_back({{"name", e.name}, {"description", e.description}, {"subgroup_size", e.subgroupSize},
                             {"statistics", stats}});
            std::printf("  %-44s [%s]%s\n", name.c_str(), e.name.c_str(), line.c_str());
        }
        pipes.push_back({{"pipeline", name}, {"executables", execs}});
    }
    json j = {{"device", m_ctx.deviceName}, {"driver", m_ctx.driverVersion}, {"renderer", m_renderer->name()},
              {"extension_available", m_ctx.pipelineExecutableInfo}, {"pipelines", pipes}};
    const std::filesystem::path path(m_options.pipelineStatsPath);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << j.dump(2) << "\n";
    std::printf("Pipeline statistics written: %s\n", m_options.pipelineStatsPath.c_str());
}

void Application::writeScreenshot() {
    // The device is idle and the copy was followed by a HOST barrier; invalidate in
    // case the readback memory is not host-coherent.
    VK_CHECK(vmaInvalidateAllocation(m_ctx.allocator, m_readback.allocation, 0, VK_WHOLE_SIZE));
    writePngRgb(m_options.screenshotPath, static_cast<const uint8_t*>(m_readback.mapped),
                m_color.extent.width, m_color.extent.height);
    std::printf("Screenshot written: %s (%ux%u)\n", m_options.screenshotPath.c_str(), m_color.extent.width,
                m_color.extent.height);
}

void Application::writeBenchmark(const std::vector<double>& cpuFrameMs, uint32_t warmupFrames) {
    using nlohmann::json;
    json j;
    j["app"] = "gsplat";
    j["build"] = kDebugBuild ? "Debug" : "Release";
    j["device"] = m_ctx.deviceName;
    j["driver"] = m_ctx.driverVersion;
    j["validation"] = m_ctx.validationEnabled;
    j["renderer"] = m_renderer->name();
    j["scene"] = m_options.scenePath;
    if (m_scene) {
        const GaussianScene& sc = *m_scene;
        j["scene_info"] = {
            {"gaussians", sc.count},
            {"sh_degree", sc.shDegree},
            {"up_axis", upAxisName(m_setup.upAxis)},
            {"file_mb", double(std::filesystem::file_size(m_scenePath)) / (1024.0 * 1024.0)},
            {"load_ms", {{"total", sc.timings.totalMs}, {"map", sc.timings.mapMs}, {"header", sc.timings.headerMs},
                         {"decode", sc.timings.decodeMs}, {"stats", sc.timings.statsMs}}},
            {"gpu_upload_ms", m_gpuScene.uploadMs},
            {"gpu_scene_mb", double(m_gpuScene.totalBytes) / (1024.0 * 1024.0)},
            {"scene_format", sceneFormatName(m_gpuScene.format)},
            {"scene_layout", sceneLayoutName(m_gpuScene.layout)},
            {"sh_degree_stored", m_gpuScene.shDegree},
            {"bytes_per_gaussian", m_gpuScene.bytesPerGaussian()},
            {"encode_ms", m_gpuScene.encodeMs},
            {"load_plus_encode_plus_upload_ms", sc.timings.totalMs + m_gpuScene.encodeMs + m_gpuScene.uploadMs},
            {"debug_fp16_positions", m_options.debugFp16Positions},
            {"sh_threshold", m_options.shThreshold},
            {"sh_variable", m_gpuScene.variableSh == VariableSh::Off       ? "off"
                            : m_gpuScene.variableSh == VariableSh::Compact ? "compact"
                                                                           : "degree-only"},
            {"sh_degree_histogram", m_gpuScene.degreeHistogram},
        };
        if (m_gpuScene.color == ColorEncoding::Codebook) {
            const CodebookStats& cb = m_gpuScene.codebook;
            j["scene_info"]["sh_codebook"] = {{"entries", cb.entries},     {"dimensions", cb.dimensions},
                                              {"samples", cb.samples},     {"iterations", cb.iterations},
                                              {"reseeded", cb.emptyReseeded}, {"train_ms", cb.trainMs},
                                              {"assign_ms", cb.assignMs},  {"total_ms", cb.totalMs},
                                              {"mse", cb.fullMse},         {"sample_mse", cb.sampleMse},
                                              {"simd", cb.simd}};
        }
        if (auto* points = dynamic_cast<PointCloudRenderer*>(m_renderer.get()))
            j["renderer_settings"] = {{"opacity_threshold", points->opacityThreshold},
                                      {"point_size_px", points->pointSizePx}};
        if (auto* splat = dynamic_cast<SplatRenderer*>(m_renderer.get()))
            j["renderer_settings"] = {{"splat_target_format", splatTargetFormatName(splat->settings().format)},
                                      {"coverage_bound", coverageBoundName(splat->settings().bound)},
                                      {"bounds", tileBoundsName(splat->settings().bounds)},
                                      {"sh_degree_evaluated", splat->effectiveShDegree()},
                                      {"sort", "cpu radix (4x8 bit LSD), indices copied to a device-local buffer"},
                                      {"instances_last_frame", splat->lastVisibleCount()}};
        if (auto* gs = dynamic_cast<SplatGpuSortRenderer*>(m_renderer.get())) {
            const SplatSettings ss = gs->settings();
            json culls = json::object();
            const auto& st = gs->lastStats();
            for (uint32_t r = 0; r < kCullReasonCount; ++r) culls[cullReasonName(r)] = st.control.cullCounts[r];
            j["renderer_settings"] = {
                {"splat_target_format", splatTargetFormatName(ss.format)},
                {"coverage_bound", coverageBoundName(ss.bound)},
                {"bounds", tileBoundsName(ss.bounds)},
                {"sh_degree_evaluated", gs->effectiveShDegree()},
                {"sort", "gpu radix (4x8 bit LSD, reduce-then-scan), indirect dispatch + draw"},
                {"radix_variant", gs->sortSettings().radixVariant == GpuRadixSort::Variant::Classic ? "classic" : "reorder"},
                {"sort_every", gs->sortSettings().sortEvery},
                {"sort_threshold_deg", gs->sortSettings().sortThresholdDeg},
                {"frames_rendered", gs->framesRendered()},
                {"sorts_performed", gs->sortsPerformed()},
                {"instances_last_frame", st.control.draw.instanceCount},
                {"cull_counts_last_frame", culls},
                {"stats_frame_index", st.frameIndex}};
        }
        if (auto* tr = dynamic_cast<SplatTileRenderer*>(m_renderer.get())) {
            const SplatSettings& ss = tr->settings();
            const TileSettings& ts = tr->tileSettings();
            const auto& st = tr->lastStats();
            json culls = json::object();
            for (uint32_t r = 0; r < kCullReasonCount; ++r) culls[cullReasonName(r)] = st.control.cullCounts[r];
            const uint32_t tileCount = tr->tilesX() * tr->tilesY();
            j["renderer_settings"] = {
                {"splat_target_format", splatTargetFormatName(ss.format)},
                {"coverage_bound", coverageBoundName(ss.bound)},
                {"sh_degree_evaluated", tr->effectiveShDegree()},
                {"tile_size", tr->tileSize()},
                {"bounds", tileBoundsName(ss.bounds)},
                {"sort_window", ts.sortWindow},
                {"render_batch", tr->renderBatch()},
                {"render_vote_every", ts.voteEvery},
                {"render_shared", ts.sharedAoS ? "aos" : "soa"},
                {"render_fp16", ts.fp16Math},
                {"coverage_mismatches", tr->coverageMismatches()},
                {"tiles", {tr->tilesX(), tr->tilesY()}},
                {"tile_sort", tileSortModeName(ts.sortMode)},
                {"tile_duplicate", tileDuplicateModeName(ts.duplicateMode)},
                {"radix_variant", ts.radixVariant == GpuRadixSort::Variant::Classic ? "classic" : "reorder"},
                {"instrumented_stats_shader", ts.stats},
                {"list_budget_mb", ts.budgetMB},
                {"list_mode", ts.adaptiveList ? "adaptive" : "fixed"},
                {"list_resizes", tr->listResizes()},
                {"render_splat", renderSplatModeName(ts.renderSplat)},
                {"render_splat_bytes", renderSplatBytes(ts.renderSplat)},
                {"list_capacity_entries", tr->capacity()},
                {"list_mb", double(tr->listBytes()) / (1024.0 * 1024.0)},
                {"renderer_buffers_mb", double(tr->totalBytes()) / (1024.0 * 1024.0)},
                {"visible_last_frame", st.control.cullCounts[0]},
                {"duplicates_last_frame", st.control.duplicates},
                {"entries_last_frame", st.control.entries},
                {"overflow_last_frame", st.control.overflow},
                {"avg_splats_per_tile_last_frame", double(st.control.duplicates) / double(std::max(tileCount, 1u))},
                {"max_duplicates", tr->maxDuplicates()},
                {"overflow_frames", tr->overflowFrames()},
                {"max_overflow", tr->maxOverflow()},
                {"cull_counts_last_frame", culls},
                {"stats_frame_index", st.frameIndex}};
        }
    }
    j["resolution"] = {m_color.extent.width, m_color.extent.height};
    j["present_mode"] = presentModeName(m_swapchain.presentMode);
    j["frames_total"] = m_options.frameCount();
    j["frames_warmup"] = warmupFrames;
    j["frames_measured"] = cpuFrameMs.size();
    j["camera_start"] = m_initialCamera.toJson();
    j["camera_path"] = "orbit 360 deg around target over all frames";

    json gpu = json::object();
    for (const GpuProfiler::Stats& s : m_profiler.stats()) {
        if (s.count == 0) continue;
        gpu[s.name] = {{"avg", s.sumMs / double(s.count)}, {"min", s.minMs}, {"max", s.maxMs}, {"samples", s.count}};
    }
    j["gpu_ms"] = gpu;

    // CPU stages inside the renderer (e.g. the Phase 2 CPU sort), measured frames only.
    json cpu = json::object();
    for (const TimingStats& s : m_cpuProfiler.stats()) {
        if (s.count == 0) continue;
        cpu[s.name] = {{"avg", s.sumMs / double(s.count)}, {"min", s.minMs}, {"max", s.maxMs}, {"samples", s.count}};
    }
    j["cpu_ms"] = cpu;

    std::vector<double> sorted = cpuFrameMs;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double v : sorted) sum += v;
    const double avg = sorted.empty() ? 0.0 : sum / double(sorted.size());
    j["cpu_frame_ms"] = {
        {"avg", avg},
        {"min", sorted.empty() ? 0.0 : sorted.front()},
        {"max", sorted.empty() ? 0.0 : sorted.back()},
        {"median", sorted.empty() ? 0.0 : sorted[sorted.size() / 2]},
        {"p95", sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, sorted.size() * 95 / 100)]},
    };
    j["fps_avg"] = avg > 0.0 ? 1000.0 / avg : 0.0;

    // GPU memory: what this process uses in device-local heaps (VMA budget query).
    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetMemoryProperties(m_ctx.allocator, &memProps);
    std::vector<VmaBudget> budgets(memProps->memoryHeapCount);
    vmaGetHeapBudgets(m_ctx.allocator, budgets.data());
    double usageMb = 0.0, allocMb = 0.0, budgetMb = 0.0;
    for (uint32_t h = 0; h < memProps->memoryHeapCount; ++h) {
        if (!(memProps->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        usageMb += double(budgets[h].usage) / (1024.0 * 1024.0);
        allocMb += double(budgets[h].statistics.allocationBytes) / (1024.0 * 1024.0);
        budgetMb += double(budgets[h].budget) / (1024.0 * 1024.0);
    }
    j["gpu_memory_mb"] = {{"device_local_process_usage", usageMb},
                          {"device_local_vma_allocations", allocMb},
                          {"device_local_budget", budgetMb}};
    j["gpu_memory"] = memoryReportJson(buildMemoryReport(m_ctx));

    const std::filesystem::path path(m_options.benchmarkPath);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write benchmark file: " + m_options.benchmarkPath);
    file << j.dump(2) << "\n";
    std::printf("Benchmark written: %s (avg CPU frame %.3f ms)\n", m_options.benchmarkPath.c_str(), avg);
}

// ---------------------------------------------------------------------------
// Runtime scene loading (UI: Open scene..., Ctrl+O, drag and drop, recent scenes;
// automation: --load-after). All of them go through requestSceneLoad().
// ---------------------------------------------------------------------------

void Application::requestSceneLoad(const std::filesystem::path& path) {
    if (m_loadJob.busy()) {
        // One load at a time: cancel the running one, start this one when it has finished.
        m_queuedScenePath = path;
        m_loadJob.cancel();
        return;
    }
    // The flags --cameras, --up, --camera and --camera-index describe the start-up scene;
    // a scene opened later gets the automatic per-scene setup (cameras.json next to the file,
    // default up axis, auto-framing), like `--scene <file>` without those flags.
    m_loadJob.start({path, "", ""}, compressionOptions());
    m_sceneMessage.clear();
    std::printf("Loading scene in the background: %s\n", pathToUtf8(path).c_str());
    std::fflush(stdout);
}

void Application::pollSceneLoad() {
    // Drag and drop: the last .ply of the batch; anything else is reported.
    if (!m_droppedPaths.empty()) {
        std::filesystem::path ply;
        for (const std::filesystem::path& p : m_droppedPaths) {
            std::string ext = pathToUtf8(p.extension());
            for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
            if (ext == ".ply") ply = p;
        }
        if (!ply.empty()) {
            requestSceneLoad(ply);
        } else {
            m_sceneMessage = "Dropped file is not a .ply: " + pathToUtf8(m_droppedPaths.front().filename());
            m_sceneMessageIsError = true;
        }
        m_droppedPaths.clear();
    }

    if (std::optional<SceneLoadJob::Result> result = m_loadJob.take()) {
        const std::string name = pathToUtf8(m_loadJob.path().filename());
        if (result->cancelled) {
            m_sceneMessage = "Cancelled loading " + name;
            m_sceneMessageIsError = false;
        } else if (!result->error.empty()) {
            m_sceneMessage = "Could not load " + name + ": " + result->error + "\nKeeping the current scene.";
            m_sceneMessageIsError = true;
        } else {
            installScene(result->loaded);
        }
        std::printf("%s\n", m_sceneMessage.c_str());
        std::fflush(stdout);
    }
    if (!m_loadJob.busy() && !m_queuedScenePath.empty()) {
        const std::filesystem::path next = std::exchange(m_queuedScenePath, {});
        requestSceneLoad(next);
    }
}

std::string Application::rendererForNewScene() const {
    if (!m_pendingRenderer.empty()) return m_pendingRenderer;
    const std::string current = m_renderer->name();
    if (current != "test") return current;
    // "test" is what runs without a scene: switch to the requested technique (default
    // splat-tiles, as with --scene), unless "test" itself was asked for.
    if (m_options.renderer == "test") return m_options.rendererExplicit ? "test" : "splat-tiles";
    return m_options.renderer;
}

void Application::waitForSubmittedFrames() {
    // The timeline value of the last submitted frame: once the GPU has reached it, no frame
    // in flight uses any resource any more. (With one queue this is what vkDeviceWaitIdle
    // would do, but it states exactly what we wait for.)
    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_frameTimeline;
    waitInfo.pValues = &m_timelineValue;
    VK_CHECK(vkWaitSemaphores(m_ctx.device, &waitInfo, UINT64_MAX));
}

bool Application::installScene(LoadedScene& loaded) {
    const auto start = std::chrono::steady_clock::now();
    const std::string name = pathToUtf8(loaded.path.filename());
    const std::string rendererName = rendererForNewScene();
    const std::string previousRenderer = m_renderer->name();

    // 1) The frames in flight read the current scene and the renderer's scene-sized
    //    buffers (splats, sort keys, tile lists, ...): wait for them, then free the renderer
    //    first so that its memory is available to the new scene.
    waitForSubmittedFrames();
    m_renderer->destroy(m_ctx);
    m_renderer.reset();
    m_pendingRenderer.clear();

    // 2) Keep the previous scene (CPU + GPU) until the new one is completely set up.
    GpuScene previousGpu = std::exchange(m_gpuScene, GpuScene{});
    std::unique_ptr<GaussianScene> previousScene = std::move(m_scene);
    SceneSetup previousSetup = std::move(m_setup);
    std::filesystem::path previousPath = std::move(m_scenePath);
    try {
        m_scene = std::move(loaded.scene);
        m_setup = std::move(loaded.setup);
        m_scenePath = loaded.path;
        // The worker encoded with the options of the moment it started; if the UI changed
        // the scene format since then, encode again here.
        const bool usable = loaded.encoded && sameCompression(loaded.encodedWith, compressionOptions());
        uploadScene(usable ? &*loaded.encoded : nullptr);
        createRenderer(rendererName);  // all of its scene-sized buffers are sized for the new scene
        if (m_options.debugFailSceneInstall)
            throw std::runtime_error("simulated GPU failure (--debug-fail-scene-install)");
    } catch (const std::exception& e) {
        // 3) Roll back: free whatever the new scene got and bring the previous one back.
        VK_CHECK(vkDeviceWaitIdle(m_ctx.device));
        if (m_renderer) m_renderer->destroy(m_ctx);
        m_renderer.reset();
        m_gpuScene.destroy(m_ctx);
        m_gpuScene = previousGpu;
        m_scene = std::move(previousScene);
        m_setup = std::move(previousSetup);
        m_scenePath = std::move(previousPath);
        createRenderer(previousRenderer);  // this configuration was running a moment ago
        m_sceneMessage = "Could not load " + name + " on the GPU: " + e.what() + "\nKeeping the current scene.";
        m_sceneMessageIsError = true;
        return false;
    }
    previousGpu.destroy(m_ctx);
    previousScene.reset();
    m_pendingSceneUpload = false;  // the upload above already used the current options

    // 4) Per-scene state, as at start-up: camera (auto-framed), training-camera preset,
    //    statistics. Renderer state (sort-every counters, cull/overflow statistics, the
    //    adaptive list capacity) starts fresh with the new renderer.
    m_options.scenePath = pathToUtf8(m_scenePath);
    m_options.cameraPath.clear();
    m_options.cameraIndex.reset();
    m_options.camerasPath.clear();
    m_options.upAxis.clear();
    m_camera = initialSceneCamera();
    m_initialCamera = m_camera;
    m_uiCameraIndex = 0;
    m_profiler.resetAccumulators(m_frameIndex);
    m_cpuProfiler.resetAccumulators();

    char text[256];
    std::snprintf(text, sizeof(text), "Loaded %s: %u gaussians in %.2f s (swap %.0f ms), renderer %s", name.c_str(),
                  m_scene->count, m_loadJob.elapsedSeconds(),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                  m_renderer->name());
    m_sceneMessage = text;
    m_sceneMessageIsError = false;
    if (!m_options.automated()) addRecentScene(m_scenePath);
    return true;
}

void Application::openSceneDialog() {
    std::filesystem::path folder;
    if (!m_scenePath.empty()) folder = m_scenePath.parent_path();
    else if (!m_recentScenes.empty()) folder = m_recentScenes.front().parent_path();
    const FileDialogResult r = openPlyFileDialog(m_window, folder);
    ImGui::GetIO().ClearInputKeys();  // e.g. the Ctrl and O key releases went to the dialog
    switch (r.status) {
        case FileDialogResult::Status::Selected: requestSceneLoad(r.path); break;
        case FileDialogResult::Status::Cancelled: break;
        case FileDialogResult::Status::Unsupported:
        case FileDialogResult::Status::Error:
            m_sceneMessage = r.error;
            m_sceneMessageIsError = true;
            break;
    }
}

void Application::buildSceneLoadUi() {
    if (m_loadJob.busy()) {
        const SceneLoadJob::Stage stage = m_loadJob.stage();
        const char* what = m_loadJob.cancelling()                   ? "cancelling"
                           : stage == SceneLoadJob::Stage::Encoding ? "encoding"
                           : stage == SceneLoadJob::Stage::Finished ? "uploading"
                                                                    : "parsing";
        ImGui::Text("Loading %s (%s, %.1f s)", pathToUtf8(m_loadJob.path().filename()).c_str(), what,
                    m_loadJob.elapsedSeconds());
        // Indeterminate progress bar (animated negative fraction): the parser reports no progress.
        ImGui::ProgressBar(-1.0f * float(ImGui::GetTime()), ImVec2(200.0f, 0.0f), "");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_loadJob.cancelling());
        if (ImGui::Button("Cancel")) {
            m_loadJob.cancel();
            m_queuedScenePath.clear();
        }
        ImGui::EndDisabled();
    } else if (ImGui::Button("Open scene... (Ctrl+O)")) {
        m_openDialogRequested = true;
    }
    if (!m_recentScenes.empty() && ImGui::TreeNode("Recent scenes")) {
        for (size_t i = 0; i < m_recentScenes.size(); ++i) {
            const std::filesystem::path p = m_recentScenes[i];
            const std::string label = pathToUtf8(p.filename()) + "##recent" + std::to_string(i);
            if (ImGui::Selectable(label.c_str())) requestSceneLoad(p);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pathToUtf8(p).c_str());
        }
        ImGui::TreePop();
    }
    if (!m_sceneMessage.empty()) {
        const ImVec4 color = m_sceneMessageIsError ? ImVec4(1.0f, 0.45f, 0.4f, 1.0f) : ImVec4(0.6f, 0.9f, 0.6f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 420.0f);
        ImGui::TextUnformatted(m_sceneMessage.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        if (ImGui::SmallButton("Dismiss")) m_sceneMessage.clear();
    }
}

// Recent scenes: at most kMaxRecentScenes paths, most recent first, kept in
// gsplat_recent_scenes.json next to the executable (interactive runs only). The file is a
// convenience: any I/O or parse error is reported and otherwise ignored.
namespace {
std::filesystem::path recentScenesFile() { return executableDirectory() / "gsplat_recent_scenes.json"; }
}  // namespace

void Application::loadRecentScenes() {
    try {
        const std::filesystem::path file = recentScenesFile();
        if (!std::filesystem::exists(file)) return;
        std::ifstream in(file);
        const nlohmann::json j = nlohmann::json::parse(in);
        m_recentScenes.clear();
        for (const auto& entry : j.at("recent_scenes"))
            if (m_recentScenes.size() < kMaxRecentScenes) m_recentScenes.push_back(pathFromUtf8(entry.get<std::string>()));
    } catch (const std::exception& e) {
        std::printf("warning: ignoring the recent-scenes file: %s\n", e.what());
        m_recentScenes.clear();
    }
}

void Application::saveRecentScenes() {
    try {
        nlohmann::json list = nlohmann::json::array();
        for (const std::filesystem::path& p : m_recentScenes) list.push_back(pathToUtf8(p));
        std::ofstream out(recentScenesFile());
        if (!out) throw std::runtime_error("cannot write " + pathToUtf8(recentScenesFile()));
        out << nlohmann::json{{"recent_scenes", list}}.dump(2) << "\n";
    } catch (const std::exception& e) {
        std::printf("warning: could not save the recent-scenes list: %s\n", e.what());
    }
}

void Application::addRecentScene(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(path, ec);
    if (ec) p = path;
    p = p.lexically_normal();
    std::erase(m_recentScenes, p);
    m_recentScenes.insert(m_recentScenes.begin(), p);
    if (m_recentScenes.size() > kMaxRecentScenes) m_recentScenes.resize(kMaxRecentScenes);
    saveRecentScenes();
}

}  // namespace gsr
