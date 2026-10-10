// Command-line options. The automation flags (--screenshot, --benchmark) make the
// renderer usable without a human looking at the window.
#pragma once

#include <optional>
#include <string>

namespace gsr {

struct Options {
    std::string scenePath;       // --scene <file.ply>
    std::string cameraPath;      // --camera <file.json>
    std::string camerasPath;     // --cameras <cameras.json> (default: next to the scene)
    std::optional<uint32_t> cameraIndex;  // --camera-index <i>: training camera preset
    std::string upAxis;          // --up <+y|-y|+z|-z> (empty = auto, see scene_transform.h)
    std::string renderer;        // --renderer <name> (empty = "splat-tiles" with a scene (Phase 6; was "splat"), else "test")
    bool rendererExplicit = false;  // --renderer was given (else the no-scene "test" switches to splat-tiles on load)
    // Splat renderer settings (Phase 2).
    std::optional<uint32_t> shDegree;     // --sh-degree <0-3>: max SH degree evaluated (default: all)
    std::string splatFormat = "rgba16f";  // --splat-format <rgba8|rgba16f|rgba32f>: blend target
    std::string splatBound = "tile";      // --splat-bound <tile|radius>: quad extent
    // GPU sort renderer (Phase 3).
    uint32_t sortEvery = 1;               // --sort-every <N>: re-sort every N frames (0 = threshold only)
    float sortThresholdDeg = 0.0f;        // --sort-threshold <deg>: re-sort when the camera moved more
    std::string radixVariant;             // --radix-variant <classic|reorder> (default: classic for
                                          // splat-gpusort, reorder for splat-tiles)
    // Compute tile rasterizer (Phase 4).
    std::string tileSort = "binning";     // --tile-sort <binning|global>
    std::string tileDuplicate = "per-entry";  // --tile-duplicate <per-entry|per-splat>
    uint32_t tileBudgetMB = 256;          // --tile-budget-mb <MB>: duplicated list + its sort scratch
    std::string tileStatsPrefix;          // --tile-stats <prefix>: instrumented render; writes
                                          // <prefix>.json and <prefix>_heatmap.png at the end of the run
    float tileStatsMax = 0.0f;            // --tile-stats-max <v>: entries heatmap scale (entries per pixel; 0 = auto)
    std::string renderSplat = "full";     // --render-splat <full|p32|p24>: splat-tiles per-entry record
    std::string tileList = "adaptive";    // --tile-list <fixed|adaptive> (Phase 6 default: adaptive; Phase 5: fixed)
    // Scene compression (Phase 5).
    std::string sceneFormat = "f32";      // --scene-format <f32|f16|q8|cb>
    std::string sceneLayout;              // --scene-layout <soa|split|aos> (default: soa for f32, else split)
    std::optional<uint32_t> sceneShDegree;  // --scene-sh-degree <0-3>: drop stored SH bands at load
    uint32_t codebookSize = 4096;         // --sh-codebook-size <K>
    uint32_t codebookIterations = 10;     // --sh-codebook-iterations <n>
    bool debugFp16Positions = false;      // --debug-fp16-positions: round positions through fp16
    // Quality vs cost (Phase 6).
    std::string bounds = "tight";         // --bounds <rect|tight|cutoff-only> (Phase 6 default: tight = exact)
    uint32_t tileSize = 16;               // --tile-size <8|16>: splat-tiles binning + render tile
    uint32_t sortWindow = 0;              // --sort-window <K>: splat-tiles StopThePop per-pixel window (0 = off)
    uint32_t renderBatch = 0;             // --render-batch <n>: splats per shared batch (experiment)
    uint32_t renderVoteEvery = 1;         // --render-vote-every <n>: whole-tile vote every n batches (0 = never)
    std::string renderShared = "soa";     // --render-shared <soa|aos>: shared-memory layout (experiment)
    bool renderFp16 = false;              // --render-fp16: exp/alpha in fp16 (experiment)
    std::string pipelineStatsPath;        // --pipeline-stats <out.json>: dump compiled-pipeline statistics, exit
    float shThreshold = -1.0f;            // --sh-threshold <x/255>: per-gaussian SH degree (< 0: off)
    std::string sequencePrefix;           // --sequence <prefix>: --screenshot run writes every frame
    // Automation: rotate the camera about its pivot by this many degrees per frame in
    // --screenshot runs too (benchmarks always orbit 360 degrees over the run).
    float orbitStepDeg = 0.0f;            // --orbit-step <deg>
    std::string screenshotPath;  // --screenshot <out.png>
    std::string benchmarkPath;   // --benchmark <out.json>
    std::optional<uint32_t> frames;  // --frames N
    uint32_t width = 1280;       // --width
    uint32_t height = 720;       // --height
    bool validation = false;     // --validation (always on in Debug builds)
    bool vsync = true;           // --no-vsync (interactive only; automation never waits for vsync)
    // Test/automation hooks of the UI's scene loading (same code path as Open scene... / drag and drop).
    std::string loadAfterPath;        // --load-after <N> <file.ply>: request this scene after N frames
    uint32_t loadAfterFrames = 0;
    bool debugFailSceneInstall = false;  // --debug-fail-scene-install: simulate a GPU failure while
                                         // installing a UI-loaded scene (tests the rollback)

    bool automated() const {
        return !screenshotPath.empty() || !benchmarkPath.empty() || !pipelineStatsPath.empty();
    }
    uint32_t frameCount() const;  // --frames, or the default for the chosen mode
};

// Parses argv. On error or --help prints usage and returns std::nullopt;
// `exitCode` is then set to the code main() should return (0 for --help, 2 for errors).
std::optional<Options> parseCommandLine(int argc, char** argv, int& exitCode);

}  // namespace gsr
