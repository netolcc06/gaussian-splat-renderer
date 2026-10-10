// CPU reference renderer: a faithful, multi-threaded port of the FORWARD pass of the
// INRIA rasterizer (diff-gaussian-rasterization). Every GPU technique is diffed
// against its output (tools/gsplat_ref, tools/imgdiff, tools/run_reference_checks.ps1).
//
// Pipeline (same structure as the CUDA code, but plain loops):
//   1. preprocess  every gaussian -> ProjectedSplat (splat_math.h), in parallel
//   2. sort        front-to-back by view depth (DepthSorter, the same code the GPU path uses)
//   3. bin         for each 16x16 tile, the splats whose TileRect contains it, in depth order
//                  (INRIA gets the same lists from one global sort of (tile | depth) keys)
//   4. rasterize   each pixel walks its tile's list front to back:
//                    alpha = min(0.99, o * exp(power)); skip power > 0 or alpha < 1/255
//                    test_T = T (1 - alpha); if test_T < 1e-4: stop (this splat is NOT added)
//                    C += color * alpha * T; T = test_T
//                  and outputs C + T * background. Tiles are processed in parallel.
//
// Coverage: by default a splat is evaluated on EVERY pixel of its tile rectangle (the
// INRIA semantics, CoverageBound::Tile); the GPU quads cover exactly that rectangle, so
// coverage matches pixel for pixel. CoverageBound::Radius restricts it to the square of
// half-size `radius` (what a tighter GPU quad would cover). See docs/notes/phase2.md.
#pragma once

#include "splat/splat_math.h"
#include "splat/tile_coverage.h"

#include <cstdint>
#include <vector>

namespace gsr {

struct GaussianScene;

struct ReferenceOptions {
    int maxShDegree = 3;            // evaluated bands = min(scene degree, this)
    bool earlyTermination = true;   // INRIA's T < 1e-4 stop; false = blend every splat (what
                                    // back-to-front hardware blending computes)
    CoverageBound bound = CoverageBound::Tile;
    // Phase 6: which tiles a splat is binned into (tile_coverage.h). rect = INRIA; tight
    // drops tiles where the splat's alpha < 1/255 everywhere (identical image); cutoff-only
    // drops the 3-sigma clip (not INRIA).
    TileBounds bounds = TileBounds::Rect;
    // Phase 6 StopThePop-style ordering (NOT INRIA): 0 = blend in the list's centre-depth
    // order; K > 0 = per-pixel re-sorting window of K splats ordered by the depth of each
    // gaussian's maximum along the pixel's ray (the GPU's `--sort-window K`, same algorithm);
    // kExactPixelSort = every pixel sorts all its splats by that depth (the ground truth).
    uint32_t sortWindow = 0;
    static constexpr uint32_t kExactPixelSort = 0xFFFFFFFFu;
    glm::vec3 background{0.0f};     // INRIA default: black
    unsigned threads = 0;           // 0 = std::thread::hardware_concurrency()
};

struct ReferenceStats {
    uint32_t gaussians = 0;
    uint32_t visible = 0;  // drawn (cullReason == Visible)
    uint32_t culledNear = 0, culledSingular = 0, culledOffScreen = 0;
    uint64_t tileInstances = 0;      // sum over splats of tiles touched
    uint64_t evaluatedFragments = 0; // (pixel, splat) pairs inside the coverage bound: what the GPU's
                                     // quads rasterize = fragment shader invocations (without early stop)
    uint64_t blendedFragments = 0;   // (pixel, splat) pairs actually blended
    uint64_t terminatedPixels = 0;   // pixels that hit the T < 1e-4 stop
    double preprocessMs = 0.0, sortMs = 0.0, binMs = 0.0, rasterMs = 0.0, totalMs = 0.0;
    unsigned threads = 0;
};

struct ReferenceImage {
    uint32_t width = 0, height = 0;
    std::vector<float> rgb;  // row 0 = top, 3 floats per pixel, NOT clamped (INRIA out_color)
    ReferenceStats stats;
};

ReferenceImage renderReference(const GaussianScene& scene, const SplatCamera& cam, const ReferenceOptions& options);

}  // namespace gsr
