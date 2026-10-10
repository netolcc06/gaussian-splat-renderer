// GaussianScene: the CPU-side copy of a 3DGS scene, laid out for direct GPU upload.
//
// Layout: structure of arrays (SoA), one std::vector per attribute. Every array is
// uploaded as-is into its own storage buffer (see render/gpu_scene.h), so each
// element type is chosen to have the same size and alignment under GLSL std430:
//
//   array          element      bytes  std430 view in GLSL
//   posAlpha       vec4         16     vec4  posAlpha[]   xyz = mean, w = alpha (sigmoid)
//   scales         vec4         16     vec4  scales[]     xyz = sigma (exp), w = 0
//   rotations      vec4         16     vec4  rotations[]  normalized quaternion, (x,y,z,w) = (qw,qx,qy,qz)
//   sh             float        4      float sh[]         sh[(i*K + k)*3 + c]
//
// with K = (shDegree+1)^2 coefficients per colour channel, k = 0 the DC term
// (f_dc), c = 0/1/2 for R/G/B. That is "per-coefficient RGB triplets" (the PLY file
// stores f_rest channel-major; the loader transposes, see ply_loader.cpp).
//
// Why SoA rather than one AoS struct per gaussian:
//   - Passes read only what they need: the point view reads posAlpha + 3 floats of
//     sh, a depth sort reads only positions, the Phase 2 preprocess reads everything.
//     With AoS every pass would pull the full 248-byte record through the cache.
//   - The SH array dominates (192 of ~240 bytes per gaussian at degree 3) and can be
//     quantized or truncated to a lower degree later without touching the rest.
//   - vec3 arrays have a 16-byte stride under std430, so vec3 data is stored as vec4
//     with a useful payload in w (alpha) or a zero pad (scales). SH is a flat float
//     array instead of vec3/vec4 elements so no padding is wasted on the largest
//     array (a vec4-per-coefficient layout would cost +33%).
//   Phase 5 revisits this (fp16, packing, SH compaction).
//
// The PLY values are pre-activation. The raw values are kept in the `raw*` arrays
// (exactly as read from the file, for tools/tests/debugging); the GPU arrays above
// hold the activated values:
//   alpha = sigmoid(opacity)            sigma = exp(scale)            q = rot / |rot|
// Positions and SH have no activation and are stored once.
//
// Coordinates are kept in the file's own (data) space. The y-down -> y-up fix for
// COLMAP scenes is a model matrix (scene/scene_transform.h), not baked into the data.
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gsr {

struct SceneStats {
    glm::vec3 bboxMin{0.0f}, bboxMax{0.0f};
    // Robust bounding box: per-axis percentiles (p = kRobustPercentile and 1 - p),
    // same definition as tools/ply_info.py: sorted[int(p * (n - 1))].
    glm::vec3 robustMin{0.0f}, robustMax{0.0f};
    static constexpr float kRobustPercentile = 0.01f;

    float alphaMin = 0.0f, alphaMax = 0.0f, alphaMean = 0.0f;
    uint32_t alphaBelowVisible = 0;  // alpha < 1/255: never contributes to a pixel
    float sigmaMin = 0.0f, sigmaMax = 0.0f;  // over all three axes
    float sigmaMaxMedian = 0.0f;             // median over gaussians of max(sigma_x, sigma_y, sigma_z)
    float quatNormMin = 0.0f, quatNormMax = 0.0f;  // raw (pre-normalization) quaternion lengths
    uint32_t zeroQuaternions = 0;  // replaced by identity
    uint64_t nonFiniteValues = 0;  // NaN/Inf anywhere in the decoded properties
};

struct LoadTimings {
    double mapMs = 0.0;     // open + map the file
    double headerMs = 0.0;  // header parse + validation
    double decodeMs = 0.0;  // payload decode + activation (multi-threaded)
    double statsMs = 0.0;   // bbox, percentiles, ranges
    double totalMs = 0.0;
};

struct GaussianScene {
    std::string sourcePath;
    uint32_t count = 0;
    int shDegree = 0;  // 0..3

    // GPU-ready (activated) arrays, see the file comment for the layout.
    std::vector<glm::vec4> posAlpha;
    std::vector<glm::vec4> scales;
    std::vector<glm::vec4> rotations;
    std::vector<float> sh;

    // Raw pre-activation values, as stored in the file.
    std::vector<float> rawOpacity;       // logit(alpha)
    std::vector<glm::vec3> rawLogScale;  // log(sigma)
    std::vector<glm::vec4> rawRotation;  // (rot_0, rot_1, rot_2, rot_3) = (w, x, y, z), unnormalized

    SceneStats stats;
    LoadTimings timings;
    std::vector<std::string> comments;   // header comment lines
    std::vector<std::string> warnings;   // non-fatal issues found while loading

    uint32_t shCoeffsPerChannel() const { return uint32_t((shDegree + 1) * (shDegree + 1)); }
    // RGB triplet of SH coefficient k of gaussian i.
    glm::vec3 shCoeff(uint32_t i, uint32_t k) const {
        const size_t base = (size_t(i) * shCoeffsPerChannel() + k) * 3;
        return {sh[base], sh[base + 1], sh[base + 2]};
    }
    // Degree-0 colour as the reference computes it before clamping: 0.5 + C0 * f_dc.
    glm::vec3 dcColor(uint32_t i) const;

    size_t cpuBytes() const;
};

// SH band-0 constant (sh_utils.py / auxiliary.h).
inline constexpr float kShC0 = 0.28209479177387814f;

// Fills scene.stats from the arrays (called by the loader; exposed for tests/tools).
void computeSceneStats(GaussianScene& scene);
// The part of the stats that needs only positions and alphas: bounding boxes (incl. the robust
// one used for auto-framing) and the alpha statistics. computeSceneStats calls it; the web build
// calls it on the positions it keeps from a streamed load.
void computePositionStats(const glm::vec4* posAlpha, size_t n, SceneStats& stats);

}  // namespace gsr
