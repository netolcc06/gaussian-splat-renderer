// The per-gaussian math of 3D Gaussian Splatting, exactly as the INRIA reference
// rasterizer (graphdeco-inria/diff-gaussian-rasterization, cuda_rasterizer/forward.cu)
// does it in its forward pass. No Vulkan here: this is shared by the CPU reference
// renderer (tools/gsplat_ref), the unit tests, and the CPU depth sort of the GPU path.
// The GPU preprocess (shaders/splat_common.glsl) mirrors these functions line by line;
// keep the two in sync.
//
// Spaces and conventions (see docs/notes/phase2.md for the derivations):
//   - DATA space: the coordinates of the .ply file. All splatting math happens here,
//     because the SH coefficients and covariances were fitted here (phase1.md).
//   - INRIA CAMERA space: x right, y DOWN, z FORWARD (OpenCV/COLMAP). Our viewer camera
//     looks down -z with +y up, so viewInria = diag(1,-1,-1) * view * model.
//   - PIXEL coordinates (INRIA convention): pixel (i, j) has coordinates (i, j), i.e. the
//     pixel CENTRE is at the integer; row 0 is the top row. In Vulkan terms that is
//     gl_FragCoord.xy - 0.5.
#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>

namespace gsr {

// --- INRIA constants (forward.cu / auxiliary.h) -------------------------------------
inline constexpr float kNearCullDepth = 0.2f;       // in_frustum(): drop if view z <= 0.2
inline constexpr float kLowPassFilter = 0.3f;       // added to the 2D covariance diagonal (px^2)
inline constexpr float kFrustumClampFactor = 1.3f;  // Jacobian evaluated at x/z, y/z clamped to 1.3*tan(fov/2)
inline constexpr float kMaxAlpha = 0.99f;           // alpha = min(0.99, opacity * G)
inline constexpr float kMinAlpha = 1.0f / 255.0f;   // smaller alphas are skipped
inline constexpr float kMinTransmittance = 1e-4f;   // front-to-back stops once T would drop below this
inline constexpr int kTileSize = 16;                // BLOCK_X = BLOCK_Y = 16

// SH constants (sh_utils.py, auxiliary.h). See docs/notes/ply-format.md.
inline constexpr float kShC1 = 0.4886025119029199f;
inline constexpr float kShC2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                   -1.0925484305920792f, 0.5462742152960396f};
inline constexpr float kShC3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                                   0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                                   -0.5900435899266435f};

// Everything the splatting math needs to know about the camera.
struct SplatCamera {
    glm::mat4 viewInria{1.0f};  // data space -> INRIA camera space (rigid transform)
    glm::vec3 position{0.0f};   // camera centre in data space (SH view directions)
    uint32_t width = 0, height = 0;
    float tanHalfFovX = 1.0f, tanHalfFovY = 1.0f;
    float focalX = 1.0f, focalY = 1.0f;  // pixels: focal = size / (2 tan(fov/2))
    uint32_t tilesX = 0, tilesY = 0;     // ceil(size / 16)
};

// From an INRIA-convention view matrix and the symmetric frustum's half-angles.
SplatCamera makeSplatCamera(const glm::mat4& viewInria, uint32_t width, uint32_t height, float tanHalfFovX,
                            float tanHalfFovY);

// From our viewer's matrices: `view` (world -> eye, -z forward, +y up), `proj` (Vulkan
// perspective from Camera::projection, y flipped) and `model` (data -> world, the scene
// transform). Only the field of view is taken from `proj`; near/far play no role in
// splatting (INRIA culls at view z <= 0.2 regardless of the near plane).
SplatCamera splatCameraFromViewer(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model,
                                  uint32_t width, uint32_t height);

// Data-space point -> INRIA camera space (INRIA: transformPoint4x3). Written out per
// component so the CPU depth sort and the preprocess get bit-identical depths.
inline glm::vec3 transformPoint(const glm::mat4& m, const glm::vec3& p) {
    return {m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
            m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
            m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2]};
}
inline float viewDepth(const glm::mat4& viewInria, const glm::vec3& p) {
    return viewInria[0][2] * p.x + viewInria[1][2] * p.y + viewInria[2][2] * p.z + viewInria[3][2];
}

// Rotation matrix of a unit quaternion q = (w, x, y, z) stored as glm::vec4(w, x, y, z)
// (our GaussianScene layout: real part first).
glm::mat3 quaternionToMatrix(const glm::vec4& q);

// 3D covariance Sigma = R S S^T R^T, S = diag(sigma) (computeCov3D).
glm::mat3 computeCov3D(const glm::vec3& sigma, const glm::vec4& q);

// EWA splatting: 2D covariance (pixels^2) of the projected gaussian, including the
// 0.3 px^2 low-pass dilation (computeCov2D). `tCam` is the mean in INRIA camera space.
// Returned as the symmetric 2x2 matrix [[x, y], [y, z]] packed into a vec3.
glm::vec3 computeCov2D(const glm::vec3& tCam, const glm::mat3& cov3D, const SplatCamera& cam);

// Mean in INRIA camera space -> pixel coordinates (INRIA: ndc2Pix of the projection).
glm::vec2 projectToPixel(const glm::vec3& tCam, const SplatCamera& cam);

// Inverse of the 2D covariance ("conic") and the pixel radius ceil(3 sqrt(lambda_max)).
// Returns false if the covariance is singular (det == 0), like INRIA.
bool conicAndRadius(const glm::vec3& cov2D, glm::vec3& conic, float& radius);

// Screen tiles touched by a splat (INRIA getRect): [min, max) in tile units, clamped to
// the tile grid. Computed with the same float expressions and integer truncation as CUDA.
struct TileRect {
    int minX = 0, minY = 0, maxX = 0, maxY = 0;
    bool empty() const { return maxX <= minX || maxY <= minY; }
};
TileRect computeTileRect(const glm::vec2& center, float radius, uint32_t tilesX, uint32_t tilesY);

// View-dependent colour from SH coefficients (computeColorFromSH). `coeffs` holds the
// RGB triplets of K = (storedDegree+1)^2 coefficients; `degree` <= storedDegree bands are
// evaluated. `dir` = normalize(mean - camera position), data space. Returns
// max(0.5 + sum, 0): clamped below only, like INRIA.
glm::vec3 evalSH(int degree, const glm::vec3* coeffs, const glm::vec3& dir);

// Why a gaussian is not drawn (0 = it is drawn). Same values as the GPU (Splat.cullReason).
// LowOpacity (opacity < 1/255: alpha can never reach the 1/255 cutoff) is only used by the
// Phase 3 GPU culling; the reference simply never blends such splats.
enum class CullReason : uint32_t { Visible = 0, NearPlane = 1, SingularCov = 2, OffScreen = 3, LowOpacity = 4 };
inline constexpr uint32_t kCullReasonCount = 5;

// Output of the preprocess for one gaussian. This is also the std430 layout of the GPU
// per-splat buffer (struct Splat in shaders/splat_common.glsl), 48 bytes.
struct ProjectedSplat {
    glm::vec2 center{0.0f};   // pixel coordinates of the projected mean
    float depth = 0.0f;       // INRIA view-space z (> 0.2 if visible)
    uint32_t cullReason = 0;  // CullReason
    glm::vec4 conicOpacity{0.0f};  // conic (a, b, c) of [[a b][b c]] and opacity
    glm::vec3 color{0.0f};    // SH colour, >= 0 (no upper clamp)
    float radius = 0.0f;      // pixels; 0 when culled (INRIA's radii[] convention)
};
static_assert(sizeof(ProjectedSplat) == 48, "must match struct Splat in shaders/splat_common.glsl");

// The whole INRIA preprocessCUDA for one gaussian.
//   posAlpha: mean (data space) + activated opacity, sigma: activated scales,
//   rotation: unit quaternion (w,x,y,z), shCoeffs: K RGB triplets, shDegree: bands to use.
ProjectedSplat preprocessGaussian(const glm::vec4& posAlpha, const glm::vec3& sigma, const glm::vec4& rotation,
                                  const glm::vec3* shCoeffs, int shDegree, const SplatCamera& cam);

// Phase 6 StopThePop: per-gaussian parameters of the depth of its maximum along a camera ray
// t (vx, vy, 1): P = Sigma^-1 in INRIA camera space, b = P mu, t* = (v . b) / (v^T P v).
// Same expressions as writeRayDepth() in shaders/splat_preprocess.comp.
struct RayDepthParams {
    float p00 = 0, p01 = 0, p02 = 0, p11 = 0, p12 = 0, p22 = 0, b0 = 0, b1 = 0, b2 = 0;
    float depthAt(float vx, float vy) const {
        const float vPv = p00 * vx * vx + 2.0f * p01 * vx * vy + 2.0f * p02 * vx + p11 * vy * vy + 2.0f * p12 * vy + p22;
        const float vb = b0 * vx + b1 * vy + b2;
        return vb / vPv;
    }
};
RayDepthParams computeRayDepthParams(const glm::vec3& sigma, const glm::vec4& rotation, const glm::vec3& tCam,
                                     const glm::mat4& viewInria);

// Opacity of a projected splat at a pixel (renderCUDA inner loop), or 0 if INRIA would
// skip it (power > 0 or alpha < 1/255). Inline: the reference renderer calls it per pixel.
inline float splatAlphaAt(const ProjectedSplat& s, float px, float py) {
    const float dx = s.center.x - px;
    const float dy = s.center.y - py;
    const glm::vec4& co = s.conicOpacity;
    const float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
    if (power > 0.0f) return 0.0f;
    const float alpha = std::fmin(kMaxAlpha, co.w * std::exp(power));
    if (alpha < kMinAlpha) return 0.0f;
    return alpha;
}

// Which pixels a splat may touch. INRIA evaluates a gaussian on every pixel of every
// 16x16 tile in its TileRect ("tile"); "radius" is the tighter square of half-size
// `radius` around the centre. The GPU quad extent follows the same choice.
enum class CoverageBound : uint32_t { Tile = 0, Radius = 1 };
const char* coverageBoundName(CoverageBound b);

// Radius bound, written as the rasterizer sees our quad: pixel centre (px + 0.5 in
// framebuffer coordinates) inside [c + 0.5 - r, c + 0.5 + r), left/top edges inclusive.
inline bool insideRadiusSquare(const ProjectedSplat& s, float px, float py) {
    const float x = px + 0.5f, y = py + 0.5f;
    const float cx = s.center.x + 0.5f, cy = s.center.y + 0.5f;
    return x >= cx - s.radius && x < cx + s.radius && y >= cy - s.radius && y < cy + s.radius;
}

}  // namespace gsr
