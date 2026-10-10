// Shared splatting math: a line-by-line GLSL mirror of src/splat/splat_math.cpp, which
// itself ports the INRIA forward pass (diff-gaussian-rasterization, forward.cu).
// Keep the two files in sync. Conventions (also in splat_math.h):
//   INRIA camera space: x right, y down, z forward. Pixel coordinates: pixel (i, j) has
//   coordinates (i, j) = gl_FragCoord.xy - 0.5.

#ifndef SPLAT_COMMON_GLSL
#define SPLAT_COMMON_GLSL

const float kNearCullDepth = 0.2;
const float kLowPassFilter = 0.3;
const float kFrustumClampFactor = 1.3;
const float kMaxAlpha = 0.99;
const float kMinAlpha = 1.0 / 255.0;
const int kTileSize = 16;

const uint kCullVisible = 0u;
const uint kCullNearPlane = 1u;
const uint kCullSingularCov = 2u;
const uint kCullOffScreen = 3u;
const uint kCullLowOpacity = 4u;  // Phase 3 (splat-gpusort): opacity < 1/255, never visible
const uint kCullReasonCount = 5u;

// Per-splat output of the preprocess. 48 bytes in std430; C++ twin: gsr::ProjectedSplat.
struct Splat {
    vec2 center;        // pixel coordinates of the projected mean
    float depth;        // INRIA view-space z
    uint cullReason;    // kCull*; 0 = drawn
    vec4 conicOpacity;  // inverse 2D covariance (a, b, c) of [[a b][b c]], opacity
    vec3 color;         // SH colour, >= 0
    float radius;       // ceil(3 sqrt(lambda_max)) in pixels, 0 if culled
};

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[](1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                               -1.0925484305920792, 0.5462742152960396);
const float SH_C3[7] = float[](-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                               0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                               -0.5900435899266435);

// Unit quaternion q = (w, x, y, z) in .xyzw -> rotation matrix (GLSL is column-major).
mat3 quaternionToMatrix(vec4 q) {
    float r = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y),   // column 0
                2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x),   // column 1
                2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y));  // column 2
}

// Sigma = R S S^T R^T.
mat3 computeCov3D(vec3 sigma, vec4 q) {
    mat3 R = quaternionToMatrix(q);
    mat3 M = mat3(R[0] * sigma.x, R[1] * sigma.y, R[2] * sigma.z);
    return M * transpose(M);
}

// EWA: J W Sigma W^T J^T + 0.3 I, packed as (a, b, c) of [[a b][b c]].
vec3 computeCov2D(vec3 t, mat3 cov3D, mat3 W, vec2 focal, vec2 tanHalfFov) {
    vec2 lim = kFrustumClampFactor * tanHalfFov;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    float tx = min(lim.x, max(-lim.x, txtz)) * t.z;
    float ty = min(lim.y, max(-lim.y, tytz)) * t.z;
    float tz = t.z;

    mat3x2 J = mat3x2(0.0);  // 3 columns, 2 rows
    J[0][0] = focal.x / tz;
    J[2][0] = -(focal.x * tx) / (tz * tz);
    J[1][1] = focal.y / tz;
    J[2][1] = -(focal.y * ty) / (tz * tz);

    mat3x2 T = J * W;
    mat2 cov = T * cov3D * transpose(T);
    return vec3(cov[0][0] + kLowPassFilter, cov[0][1], cov[1][1] + kLowPassFilter);
}

vec2 projectToPixel(vec3 t, uvec2 imageSize, vec2 tanHalfFov) {
    vec2 ndc = t.xy / (t.z * tanHalfFov);
    return ((ndc + 1.0) * vec2(imageSize) - 1.0) * 0.5;
}

// Returns false for a singular covariance. radius = ceil(3 sqrt(lambda_max)).
bool conicAndRadius(vec3 cov, out vec3 conic, out float radius) {
    float det = cov.x * cov.z - cov.y * cov.y;
    conic = vec3(0.0);
    radius = 0.0;
    if (det == 0.0) return false;
    float detInv = 1.0 / det;
    conic = vec3(cov.z * detInv, -cov.y * detInv, cov.x * detInv);
    float mid = 0.5 * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    return true;
}

// INRIA getRect: touched tiles [min, max) clamped to the grid; int() truncates like CUDA.
ivec4 computeTileRect(vec2 p, float radius, uvec2 tiles) {
    ivec2 grid = ivec2(tiles);
    float r = float(int(radius));
    ivec2 lo = min(grid, max(ivec2(0), ivec2(int((p.x - r) / float(kTileSize)), int((p.y - r) / float(kTileSize)))));
    ivec2 hi = min(grid, max(ivec2(0), ivec2(int((p.x + r + float(kTileSize - 1)) / float(kTileSize)),
                                             int((p.y + r + float(kTileSize - 1)) / float(kTileSize)))));
    return ivec4(lo, hi);
}

// computeColorFromSH. sh[k] = RGB of coefficient k; dir = normalize(mean - camPos).
vec3 evalSH(uint degree, vec3 sh[16], vec3 dir) {
    vec3 result = SH_C0 * sh[0];
    if (degree > 0u) {
        float x = dir.x, y = dir.y, z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (degree > 1u) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0 * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];
            if (degree > 2u) {
                result = result + SH_C3[0] * y * (3.0 * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] +
                         SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0 * yy) * sh[15];
            }
        }
    }
    result += 0.5;
    return max(result, vec3(0.0));
}

#endif  // SPLAT_COMMON_GLSL
