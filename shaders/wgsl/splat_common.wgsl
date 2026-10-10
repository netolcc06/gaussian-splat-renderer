// Port of shaders/splat_common.glsl (web phase W2): the per-gaussian INRIA math, a line-by-line
// mirror of src/splat/splat_math.cpp. Conventions as there: INRIA camera space (x right, y down,
// z forward); pixel (i, j) has coordinates (i, j). Keep the three files in sync
// (shaders/wgsl/PORTS.md).
//
// Differences from the GLSL: no `out` parameters (conicAndRadius returns a struct), matrices are
// built from columns explicitly, and there is no `precise` in WGSL. The splat-tiles port never
// recomputes a float result in two pipelines and expects the same bits (coverage spans are stored
// as integers, DESIGN.md section 3), so FMA contraction only moves results by an ulp.

const kNearCullDepth: f32 = 0.2;
const kLowPassFilter: f32 = 0.3;
const kFrustumClampFactor: f32 = 1.3;
const kMaxAlpha: f32 = 0.99;
const kMinAlpha: f32 = 1.0 / 255.0;
const kTileSize: i32 = 16;  // INRIA's getRect tile (BLOCK_X = BLOCK_Y)

const kCullVisible: u32 = 0u;
const kCullNearPlane: u32 = 1u;
const kCullSingularCov: u32 = 2u;
const kCullOffScreen: u32 = 3u;
const kCullLowOpacity: u32 = 4u;  // opacity < 1/255: never visible
const kCullReasonCount: u32 = 5u;

// Per-splat output of the preprocess. 48 bytes; C++ twin: gsr::ProjectedSplat (splat_math.h).
struct Splat {
    center: vec2<f32>,        // pixel coordinates of the projected mean
    depth: f32,               // INRIA view-space z
    cullReason: u32,          // kCull*; 0 = drawn
    conicOpacity: vec4<f32>,  // inverse 2D covariance (a, b, c) of [[a b][b c]], opacity
    color: vec3<f32>,         // SH colour, >= 0
    radius: f32,              // ceil(3 sqrt(lambda_max)) in pixels, 0 if culled
}

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
                            0.5462742152960396);
const SH_C3 = array<f32, 7>(-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                            -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

// Unit quaternion q = (w, x, y, z) in .xyzw -> rotation matrix (columns, as GLSL).
fn quaternionToMatrix(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),   // column 0
        vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),   // column 1
        vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)));  // column 2
}

// Sigma = R S S^T R^T.
fn computeCov3D(sigma: vec3<f32>, q: vec4<f32>) -> mat3x3<f32> {
    let R = quaternionToMatrix(q);
    let M = mat3x3<f32>(R[0] * sigma.x, R[1] * sigma.y, R[2] * sigma.z);
    return M * transpose(M);
}

// EWA: J W Sigma W^T J^T + 0.3 I, packed as (a, b, c) of [[a b][b c]].
fn computeCov2D(t: vec3<f32>, cov3D: mat3x3<f32>, W: mat3x3<f32>, focal: vec2<f32>, tanHalfFov: vec2<f32>) -> vec3<f32> {
    let lim = kFrustumClampFactor * tanHalfFov;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    let tx = min(lim.x, max(-lim.x, txtz)) * t.z;
    let ty = min(lim.y, max(-lim.y, tytz)) * t.z;
    let tz = t.z;

    // 3 columns, 2 rows.
    let J = mat3x2<f32>(vec2<f32>(focal.x / tz, 0.0), vec2<f32>(0.0, focal.y / tz),
                        vec2<f32>(-(focal.x * tx) / (tz * tz), -(focal.y * ty) / (tz * tz)));
    let T = J * W;
    let cov = T * cov3D * transpose(T);
    return vec3<f32>(cov[0][0] + kLowPassFilter, cov[0][1], cov[1][1] + kLowPassFilter);
}

fn projectToPixel(t: vec3<f32>, imageSize: vec2<u32>, tanHalfFov: vec2<f32>) -> vec2<f32> {
    let ndc = t.xy / (t.z * tanHalfFov);
    return ((ndc + 1.0) * vec2<f32>(imageSize) - 1.0) * 0.5;
}

struct ConicRadius {
    ok: bool,  // false: singular covariance
    conic: vec3<f32>,
    radius: f32,  // ceil(3 sqrt(lambda_max))
}

fn conicAndRadius(cov: vec3<f32>) -> ConicRadius {
    var r: ConicRadius;
    r.ok = false;
    r.conic = vec3<f32>(0.0);
    r.radius = 0.0;
    let det = cov.x * cov.z - cov.y * cov.y;
    if (det == 0.0) {
        return r;
    }
    let detInv = 1.0 / det;
    r.conic = vec3<f32>(cov.z * detInv, -cov.y * detInv, cov.x * detInv);
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    r.radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    r.ok = true;
    return r;
}

// INRIA getRect: touched tiles [min, max) clamped to the grid; i32() truncates toward zero like
// CUDA's int().
fn computeTileRect(p: vec2<f32>, radius: f32, tiles: vec2<u32>) -> vec4<i32> {
    let grid = vec2<i32>(tiles);
    let r = f32(i32(radius));
    let S = f32(kTileSize);
    let lo = min(grid, max(vec2<i32>(0), vec2<i32>(i32((p.x - r) / S), i32((p.y - r) / S))));
    let hi = min(grid, max(vec2<i32>(0), vec2<i32>(i32((p.x + r + (S - 1.0)) / S), i32((p.y + r + (S - 1.0)) / S))));
    return vec4<i32>(lo, hi);
}

// computeColorFromSH. sh[k] = RGB of coefficient k; dir = normalize(mean - camPos).
fn evalSH(degree: u32, sh: array<vec3<f32>, 16>, dir: vec3<f32>) -> vec3<f32> {
    var result = SH_C0 * sh[0];
    if (degree > 0u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (degree > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;
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
    return max(result, vec3<f32>(0.0));
}
