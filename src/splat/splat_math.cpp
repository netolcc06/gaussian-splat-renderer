#include "splat/splat_math.h"

#include "scene/gaussian_scene.h"  // kShC0

#include <algorithm>

namespace gsr {

SplatCamera makeSplatCamera(const glm::mat4& viewInria, uint32_t width, uint32_t height, float tanHalfFovX,
                            float tanHalfFovY) {
    SplatCamera cam;
    cam.viewInria = viewInria;
    // The camera centre c satisfies viewInria * c = 0, i.e. c = -R^T t for a rigid transform.
    const glm::mat3 R(viewInria);
    const glm::vec3 t(viewInria[3]);
    cam.position = -(glm::transpose(R) * t);
    cam.width = width;
    cam.height = height;
    cam.tanHalfFovX = tanHalfFovX;
    cam.tanHalfFovY = tanHalfFovY;
    // INRIA: focal = W / (2 tan(fovx / 2)); with square pixels focalX == focalY.
    cam.focalX = float(width) / (2.0f * tanHalfFovX);
    cam.focalY = float(height) / (2.0f * tanHalfFovY);
    cam.tilesX = (width + kTileSize - 1) / kTileSize;
    cam.tilesY = (height + kTileSize - 1) / kTileSize;
    return cam;
}

SplatCamera splatCameraFromViewer(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model,
                                  uint32_t width, uint32_t height) {
    // Our eye space looks down -z with +y up; INRIA's camera looks down +z with +y down.
    // The two differ by a 180 degree rotation about x: (x, y, z) -> (x, -y, -z).
    glm::mat4 flip(1.0f);
    flip[1][1] = -1.0f;
    flip[2][2] = -1.0f;
    const glm::mat4 viewInria = flip * view * model;
    // Camera::projection: proj[0][0] = 1 / (aspect tan(fovy/2)) = 1 / tan(fovx/2) and
    // proj[1][1] = -1 / tan(fovy/2) (negative because of the Vulkan y flip).
    const float tanHalfFovX = 1.0f / proj[0][0];
    const float tanHalfFovY = 1.0f / std::fabs(proj[1][1]);
    return makeSplatCamera(viewInria, width, height, tanHalfFovX, tanHalfFovY);
}

glm::mat3 quaternionToMatrix(const glm::vec4& q) {
    const float r = q.x, x = q.y, y = q.z, z = q.w;  // q = (w, x, y, z)
    // Written as rows of the rotation matrix; GLM is column-major (m[col][row]), so we
    // fill the columns with the transposed values.
    glm::mat3 R;
    R[0] = {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y)};
    R[1] = {2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + r * x)};
    R[2] = {2.0f * (x * z + r * y), 2.0f * (y * z - r * x), 1.0f - 2.0f * (x * x + y * y)};
    return R;
}

glm::mat3 computeCov3D(const glm::vec3& sigma, const glm::vec4& q) {
    // M = R S scales the columns of R; Sigma = M M^T = R S S^T R^T.
    const glm::mat3 R = quaternionToMatrix(q);
    glm::mat3 M;
    M[0] = R[0] * sigma.x;
    M[1] = R[1] * sigma.y;
    M[2] = R[2] * sigma.z;
    return M * glm::transpose(M);
}

glm::vec3 computeCov2D(const glm::vec3& tCam, const glm::mat3& cov3D, const SplatCamera& cam) {
    // The Jacobian of the perspective projection is evaluated at a point whose x/z and
    // y/z are clamped to 1.3x the half-FOV: gaussians far outside the view would
    // otherwise get huge, badly conditioned footprints.
    const float limX = kFrustumClampFactor * cam.tanHalfFovX;
    const float limY = kFrustumClampFactor * cam.tanHalfFovY;
    const float txtz = tCam.x / tCam.z;
    const float tytz = tCam.y / tCam.z;
    const float tx = std::min(limX, std::max(-limX, txtz)) * tCam.z;
    const float ty = std::min(limY, std::max(-limY, tytz)) * tCam.z;
    const float tz = tCam.z;

    // J = d(pixel)/d(camera point), 2x3. glm::mat3x2 = 3 columns, 2 rows.
    glm::mat3x2 J(0.0f);
    J[0][0] = cam.focalX / tz;                   // d px / d x
    J[2][0] = -(cam.focalX * tx) / (tz * tz);    // d px / d z
    J[1][1] = cam.focalY / tz;                   // d py / d y
    J[2][1] = -(cam.focalY * ty) / (tz * tz);    // d py / d z

    // W = rotation part of the view transform. T = J W maps data-space offsets to pixels.
    const glm::mat3 W(cam.viewInria);
    const glm::mat3x2 T = J * W;
    const glm::mat2 cov = T * cov3D * glm::transpose(T);

    // Low-pass filter: every splat is at least ~1 pixel wide (EWA "screen-space prefilter").
    return {cov[0][0] + kLowPassFilter, cov[0][1], cov[1][1] + kLowPassFilter};
}

glm::vec2 projectToPixel(const glm::vec3& tCam, const SplatCamera& cam) {
    // INRIA: p_proj = projmatrix * p (NDC, symmetric frustum) then
    // ndc2Pix(v, S) = ((v + 1) * S - 1) / 2. NDC x = x / (z tan(fovx/2)).
    // (CUDA divides by w + 1e-7; with z > 0.2 that changes nothing measurable.)
    const float ndcX = tCam.x / (tCam.z * cam.tanHalfFovX);
    const float ndcY = tCam.y / (tCam.z * cam.tanHalfFovY);
    return {((ndcX + 1.0f) * float(cam.width) - 1.0f) * 0.5f, ((ndcY + 1.0f) * float(cam.height) - 1.0f) * 0.5f};
}

bool conicAndRadius(const glm::vec3& cov2D, glm::vec3& conic, float& radius) {
    const float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det == 0.0f) return false;
    const float detInv = 1.0f / det;
    conic = {cov2D.z * detInv, -cov2D.y * detInv, cov2D.x * detInv};
    // Eigenvalues of [[a b][b c]]: mid +- sqrt(mid^2 - det). INRIA guards the sqrt with
    // max(0.1, .) and takes the larger one (lambda1, since the root is >= 0).
    const float mid = 0.5f * (cov2D.x + cov2D.z);
    const float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - det));
    const float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - det));
    radius = std::ceil(3.0f * std::sqrt(std::max(lambda1, lambda2)));
    return true;
}

TileRect computeTileRect(const glm::vec2& p, float radius, uint32_t tilesX, uint32_t tilesY) {
    // Literal port of getRect (auxiliary.h). The (int) casts truncate toward zero; together
    // with the clamps this equals floor() for the minimum and ceil() for the maximum.
    const int gx = int(tilesX), gy = int(tilesY);
    const int r = int(radius);
    TileRect t;
    t.minX = std::min(gx, std::max(0, int((p.x - float(r)) / float(kTileSize))));
    t.minY = std::min(gy, std::max(0, int((p.y - float(r)) / float(kTileSize))));
    t.maxX = std::min(gx, std::max(0, int((p.x + float(r) + float(kTileSize - 1)) / float(kTileSize))));
    t.maxY = std::min(gy, std::max(0, int((p.y + float(r) + float(kTileSize - 1)) / float(kTileSize))));
    return t;
}

glm::vec3 evalSH(int degree, const glm::vec3* sh, const glm::vec3& dir) {
    glm::vec3 result = kShC0 * sh[0];
    if (degree > 0) {
        const float x = dir.x, y = dir.y, z = dir.z;
        result = result - kShC1 * y * sh[1] + kShC1 * z * sh[2] - kShC1 * x * sh[3];
        if (degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            result = result + kShC2[0] * xy * sh[4] + kShC2[1] * yz * sh[5] +
                     kShC2[2] * (2.0f * zz - xx - yy) * sh[6] + kShC2[3] * xz * sh[7] +
                     kShC2[4] * (xx - yy) * sh[8];
            if (degree > 2) {
                result = result + kShC3[0] * y * (3.0f * xx - yy) * sh[9] + kShC3[1] * xy * z * sh[10] +
                         kShC3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         kShC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         kShC3[4] * x * (4.0f * zz - xx - yy) * sh[13] + kShC3[5] * z * (xx - yy) * sh[14] +
                         kShC3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;
    return glm::max(result, glm::vec3(0.0f));
}

ProjectedSplat preprocessGaussian(const glm::vec4& posAlpha, const glm::vec3& sigma, const glm::vec4& rotation,
                                  const glm::vec3* shCoeffs, int shDegree, const SplatCamera& cam) {
    ProjectedSplat s;
    const glm::vec3 mean(posAlpha);

    // 1) Near-plane cull (in_frustum). INRIA has no other frustum test here.
    const glm::vec3 tCam = transformPoint(cam.viewInria, mean);
    s.depth = tCam.z;
    if (tCam.z <= kNearCullDepth) {
        s.cullReason = uint32_t(CullReason::NearPlane);
        return s;
    }

    // 2) 3D covariance -> 2D covariance (EWA) -> conic + radius.
    const glm::mat3 cov3D = computeCov3D(sigma, rotation);
    const glm::vec3 cov2D = computeCov2D(tCam, cov3D, cam);
    glm::vec3 conic;
    float radius = 0.0f;
    if (!conicAndRadius(cov2D, conic, radius)) {
        s.cullReason = uint32_t(CullReason::SingularCov);
        return s;
    }

    // 3) Screen position and touched tiles; nothing on screen -> culled.
    const glm::vec2 center = projectToPixel(tCam, cam);
    if (computeTileRect(center, radius, cam.tilesX, cam.tilesY).empty()) {
        s.center = center;
        s.cullReason = uint32_t(CullReason::OffScreen);
        return s;
    }

    // 4) View-dependent colour, direction from the camera to the mean (data space).
    const glm::vec3 dir = glm::normalize(mean - cam.position);
    s.color = evalSH(shDegree, shCoeffs, dir);

    s.center = center;
    s.conicOpacity = glm::vec4(conic, posAlpha.w);
    s.radius = radius;
    s.cullReason = uint32_t(CullReason::Visible);
    return s;
}

RayDepthParams computeRayDepthParams(const glm::vec3& sigma, const glm::vec4& rotation, const glm::vec3& tCam,
                                     const glm::mat4& viewInria) {
    const glm::mat3 M = glm::mat3(viewInria) * quaternionToMatrix(rotation);  // gaussian axes, camera space
    const float smax = std::max(sigma.x, std::max(sigma.y, sigma.z));
    const glm::vec3 s = glm::max(sigma, glm::vec3(std::max(1e-4f * smax, 1e-12f)));
    const glm::vec3 w = 1.0f / (s * s);
    glm::mat3 P(0.0f);
    for (int k = 0; k < 3; ++k) P += w[k] * glm::outerProduct(M[k], M[k]);
    const glm::vec3 b = P * tCam;
    return {P[0][0], P[0][1], P[0][2], P[1][1], P[1][2], P[2][2], b.x, b.y, b.z};
}

const char* coverageBoundName(CoverageBound b) { return b == CoverageBound::Tile ? "tile" : "radius"; }

}  // namespace gsr
