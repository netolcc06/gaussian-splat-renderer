// Tests for the splatting math (src/splat/splat_math.*), the CPU depth sort and the CPU
// reference renderer. Run through CTest (`splat_math`).
//
// The expected values are derived independently where possible:
//   - covariance: from glm::rotate (axis-angle) instead of the quaternion formula,
//   - EWA projection: from a NUMERICAL Jacobian of the pinhole projection (central
//     differences in double precision), so a sign/transpose slip in J or W shows up,
//   - SH: closed-form values for single basis functions,
//   - compositing: hand-written front-to-back and back-to-front sums.
#include "scene/gaussian_scene.h"
#include "splat/depth_sort.h"
#include "splat/reference_renderer.h"
#include "splat/splat_math.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace gsr;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

bool near(double a, double b, double relTol, double absTol = 0.0) {
    return std::fabs(a - b) <= absTol + relTol * std::max(std::fabs(a), std::fabs(b));
}

glm::vec4 quatAxisAngle(glm::vec3 axis, float angle) {  // (w, x, y, z)
    axis = glm::normalize(axis);
    return {std::cos(angle / 2), axis.x * std::sin(angle / 2), axis.y * std::sin(angle / 2),
            axis.z * std::sin(angle / 2)};
}

// A camera at the origin looking down +z (INRIA convention), W x H, focal f.
SplatCamera axisCamera(uint32_t w, uint32_t h, float f) {
    return makeSplatCamera(glm::mat4(1.0f), w, h, float(w) / (2.0f * f), float(h) / (2.0f * f));
}

// --- Covariance ----------------------------------------------------------------------
void testCov3D() {
    std::printf("cov3D\n");
    // Identity rotation: Sigma = diag(sigma^2).
    const glm::mat3 a = computeCov3D({2.0f, 1.0f, 0.5f}, {1, 0, 0, 0});
    CHECK(near(a[0][0], 4.0, 1e-6) && near(a[1][1], 1.0, 1e-6) && near(a[2][2], 0.25, 1e-6), "diag");
    CHECK(std::fabs(a[0][1]) < 1e-6f && std::fabs(a[0][2]) < 1e-6f && std::fabs(a[1][2]) < 1e-6f, "off-diagonal");

    // 90 degrees about z maps local x to world y: the variances of x and y swap.
    const glm::mat3 b = computeCov3D({2.0f, 1.0f, 0.5f}, quatAxisAngle({0, 0, 1}, glm::half_pi<float>()));
    CHECK(near(b[0][0], 1.0, 1e-5) && near(b[1][1], 4.0, 1e-5) && near(b[2][2], 0.25, 1e-5), "rotated diag %f %f",
          b[0][0], b[1][1]);

    // Arbitrary rotation vs glm::rotate (independent axis-angle formula).
    const glm::vec3 axis = glm::normalize(glm::vec3(0.3f, -0.8f, 0.5f));
    const float angle = 1.1f;
    const glm::vec3 sigma(0.7f, 0.2f, 0.05f);
    const glm::mat3 R(glm::rotate(glm::mat4(1.0f), angle, axis));
    const glm::mat3 S2(glm::vec3(sigma.x * sigma.x, 0, 0), glm::vec3(0, sigma.y * sigma.y, 0),
                       glm::vec3(0, 0, sigma.z * sigma.z));
    const glm::mat3 expected = R * S2 * glm::transpose(R);
    const glm::mat3 got = computeCov3D(sigma, quatAxisAngle(axis, angle));
    double worst = 0.0;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) worst = std::max(worst, double(std::fabs(got[c][r] - expected[c][r])));
    CHECK(worst < 1e-6, "vs glm::rotate, max diff %g", worst);
    CHECK(got[0][1] == got[1][0] && got[0][2] == got[2][0] && got[1][2] == got[2][1], "symmetric");
}

// --- EWA projection, conic, radius --------------------------------------------------
void testCov2DOnAxis() {
    std::printf("cov2D on axis\n");
    // Isotropic sigma 0.1 at depth 5, focal 500: J = diag(f/z, f/z) = 100 px per unit, so
    // cov2D = (100 * 0.1)^2 + 0.3 = 100.3 on the diagonal.
    const SplatCamera cam = axisCamera(1000, 800, 500.0f);
    const glm::vec3 t(0.0f, 0.0f, 5.0f);
    const glm::vec3 cov = computeCov2D(t, computeCov3D(glm::vec3(0.1f), {1, 0, 0, 0}), cam);
    CHECK(near(cov.x, 100.3, 1e-5) && near(cov.z, 100.3, 1e-5) && std::fabs(cov.y) < 1e-6f, "cov %f %f %f", cov.x,
          cov.y, cov.z);
    glm::vec3 conic;
    float radius = 0.0f;
    CHECK(conicAndRadius(cov, conic, radius), "invertible");
    CHECK(near(conic.x, 1.0 / 100.3, 1e-5) && near(conic.z, 1.0 / 100.3, 1e-5) && conic.y == 0.0f, "conic");
    // lambda = 100.3 + sqrt(max(0.1, 0)) = 100.616; radius = ceil(3 sqrt(100.616)) = ceil(30.09) = 31.
    CHECK(radius == 31.0f, "radius %f", radius);
    // Pixel centre: principal point ((W-1)/2, (H-1)/2) in INRIA pixel coordinates.
    const glm::vec2 c = projectToPixel(t, cam);
    CHECK(near(c.x, 499.5, 1e-6) && near(c.y, 399.5, 1e-6), "centre %f %f", c.x, c.y);
}

// Pinhole projection in pixels (INRIA convention) of a camera-space point, in double.
glm::dvec2 projectD(const glm::dvec3& p, const SplatCamera& cam) {
    return {cam.focalX * p.x / p.z + (cam.width - 1) * 0.5, cam.focalY * p.y / p.z + (cam.height - 1) * 0.5};
}

void testCov2DNumericalJacobian() {
    std::printf("cov2D vs numerical Jacobian\n");
    // A rotated, translated camera and an anisotropic gaussian well inside the view
    // (no clamping): cov2D must equal Jn W Sigma W^T Jn^T + 0.3 I with Jn = d(pixel)/d(cam).
    glm::mat4 view = glm::rotate(glm::mat4(1.0f), 0.4f, glm::normalize(glm::vec3(0.2f, 1.0f, 0.1f)));
    view = glm::translate(view, glm::vec3(0.3f, -0.2f, 0.5f));
    const SplatCamera cam = makeSplatCamera(view, 640, 480, 0.6f, 0.45f);
    // Pick the camera-space point, then map it back to data space.
    const glm::vec3 mean = glm::vec3(glm::inverse(cam.viewInria) * glm::vec4(0.9f, -0.5f, 3.0f, 1.0f));
    const glm::vec3 t = transformPoint(cam.viewInria, mean);
    CHECK(t.z > 1.0f && std::fabs(t.x / t.z) < cam.tanHalfFovX && std::fabs(t.y / t.z) < cam.tanHalfFovY,
          "test point inside the view (t = %f %f %f)", t.x, t.y, t.z);
    const glm::mat3 sigma3 = computeCov3D({0.3f, 0.08f, 0.02f}, quatAxisAngle({1, 2, 3}, 0.7f));

    glm::dmat3x2 Jn(0.0);  // 3 columns (d/dx, d/dy, d/dz), 2 rows (px, py)
    const double h = 1e-5;
    for (int k = 0; k < 3; ++k) {
        glm::dvec3 dp(0.0);
        dp[k] = h;
        Jn[k] = (projectD(glm::dvec3(t) + dp, cam) - projectD(glm::dvec3(t) - dp, cam)) / (2.0 * h);
    }
    const glm::dmat3 W(glm::mat3(cam.viewInria));
    const glm::dmat2 expected = (Jn * W) * glm::dmat3(sigma3) * glm::transpose(Jn * W);
    const glm::vec3 got = computeCov2D(t, sigma3, cam);
    CHECK(near(got.x, expected[0][0] + 0.3, 1e-4), "a %f vs %f", got.x, expected[0][0] + 0.3);
    CHECK(near(got.y, expected[0][1], 1e-4, 1e-4), "b %f vs %f", got.y, expected[0][1]);
    CHECK(near(got.z, expected[1][1] + 0.3, 1e-4), "c %f vs %f", got.z, expected[1][1] + 0.3);
    // The projected centre matches the double-precision pinhole too.
    const glm::vec2 c = projectToPixel(t, cam);
    const glm::dvec2 ce = projectD(glm::dvec3(t), cam);
    CHECK(near(c.x, ce.x, 1e-5, 1e-3) && near(c.y, ce.y, 1e-5, 1e-3), "centre %f %f vs %f %f", c.x, c.y, ce.x, ce.y);

    // Conic is the inverse: cov * conic = I.
    glm::vec3 conic;
    float radius;
    CHECK(conicAndRadius(got, conic, radius), "invertible");
    const glm::mat2 m(got.x, got.y, got.y, got.z), inv(conic.x, conic.y, conic.y, conic.z);
    const glm::mat2 I = m * inv;
    CHECK(near(I[0][0], 1, 1e-4) && near(I[1][1], 1, 1e-4) && std::fabs(I[0][1]) < 1e-4f, "cov * conic = I");
    // Radius covers 3 sigma of the major axis: lambda_max from the closed form.
    const double mid = 0.5 * (got.x + got.z), det = double(got.x) * got.z - double(got.y) * got.y;
    const double lmax = mid + std::sqrt(std::max(0.1, mid * mid - det));
    CHECK(radius == float(std::ceil(3.0 * std::sqrt(lmax))), "radius %f vs %f", radius, std::ceil(3.0 * std::sqrt(lmax)));
}

void testCov2DClamp() {
    std::printf("cov2D frustum clamp\n");
    // A gaussian far left of the view: J is evaluated at x/z = -1.3 tan(fovx/2), but the
    // centre itself is NOT clamped (it projects off screen).
    const SplatCamera cam = axisCamera(400, 300, 200.0f);  // tanHalfFovX = 1.0
    const glm::vec3 t(-10.0f, 0.0f, 2.0f);                   // x/z = -5
    const glm::mat3 sigma3 = computeCov3D({0.2f, 0.1f, 0.3f}, {1, 0, 0, 0});
    const glm::vec3 got = computeCov2D(t, sigma3, cam);
    const double tx = -1.3 * 1.0 * 2.0, tz = 2.0, f = 200.0;
    // J row 0 = (f/z, 0, -f tx / z^2); Sigma diagonal -> a = (f/z)^2 sx^2 + (f tx/z^2)^2 sz^2.
    const double a = (f / tz) * (f / tz) * 0.04 + (f * tx / (tz * tz)) * (f * tx / (tz * tz)) * 0.09 + 0.3;
    CHECK(near(got.x, a, 1e-5), "clamped a %f vs %f", got.x, a);
    CHECK(near(got.z, (f / tz) * (f / tz) * 0.01 + 0.3, 1e-5), "c %f", got.z);
    const glm::vec2 c = projectToPixel(t, cam);
    CHECK(near(c.x, 200.0 * -5.0 + 199.5, 1e-5), "unclamped centre x %f", c.x);
}

// --- Tile rect and the preprocess culls ---------------------------------------------
void testTileRectAndCulls() {
    std::printf("tile rect and culls\n");
    TileRect r = computeTileRect({100.5f, 40.0f}, 10.0f, 10, 10);
    // x: int(90.5/16)=5, int((110.5+15)/16)=int(7.84)=7; y: int(30/16)=1, int(65/16)=4.
    CHECK(r.minX == 5 && r.maxX == 7 && r.minY == 1 && r.maxY == 4, "rect %d %d %d %d", r.minX, r.maxX, r.minY,
          r.maxY);
    r = computeTileRect({-50.0f, 10.0f}, 10.0f, 10, 10);  // entirely left of the screen
    CHECK(r.empty(), "off-screen rect empty");
    r = computeTileRect({1000.0f, 10.0f}, 30.0f, 10, 10);  // right of a 160 px wide screen
    CHECK(r.empty(), "off-screen right");
    r = computeTileRect({5.0f, 5.0f}, 100.0f, 10, 10);  // larger than the screen: clamped
    CHECK(r.minX == 0 && r.maxX == 7 && r.minY == 0 && r.maxY == 7, "clamped %d %d", r.maxX, r.maxY);

    const SplatCamera cam = axisCamera(64, 64, 64.0f);
    const glm::vec3 sh[1] = {glm::vec3(0.0f)};
    auto pre = [&](glm::vec3 p) {
        return preprocessGaussian(glm::vec4(p, 0.5f), glm::vec3(0.1f), {1, 0, 0, 0}, sh, 0, cam);
    };
    CHECK(pre({0, 0, 0.2f}).cullReason == uint32_t(CullReason::NearPlane), "z = 0.2 culled");
    CHECK(pre({0, 0, -1.0f}).cullReason == uint32_t(CullReason::NearPlane), "behind culled");
    CHECK(pre({0, 0, 0.25f}).cullReason == uint32_t(CullReason::Visible), "z = 0.25 visible");
    CHECK(pre({50, 0, 1.0f}).cullReason == uint32_t(CullReason::OffScreen), "off screen culled");
    const ProjectedSplat s = pre({0, 0, 2.0f});
    CHECK(s.radius > 0.0f && s.depth == 2.0f && s.conicOpacity.w == 0.5f, "visible splat fields");
    CHECK(near(s.color.r, 0.5, 1e-6), "grey from zero SH");
}

// --- Spherical harmonics --------------------------------------------------------------
void testSH() {
    std::printf("spherical harmonics\n");
    glm::vec3 sh[16] = {};
    sh[0] = glm::vec3(1.0f, -1.0f, 0.0f);
    const glm::vec3 d0 = evalSH(3, sh, {0, 0, 1});
    CHECK(near(d0.r, 0.5 + kShC0, 1e-6) && near(d0.g, 0.5 - kShC0, 1e-6) && near(d0.b, 0.5, 1e-6), "DC");
    sh[0] = glm::vec3(-5.0f);
    CHECK(evalSH(0, sh, {0, 0, 1}) == glm::vec3(0.0f), "clamped below at 0");

    // Band 1 basis: Y1 = -C1 y, Y2 = C1 z, Y3 = -C1 x.
    const glm::vec3 dir = glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f));
    const float expect1[3] = {-kShC1 * dir.y, kShC1 * dir.z, -kShC1 * dir.x};
    for (int k = 1; k <= 3; ++k) {
        glm::vec3 c[16] = {};
        c[k] = glm::vec3(1.0f, 0.0f, 0.0f);
        CHECK(near(evalSH(1, c, dir).r, 0.5 + expect1[k - 1], 1e-6), "band 1 k=%d", k);
        CHECK(near(evalSH(0, c, dir).r, 0.5, 1e-6), "degree 0 ignores band 1 (k=%d)", k);
    }
    // Band 2: Y6 = C2[2] (2z^2 - x^2 - y^2); band 3: Y15 = C3[6] x (x^2 - 3y^2).
    glm::vec3 c6[16] = {};
    c6[6] = glm::vec3(1.0f);
    const double y6 = 0.31539156525252005 * (2 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y);
    CHECK(near(evalSH(2, c6, dir).g, 0.5 + y6, 1e-6), "band 2 k=6");
    CHECK(near(evalSH(1, c6, dir).g, 0.5, 1e-6), "degree 1 ignores band 2");
    glm::vec3 c15[16] = {};
    c15[15] = glm::vec3(1.0f);
    const double y15 = -0.5900435899266435 * dir.x * (dir.x * dir.x - 3 * dir.y * dir.y);
    CHECK(near(evalSH(3, c15, dir).b, 0.5 + y15, 1e-6), "band 3 k=15");
    // Real SH are orthonormal on the sphere: integral of Y9^2 = 1, i.e. the mean of Y9^2
    // over uniform directions is 1 / (4 pi) (3DGS normalization C3[0] = sqrt(35 / (32 pi))).
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    double acc = 0.0;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
        glm::dvec3 v(nd(rng), nd(rng), nd(rng));
        v = glm::normalize(v);
        const double y9 = -0.5900435899266435 * v.y * (3 * v.x * v.x - v.y * v.y);
        acc += y9 * y9;
    }
    const double integral = acc / n * 4.0 * 3.14159265358979;
    CHECK(near(integral, 1.0, 0.02), "integral of Y9^2 over the sphere = %f (expected 1)", integral);
}

// --- Depth sort ---------------------------------------------------------------------
void testDepthSort() {
    std::printf("depth sort\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 30.0f);
    std::vector<glm::vec4> pos(20000);
    for (size_t i = 0; i < pos.size(); ++i) {
        // Quantized depths create many exact ties, which must keep index order.
        const float z = (i % 3 == 0) ? std::floor(u(rng)) : u(rng);
        pos[i] = glm::vec4(u(rng), u(rng), z, 1.0f);
    }
    DepthSorter sorter;
    sorter.sort(pos.data(), uint32_t(pos.size()), glm::mat4(1.0f));
    std::vector<uint32_t> expected;
    for (uint32_t i = 0; i < pos.size(); ++i)
        if (pos[i].z > kNearCullDepth) expected.push_back(i);
    std::stable_sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) { return pos[a].z < pos[b].z; });
    CHECK(sorter.order() == expected, "radix order == stable_sort order (%zu vs %zu entries)", sorter.order().size(),
          expected.size());
    sorter.sort(pos.data(), 0, glm::mat4(1.0f));
    CHECK(sorter.order().empty(), "empty input");
}

// --- Reference renderer: compositing ------------------------------------------------
GaussianScene makeScene(const std::vector<glm::vec3>& means, const std::vector<glm::vec3>& colors,
                        const std::vector<float>& opacities, float sigma) {
    GaussianScene s;
    s.count = uint32_t(means.size());
    s.shDegree = 0;
    for (size_t i = 0; i < means.size(); ++i) {
        s.posAlpha.push_back(glm::vec4(means[i], opacities[i]));
        s.scales.push_back(glm::vec4(sigma, sigma, sigma, 0.0f));
        s.rotations.push_back(glm::vec4(1, 0, 0, 0));
        const glm::vec3 dc = (colors[i] - 0.5f) / kShC0;  // colour = 0.5 + C0 * dc
        s.sh.insert(s.sh.end(), {dc.r, dc.g, dc.b});
    }
    return s;
}

void testReferenceSingle() {
    std::printf("reference renderer: single gaussian\n");
    // sigma 0.25 at depth 4, focal 64: cov2D = (16 * 0.25)^2 + 0.3 = 16.3, centre (31.5, 31.5).
    const SplatCamera cam = axisCamera(64, 64, 64.0f);
    const GaussianScene scene = makeScene({{0, 0, 4}}, {{1.0f, 0.5f, 0.25f}}, {0.8f}, 0.25f);
    ReferenceOptions opt;
    opt.threads = 2;
    const ReferenceImage img = renderReference(scene, cam, opt);
    CHECK(img.stats.visible == 1, "visible");
    auto at = [&](uint32_t x, uint32_t y) {
        const float* p = &img.rgb[(size_t(y) * 64 + x) * 3];
        return glm::vec3(p[0], p[1], p[2]);
    };
    for (auto [x, y] : {std::pair{31u, 31u}, std::pair{35u, 30u}, std::pair{20u, 40u}}) {
        const double dx = 31.5 - x, dy = 31.5 - y;
        const double alpha = std::min(0.99, 0.8 * std::exp(-0.5 * (dx * dx + dy * dy) / 16.3));
        const double expected = alpha >= 1.0 / 255.0 ? alpha : 0.0;  // colour r = 1
        CHECK(near(at(x, y).r, expected, 1e-4, 1e-6), "pixel (%u,%u) r %f vs %f", x, y, at(x, y).r, expected);
        CHECK(near(at(x, y).g, 0.5 * expected, 1e-4, 1e-6), "pixel (%u,%u) g", x, y);
    }
    // Coverage bounds. radius = ceil(3 sqrt(16.3 + sqrt(0.1))) = ceil(12.23) = 13. With
    // opacity 0.99, alpha >= 1/255 out to d = sqrt(2 * 16.3 * ln(0.99 * 255)) = 13.4 px,
    // so a pixel at 13.2 px is visible under INRIA's tile bound but outside the radius
    // square. Centre x = 16 X + 31.5 = 31.8 for X = 0.01875; pixel (45, 31): dx = 13.2.
    const GaussianScene edge = makeScene({{0.01875f, 0, 4}}, {{1.0f, 1.0f, 1.0f}}, {0.99f}, 0.25f);
    opt.bound = CoverageBound::Tile;
    const ReferenceImage tileImg = renderReference(edge, cam, opt);
    opt.bound = CoverageBound::Radius;
    const ReferenceImage radiusImg = renderReference(edge, cam, opt);
    const size_t edgePixel = (size_t(31) * 64 + 45) * 3;
    const double expectEdge = 0.99 * std::exp(-0.5 * (13.2 * 13.2 + 0.5 * 0.5) / 16.3);
    CHECK(near(tileImg.rgb[edgePixel], expectEdge, 1e-3), "tile bound: pixel 13.2 px away = %f (expected %f)",
          tileImg.rgb[edgePixel], expectEdge);
    CHECK(radiusImg.rgb[edgePixel] == 0.0f, "radius bound clips it (%f)", radiusImg.rgb[edgePixel]);
    const size_t inner = (size_t(31) * 64 + 44) * 3;  // 12.2 px: inside both
    CHECK(tileImg.rgb[inner] == radiusImg.rgb[inner] && tileImg.rgb[inner] > 0.0f, "inside the square: identical");
}

void testReferenceCompositing() {
    std::printf("reference renderer: compositing and early termination\n");
    // Four co-centred splats at increasing depth, colours R, G, B, white. At the centre
    // pixel the gaussian factor is ~1, so alpha = min(0.99, opacity).
    const SplatCamera cam = axisCamera(32, 32, 32.0f);
    const std::vector<glm::vec3> means = {{0, 0, 2}, {0, 0, 3}, {0, 0, 4}, {0, 0, 5}};
    const std::vector<glm::vec3> colors = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    const GaussianScene scene = makeScene(means, colors, {0.5f, 0.6f, 0.7f, 0.8f}, 1.0f);
    ReferenceOptions opt;
    opt.threads = 1;
    opt.earlyTermination = false;
    const ReferenceImage img = renderReference(scene, cam, opt);

    // Pixel (16, 16): offset 0.5 px from the centre (15.5, 15.5) in both axes.
    const size_t p = (size_t(16) * 32 + 16) * 3;
    std::vector<ProjectedSplat> s;
    for (size_t i = 0; i < 4; ++i) {
        const glm::vec3 c[1] = {glm::vec3(scene.sh[i * 3], scene.sh[i * 3 + 1], scene.sh[i * 3 + 2])};
        s.push_back(preprocessGaussian(scene.posAlpha[i], glm::vec3(scene.scales[i]), scene.rotations[i], c, 0, cam));
    }
    // Front to back: C = sum c_i a_i T_i, T_{i+1} = T_i (1 - a_i).
    glm::dvec3 front(0.0);
    double T = 1.0;
    for (const ProjectedSplat& sp : s) {
        const double a = splatAlphaAt(sp, 16.0f, 16.0f);
        front += glm::dvec3(sp.color) * a * T;
        T *= 1.0 - a;
    }
    // Back to front, premultiplied "over": dst = c a + (1 - a) dst.
    glm::dvec3 back(0.0);
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        const double a = splatAlphaAt(*it, 16.0f, 16.0f);
        back = glm::dvec3(it->color) * a + (1.0 - a) * back;
    }
    CHECK(glm::length(front - back) < 1e-12, "front-to-back == back-to-front without early termination");
    CHECK(near(img.rgb[p], front.r, 1e-5) && near(img.rgb[p + 1], front.g, 1e-5) && near(img.rgb[p + 2], front.b, 1e-5),
          "renderer %f %f %f vs %f %f %f", img.rgb[p], img.rgb[p + 1], img.rgb[p + 2], front.r, front.g, front.b);

    // Early termination: opacities 0.99, 0.95, 0.95: T = 1 -> 0.01 -> 5e-4; the third
    // splat would make T = 2.5e-5 < 1e-4, so INRIA stops BEFORE blending it (blue = 0)
    // and never reaches the fourth.
    const GaussianScene opaque = makeScene(means, colors, {0.99f, 0.95f, 0.95f, 0.95f}, 4.0f);
    opt.earlyTermination = true;
    const ReferenceImage e = renderReference(opaque, cam, opt);
    const float* c = &e.rgb[p];
    CHECK(c[2] == 0.0f, "blue (third splat) not blended: %g", c[2]);
    CHECK(e.stats.terminatedPixels > 0, "terminated pixels counted");
    opt.earlyTermination = false;
    const ReferenceImage ne = renderReference(opaque, cam, opt);
    CHECK(ne.rgb[p + 2] > 0.0f && ne.rgb[p + 2] < 2e-3f, "without termination the hidden splats add < 0.2%%: %g",
          ne.rgb[p + 2]);
}

}  // namespace

int main() {
    testCov3D();
    testCov2DOnAxis();
    testCov2DNumericalJacobian();
    testCov2DClamp();
    testTileRectAndCulls();
    testSH();
    testDepthSort();
    testReferenceSingle();
    testReferenceCompositing();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
