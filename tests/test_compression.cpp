// CPU tests for Phase 5 scene compression (CTest `compression`), no GPU needed:
//
//   1. half <-> float: every finite half survives float -> half -> float -> half exactly
//      (incl. subnormals and signed zero); rounding is to nearest, ties to even (checked at
//      the midpoint between every pair of adjacent halves); the relative error of random
//      floats is <= 2^-11 in the normal range and the absolute error <= 2^-25 below it;
//      overflow -> Inf, NaN stays NaN.
//   2. Q8 band quantization: |v - q s| <= s / 2, 127 s >= max |v| > 127 (s - 1 ulp), zero bands.
//   3. Every format x layout on a synthetic SH3 scene, decoded on the CPU:
//      positions bit-exact everywhere; f32 records bit-exact; fp16 attributes within the
//      half rounding bound (quaternion: renormalized); q8 rest within s/2 per band; cb rest
//      equal to the assigned codebook entry, DC within the half bound; split == aos decode;
//      record sizes (f16 128 B, q8 96 B, cb 40 B per gaussian); dropping SH bands.
//   4. k-means: identical result for 1 and 8 threads and on a second run (determinism);
//      the final assignment is the exact nearest centroid (checked in double precision,
//      ties within float rounding allowed); the error after 10 iterations is below the
//      error after 1.
//   5. Phase 6 per-gaussian SH degree: the band bound is never exceeded (and tight for band 1),
//      selectShDegree picks the highest band above the opacity-weighted threshold, and every
//      format x layout encodes/decodes it (kept coefficients unchanged, dropped ones zero,
//      compacted layouts smaller).
//
// Exit code: 0 = all checks passed, 1 = a check failed.
#include "scene/gaussian_scene.h"
#include "scene/half.h"
#include "scene/scene_compression.h"
#include "splat/splat_math.h"  // SH constants (Phase 6 band bound test)

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
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

// ---------------------------------------------------------------------------------------
// 1. Half floats
// ---------------------------------------------------------------------------------------
void testHalf() {
    std::printf("half <-> float\n");
    uint32_t badRoundTrip = 0, badTies = 0, finite = 0;
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        const uint16_t hb = uint16_t(h);
        const uint32_t exponent = (h >> 10) & 0x1Fu;
        if (exponent == 31) continue;  // Inf/NaN checked below
        ++finite;
        const float f = halfToFloat(hb);
        if (floatToHalf(f) != hb) ++badRoundTrip;
        // Midpoint between h and the next half of larger magnitude (same sign): must round
        // to the one with an even mantissa.
        const uint32_t mag = h & 0x7FFFu;
        if (mag < 0x7BFFu) {
            const uint16_t next = uint16_t(h + 1);
            const double mid = (double(halfToFloat(hb)) + double(halfToFloat(next))) * 0.5;
            const float midF = float(mid);  // exact: a half midpoint has <= 12 significant bits
            const uint16_t r = floatToHalf(midF);
            const uint16_t even = (h & 1u) ? next : hb;
            if (r != even) ++badTies;
            // Just above / below the midpoint round away / towards.
            const float above = std::nextafter(midF, std::copysign(std::numeric_limits<float>::infinity(), midF));
            const float below = std::nextafter(midF, 0.0f);
            if (floatToHalf(above) != next || floatToHalf(below) != hb) ++badTies;
        }
    }
    CHECK(finite == 63488u, "finite half count %u", finite);
    CHECK(badRoundTrip == 0, "%u halves do not survive half -> float -> half", badRoundTrip);
    CHECK(badTies == 0, "%u midpoint rounding errors", badTies);
    CHECK(floatToHalf(65504.0f) == 0x7BFFu, "max half");
    CHECK(floatToHalf(65519.0f) == 0x7BFFu && floatToHalf(65520.0f) == 0x7C00u, "overflow threshold");
    CHECK(floatToHalf(1e30f) == 0x7C00u && floatToHalf(-1e30f) == 0xFC00u, "overflow -> Inf");
    CHECK(std::isinf(halfToFloat(0x7C00u)) && std::isnan(halfToFloat(floatToHalf(std::nanf("")))), "Inf/NaN");
    CHECK(floatToHalf(-0.0f) == 0x8000u && floatToHalf(1e-9f) == 0, "signed zero / underflow");
    CHECK(halfToFloat(1) == std::ldexp(1.0f, -24), "smallest subnormal");

    std::mt19937 rng(7);
    double worstRel = 0.0, worstAbs = 0.0;
    for (int i = 0; i < 2'000'000; ++i) {
        const float e = std::uniform_real_distribution<float>(-26.0f, 15.9f)(rng);
        const float v = std::exp2(e) * (rng() & 1u ? -1.0f : 1.0f);
        const float r = roundToHalf(v);
        if (std::abs(v) >= std::ldexp(1.0f, -14)) worstRel = std::max(worstRel, std::abs(double(r) - v) / std::abs(double(v)));
        else worstAbs = std::max(worstAbs, std::abs(double(r) - v));
    }
    std::printf("  worst relative error (normal range) %.3e (bound 2^-11 = %.3e), worst absolute error (subnormal) "
                "%.3e (bound 2^-25 = %.3e)\n",
                worstRel, std::ldexp(1.0, -11), worstAbs, std::ldexp(1.0, -25));
    CHECK(worstRel <= std::ldexp(1.0, -11), "relative error %.3e", worstRel);
    CHECK(worstAbs <= std::ldexp(1.0, -25), "absolute error %.3e", worstAbs);
}

// ---------------------------------------------------------------------------------------
// 2. Q8
// ---------------------------------------------------------------------------------------
void testQ8() {
    std::printf("Q8 band quantization\n");
    std::mt19937 rng(11);
    uint32_t bad = 0, badScale = 0;
    double worstRatio = 0.0;
    for (int t = 0; t < 200'000; ++t) {
        const uint32_t n = (t % 3 == 0) ? 9u : (t % 3 == 1) ? 15u : 21u;
        float v[21];
        const float mag = std::exp2(std::uniform_real_distribution<float>(-20.0f, 3.0f)(rng));
        for (uint32_t i = 0; i < n; ++i) v[i] = std::uniform_real_distribution<float>(-mag, mag)(rng);
        if (t % 1000 == 0) v[rng() % n] = mag * 1.0f;  // exact maximum
        int8_t q[21];
        const uint16_t sh = quantizeBandQ8(v, n, q);
        const float s = halfToFloat(sh);
        float m = 0.0f;
        for (uint32_t i = 0; i < n; ++i) m = std::max(m, std::abs(v[i]));
        if (!(s * 127.0f >= m) || (sh > 1 && halfToFloat(uint16_t(sh - 1)) * 127.0f >= m)) ++badScale;
        for (uint32_t i = 0; i < n; ++i) {
            const double err = std::abs(double(v[i]) - double(q[i]) * s);
            if (err > 0.5 * s * (1.0 + 1e-6) || q[i] < -127) ++bad;
            worstRatio = std::max(worstRatio, err / s);
        }
    }
    float zeros[9] = {};
    int8_t qz[9];
    CHECK(quantizeBandQ8(zeros, 9, qz) == 0 && std::all_of(qz, qz + 9, [](int8_t x) { return x == 0; }), "zero band");
    std::printf("  worst |v - q s| / s = %.4f (bound 0.5)\n", worstRatio);
    CHECK(bad == 0, "%u quantized values outside s/2", bad);
    CHECK(badScale == 0, "%u scales not minimal / too small", badScale);
}

// ---------------------------------------------------------------------------------------
// 3. Formats
// ---------------------------------------------------------------------------------------
GaussianScene makeScene(uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);
    GaussianScene s;
    s.count = n;
    s.shDegree = 3;
    s.posAlpha.resize(n);
    s.scales.resize(n);
    s.rotations.resize(n);
    s.sh.resize(size_t(n) * 48);
    for (uint32_t i = 0; i < n; ++i) {
        s.posAlpha[i] = {50.0f * u(rng), 50.0f * u(rng), 50.0f * u(rng), 0.5f + 0.5f * u(rng)};
        s.scales[i] = {std::exp(-7.0f + 5.0f * u(rng)), std::exp(-7.0f + 5.0f * u(rng)), std::exp(-7.0f + 5.0f * u(rng)), 0.0f};
        glm::vec4 q(g(rng), g(rng), g(rng), g(rng));
        s.rotations[i] = q / glm::length(q);
        for (uint32_t k = 0; k < 16; ++k)
            for (uint32_t c = 0; c < 3; ++c) {
                // DC ~ N(0, 1); rest smaller, with clusters (shared directions) and some zero rows.
                const float v = k == 0 ? g(rng) : (i % 7 == 0 ? 0.0f : 0.15f * g(rng) + 0.1f * float(int(i % 5) - 2));
                s.sh[(size_t(i) * 16 + k) * 3 + c] = v;
            }
    }
    return s;
}

float halfBound(float v) {
    // Round-to-nearest error bound of the half representation of v.
    const float a = std::abs(v);
    if (a < std::ldexp(1.0f, -14)) return std::ldexp(1.0f, -25);
    return a * std::ldexp(1.0f, -11) * 1.0001f;
}

uint32_t bandOf(uint32_t k) { return k < 4 ? 1u : k < 9 ? 2u : 3u; }

void testFormats() {
    std::printf("scene formats (encode -> CPU decode)\n");
    const GaussianScene scene = makeScene(20'000, 3);
    const uint32_t n = scene.count;
    struct Case {
        SceneFormat f;
        SceneLayout l;
        double bytes;
    };
    const Case cases[] = {{SceneFormat::F32, SceneLayout::Soa, 240.0},  {SceneFormat::F32, SceneLayout::Split, 240.0},
                          {SceneFormat::F32, SceneLayout::Aos, 240.0},  {SceneFormat::F16, SceneLayout::Split, 128.0},
                          {SceneFormat::F16, SceneLayout::Aos, 128.0},  {SceneFormat::Q8, SceneLayout::Split, 96.0},
                          {SceneFormat::Q8, SceneLayout::Aos, 96.0},    {SceneFormat::Codebook, SceneLayout::Split, 40.0},
                          {SceneFormat::Codebook, SceneLayout::Aos, 48.0}};
    std::vector<std::vector<glm::vec3>> splitSh;  // decoded SH per format (split), to compare with aos
    for (const Case& c : cases) {
        CompressionOptions o;
        o.format = c.f;
        o.layout = c.l;
        o.codebookSize = 256;
        const EncodedScene e = encodeScene(scene, o);
        const std::string name = std::string(sceneFormatName(c.f)) + "/" + sceneLayoutName(c.l);
        // Codebook bytes are amortized: count them separately.
        const double cbBytes = c.f == SceneFormat::Codebook ? double(e.buffers[2].size() * 4) : 0.0;
        const double perGaussian = (double(e.totalBytes()) - cbBytes) / n;
        CHECK(std::abs(perGaussian - c.bytes) < 1e-9, "%s: %.2f B/gaussian, expected %.0f", name.c_str(), perGaussian,
              c.bytes);
        uint32_t badPos = 0, badGeom = 0, badSh = 0;
        DecodedGaussian d;
        for (uint32_t i = 0; i < n; ++i) {
            decodeGaussian(e, i, d);
            const glm::vec4 pa = scene.posAlpha[i];
            badPos += d.posAlpha.x != pa.x || d.posAlpha.y != pa.y || d.posAlpha.z != pa.z;
            const glm::vec3 sg(scene.scales[i]);
            const glm::vec4 q = scene.rotations[i];
            if (c.f == SceneFormat::F32) {
                badGeom += d.posAlpha.w != pa.w || d.sigma != sg || d.rotation != q;
            } else {
                badGeom += std::abs(d.posAlpha.w - pa.w) > halfBound(pa.w);
                for (int a = 0; a < 3; ++a) badGeom += std::abs(d.sigma[a] - sg[a]) > halfBound(sg[a]);
                // Quaternion: per-component half rounding (<= 2^-12 for |q| <= 1), then
                // renormalization (changes the length by at most ~2^-11).
                badGeom += std::abs(glm::length(d.rotation) - 1.0f) > 1e-6f;
                for (int a = 0; a < 4; ++a) badGeom += std::abs(d.rotation[a] - q[a]) > 1.0e-3f;
            }
            for (uint32_t k = 0; k < 16; ++k)
                for (int ch = 0; ch < 3; ++ch) {
                    const float v = scene.sh[(size_t(i) * 16 + k) * 3 + ch], r = d.sh[k][ch];
                    bool ok = true;
                    switch (c.f) {
                        case SceneFormat::F32: ok = r == v; break;
                        case SceneFormat::F16: ok = std::abs(r - v) <= halfBound(v); break;
                        case SceneFormat::Q8:
                        case SceneFormat::Codebook:
                            if (k == 0) ok = std::abs(r - v) <= halfBound(v);
                            break;
                    }
                    badSh += !ok;
                }
            if (c.f == SceneFormat::Q8) {
                // Rest: |v - r| <= s_band / 2 with s_band = the band's half scale (recompute it).
                for (uint32_t band = 1; band <= 3; ++band) {
                    float m = 0.0f;
                    for (uint32_t k = band * band; k < (band + 1) * (band + 1); ++k)
                        for (int ch = 0; ch < 3; ++ch) m = std::max(m, std::abs(scene.sh[(size_t(i) * 16 + k) * 3 + ch]));
                    const float s = m / 127.0f * (1.0f + std::ldexp(1.0f, -9)) + std::ldexp(1.0f, -24);
                    for (uint32_t k = band * band; k < (band + 1) * (band + 1); ++k)
                        for (int ch = 0; ch < 3; ++ch)
                            badSh += std::abs(d.sh[k][ch] - scene.sh[(size_t(i) * 16 + k) * 3 + ch]) > 0.5f * s * 1.0001f;
                }
            }
            if (c.f == SceneFormat::Codebook) {
                // Rest == the codebook entry of this gaussian (read straight from the buffers).
                const std::vector<uint32_t>& colors = c.l == SceneLayout::Aos ? e.buffers[0] : e.buffers[1];
                const uint32_t w1 = c.l == SceneLayout::Aos ? colors[(size_t(i) * e.colorStride + e.colorOffset) * 4 + 1]
                                                             : colors[size_t(i) * 2 + 1];
                const uint32_t index = w1 >> 16;
                badSh += index >= e.codebook.entries;
                for (uint32_t k = 1; k < 16; ++k)
                    for (int ch = 0; ch < 3; ++ch)
                        badSh += d.sh[k][ch] != bitsToFloat(e.buffers[2][size_t(index) * e.codebookStride * 4 + (k - 1) * 3 + ch]);
            }
        }
        CHECK(badPos == 0, "%s: %u positions not bit-exact", name.c_str(), badPos);
        CHECK(badGeom == 0, "%s: %u geometry values outside their bound", name.c_str(), badGeom);
        CHECK(badSh == 0, "%s: %u SH values outside their bound", name.c_str(), badSh);
        // split and aos must decode to the same values.
        std::vector<glm::vec3> all;
        all.reserve(size_t(n) * 16);
        for (uint32_t i = 0; i < n; ++i) {
            decodeGaussian(e, i, d);
            all.insert(all.end(), d.sh.begin(), d.sh.end());
            all.push_back(glm::vec3(d.posAlpha.w, d.sigma.x, d.sigma.y));
        }
        if (c.l == SceneLayout::Split) splitSh = {all};
        else if (c.l == SceneLayout::Aos && !splitSh.empty())
            CHECK(all == splitSh[0], "%s: aos decodes differently from split", name.c_str());
        std::printf("  %-12s %6.1f B/gaussian (+%.0f KB codebook), positions/geometry/SH mismatches %u/%u/%u\n",
                    name.c_str(), perGaussian, cbBytes / 1024.0, badPos, badGeom, badSh);
    }

    // Dropping stored bands: degree 1 keeps coefficients 0-3 exactly (f32) / within half bounds.
    for (SceneFormat f : {SceneFormat::F32, SceneFormat::F16, SceneFormat::Q8, SceneFormat::Codebook}) {
        CompressionOptions o;
        o.format = f;
        o.layout = f == SceneFormat::F32 ? SceneLayout::Soa : SceneLayout::Split;
        o.maxShDegree = 1;
        o.codebookSize = 64;
        const EncodedScene e = encodeScene(scene, o);
        CHECK(e.shDegree == 1 && e.shCoeffs == 4, "%s degree 1: stored degree %d", sceneFormatName(f), e.shDegree);
        uint32_t bad = 0;
        DecodedGaussian d;
        for (uint32_t i = 0; i < n; i += 13) {
            decodeGaussian(e, i, d);
            bad += d.sh.size() != 4;
            const float v = scene.sh[(size_t(i) * 16 + 0) * 3 + 0];
            bad += std::abs(d.sh[0].x - v) > (f == SceneFormat::F32 ? 0.0f : halfBound(v));
        }
        CHECK(bad == 0, "%s degree 1: %u mismatches", sceneFormatName(f), bad);
        const double bytes = double(e.totalBytes() - (f == SceneFormat::Codebook ? e.buffers[2].size() * 4 : 0)) / n;
        std::printf("  degree 1 stored, %-3s: %.1f B/gaussian\n", sceneFormatName(f), bytes);
    }
}

// ---------------------------------------------------------------------------------------
// 4. k-means
// ---------------------------------------------------------------------------------------
void testKMeans() {
    std::printf("k-means codebook\n");
    const GaussianScene scene = makeScene(40'000, 5);
    const uint32_t n = scene.count, dims = 45;
    std::vector<float> points(size_t(n) * dims);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t k = 1; k < 16; ++k)
            for (uint32_t c = 0; c < 3; ++c) points[size_t(i) * dims + (k - 1) * 3 + c] = scene.sh[(size_t(i) * 16 + k) * 3 + c];

    KMeansOptions o;
    o.k = 512;
    o.iterations = 10;
    o.threads = 1;
    const auto t0 = std::chrono::steady_clock::now();
    const KMeansResult a = kmeans(points.data(), n, dims, o);
    const double ms1 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    o.threads = 8;
    const auto t1 = std::chrono::steady_clock::now();
    const KMeansResult b = kmeans(points.data(), n, dims, o);
    const double ms8 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    const KMeansResult c = kmeans(points.data(), n, dims, o);
    CHECK(a.centroids == b.centroids && a.assignment == b.assignment, "1 vs 8 threads differ");
    CHECK(b.centroids == c.centroids && b.assignment == c.assignment, "two runs differ");
    std::printf("  k=%u, %u points x %u dims: 1 thread %.0f ms, 8 threads %.0f ms, MSE %.4g, %u iterations, %s; "
                "deterministic: %s\n",
                o.k, n, dims, ms1, ms8, b.stats.fullMse, b.stats.iterations, b.stats.simd ? "AVX2" : "scalar",
                (a.assignment == b.assignment && b.assignment == c.assignment) ? "yes" : "NO");

    // Exact nearest centroid (double precision), allowing float-rounding ties.
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < n; i += 7) {
        const float* x = &points[size_t(i) * dims];
        double best = std::numeric_limits<double>::infinity();
        for (uint32_t j = 0; j < o.k; ++j) {
            double d = 0.0;
            for (uint32_t t = 0; t < dims; ++t) {
                const double e = double(x[t]) - b.centroids[size_t(j) * dims + t];
                d += e * e;
            }
            best = std::min(best, d);
        }
        double mine = 0.0;
        for (uint32_t t = 0; t < dims; ++t) {
            const double e = double(x[t]) - b.centroids[size_t(b.assignment[i]) * dims + t];
            mine += e * e;
        }
        wrong += mine > best + 1e-5 * (1.0 + best);
    }
    CHECK(wrong == 0, "%u points not assigned to their nearest centroid", wrong);

    o.iterations = 1;
    const KMeansResult one = kmeans(points.data(), n, dims, o);
    std::printf("  MSE after 1 iteration %.4g, after %u: %.4g\n", one.stats.fullMse, b.stats.iterations, b.stats.fullMse);
    CHECK(b.stats.fullMse < one.stats.fullMse, "Lloyd iterations did not reduce the error");

    // Scalar kernel = same algorithm (may round differently from FMA, so compare the error).
    o.iterations = 10;
    o.allowSimd = false;
    const KMeansResult sc = kmeans(points.data(), n, dims, o);
    CHECK(std::abs(sc.stats.fullMse - b.stats.fullMse) <= 0.02 * b.stats.fullMse, "scalar MSE %.4g vs %.4g",
          sc.stats.fullMse, b.stats.fullMse);
}

// ---------------------------------------------------------------------------------------
// 5. Phase 6: per-gaussian SH degree (--sh-threshold)
// ---------------------------------------------------------------------------------------
// The colour contribution of band l alone for direction dir (the terms of evalSH()).
glm::vec3 bandContribution(const glm::vec3* sh, int band, const glm::vec3& d) {
    const float x = d.x, y = d.y, z = d.z;
    if (band == 1) return -kShC1 * y * sh[1] + kShC1 * z * sh[2] - kShC1 * x * sh[3];
    const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    if (band == 2)
        return kShC2[0] * xy * sh[4] + kShC2[1] * yz * sh[5] + kShC2[2] * (2.0f * zz - xx - yy) * sh[6] +
               kShC2[3] * xz * sh[7] + kShC2[4] * (xx - yy) * sh[8];
    return kShC3[0] * y * (3.0f * xx - yy) * sh[9] + kShC3[1] * xy * z * sh[10] + kShC3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
           kShC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] + kShC3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
           kShC3[5] * z * (xx - yy) * sh[14] + kShC3[6] * x * (xx - 3.0f * yy) * sh[15];
}

void testVariableSh() {
    std::printf("Per-gaussian SH degree (Phase 6)\n");
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0f, 1.0f);
    // a) The band bound: never exceeded over many directions; tight for band 1 (the
    //    band-1 terms are a linear function of dir, whose maximum IS the Cauchy-Schwarz bound).
    std::vector<glm::vec3> dirs(4000);
    for (auto& d : dirs) d = glm::normalize(glm::vec3(g(rng), g(rng), g(rng)));
    uint32_t exceeded = 0;
    double band1Tightness = 1.0, band3Tightness = 0.0;
    for (int trial = 0; trial < 100; ++trial) {
        glm::vec3 sh[16];
        for (auto& c : sh) c = glm::vec3(g(rng), g(rng), g(rng));
        for (int band = 1; band <= 3; ++band) {
            const float bound = shBandMaxContribution(sh, band);
            float maxSeen = 0.0f;
            for (const auto& d : dirs) {
                const glm::vec3 v = bandContribution(sh, band, d);
                maxSeen = std::max(maxSeen, std::max(std::abs(v.x), std::max(std::abs(v.y), std::abs(v.z))));
            }
            exceeded += maxSeen > bound * 1.0001f;
            if (band == 1) band1Tightness = std::min(band1Tightness, double(maxSeen / bound));
            if (band == 3) band3Tightness += double(maxSeen / bound) / 100.0;
        }
    }
    std::printf("  band bound exceeded: %u (must be 0); band 1 max/bound >= %.4f; band 3 mean max/bound %.3f\n",
                exceeded, band1Tightness, band3Tightness);
    CHECK(exceeded == 0, "the band contribution bound was exceeded %u times", exceeded);
    CHECK(band1Tightness > 0.99, "band-1 bound not tight (%.4f)", band1Tightness);

    // b) Selection: the highest band whose opacity-weighted bound reaches the threshold.
    {
        glm::vec3 sh[16] = {};
        CHECK(selectShDegree(sh, 3, 1.0f, 0.0f) == 0, "all-zero rest must give degree 0 even at threshold 0");
        sh[5] = glm::vec3(0.0f, 0.1f, 0.0f);  // band 2 only
        const float b2 = shBandMaxContribution(sh, 2);
        CHECK(selectShDegree(sh, 3, 1.0f, b2 * 255.0f * 0.99f) == 2, "band 2 above the threshold");
        CHECK(selectShDegree(sh, 3, 1.0f, b2 * 255.0f * 1.01f) == 0, "band 2 below the threshold");
        CHECK(selectShDegree(sh, 3, 0.5f, b2 * 255.0f * 0.6f) == 0, "opacity scales the contribution");
        CHECK(selectShDegree(sh, 1, 1.0f, 0.0f) == 0, "stored degree 1: band 2 is not considered");
        sh[10] = glm::vec3(1e-6f, 0.0f, 0.0f);  // a tiny band 3
        CHECK(selectShDegree(sh, 3, 1.0f, 0.0f) == 3, "threshold 0 keeps every non-zero band");
    }

    // c) Encode/decode with a threshold, every format x layout: per-gaussian degree as
    //    selected, kept coefficients identical to the decode without threshold, dropped ones 0,
    //    and fewer bytes for the compacted layouts.
    const GaussianScene scene = makeScene(20'000, 9);
    struct Case {
        SceneFormat f;
        SceneLayout l;
    };
    const Case cases[] = {{SceneFormat::F32, SceneLayout::Soa},   {SceneFormat::F32, SceneLayout::Split},
                          {SceneFormat::F32, SceneLayout::Aos},   {SceneFormat::F16, SceneLayout::Split},
                          {SceneFormat::F16, SceneLayout::Aos},   {SceneFormat::Q8, SceneLayout::Split},
                          {SceneFormat::Codebook, SceneLayout::Split}};
    for (const Case& c : cases) {
        CompressionOptions o;
        o.format = c.f;
        o.layout = c.l;
        o.codebookSize = 256;
        const EncodedScene full = encodeScene(scene, o);
        o.shThreshold = 20.0f;  // drops a good share of bands in the synthetic scene
        const EncodedScene var = encodeScene(scene, o);
        uint32_t badDegree = 0, badKept = 0, badDropped = 0;
        uint32_t hist[4] = {0, 0, 0, 0};
        DecodedGaussian a, b;
        std::vector<glm::vec3> src(16);
        for (uint32_t i = 0; i < scene.count; ++i) {
            for (uint32_t k = 0; k < 16; ++k) src[k] = scene.shCoeff(i, k);
            const int d = selectShDegree(src.data(), 3, scene.posAlpha[i].w, 20.0f);
            hist[d]++;
            decodeGaussian(full, i, a);
            decodeGaussian(var, i, b);
            const uint32_t kept = uint32_t((d + 1) * (d + 1));
            for (uint32_t k = 0; k < 16; ++k) {
                if (k < kept) {
                    // cb: the codebook was trained on the zeroed vectors, so only DC is compared.
                    if ((c.f != SceneFormat::Codebook || k == 0) && a.sh[k] != b.sh[k]) ++badKept;
                } else if (b.sh[k] != glm::vec3(0.0f)) {
                    ++badDropped;
                }
            }
            badDegree += var.degreeHistogram[size_t(d)] == 0;
        }
        const bool compact = c.l != SceneLayout::Aos && c.f != SceneFormat::Codebook;
        std::printf("  %-3s %-5s degrees 0/1/2/3 = %u/%u/%u/%u, %.1f -> %.1f B/gaussian (%s)\n", sceneFormatName(c.f),
                    sceneLayoutName(c.l), hist[0], hist[1], hist[2], hist[3], full.bytesPerGaussian(),
                    var.bytesPerGaussian(), compact ? "compacted" : "degree only");
        CHECK(var.variableSh == (compact ? VariableSh::Compact : VariableSh::DegreeOnly), "%s %s: variable mode",
              sceneFormatName(c.f), sceneLayoutName(c.l));
        CHECK(badKept == 0 && badDropped == 0, "%s %s: %u kept coefficients differ, %u dropped not zero",
              sceneFormatName(c.f), sceneLayoutName(c.l), badKept, badDropped);
        CHECK(var.degreeHistogram[0] == hist[0] && var.degreeHistogram[3] == hist[3] && badDegree == 0,
              "%s %s: degree histogram differs from selectShDegree", sceneFormatName(c.f), sceneLayoutName(c.l));
        CHECK(hist[0] > 0 && hist[3] > 0 && hist[0] + hist[3] < scene.count, "test scene should mix degrees");
        if (compact) CHECK(var.totalBytes() < full.totalBytes(), "%s: compaction saved nothing", sceneFormatName(c.f));
        else CHECK(var.totalBytes() == full.totalBytes(), "%s %s: degree-only must not change sizes", sceneFormatName(c.f),
                   sceneLayoutName(c.l));
    }
}

}  // namespace

int main() {
    testHalf();
    testQ8();
    testFormats();
    testKMeans();
    testVariableSh();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
