// k-means (Lloyd) for the SH codebook (scene_compression.h, format `cb`), on the CPU at
// load time. The algorithm, its cost model and the parameter choices are documented in
// docs/notes/phase5.md ("The codebook").
//
//   1. training sample: S = min(n, 32 k) points, a deterministic random subset (sorted
//      by index so that the passes stream through memory);
//   2. initial centroids: k distinct sample points (deterministic random choice);
//   3. `iterations` Lloyd steps on the sample:
//        assign  every sample point -> nearest centroid       (parallel, the hot loop)
//        update  centroid = mean of its points                (sequential, double sums)
//        empty clusters are re-seeded with the sample points farthest from their centroid;
//   4. final assignment of ALL n points to the trained centroids (parallel).
//
// Nearest centroid: argmin_j |x - c_j|^2 = argmin_j (|c_j|^2 - 2 x.c_j); |x|^2 is the same
// for every j. The kernel computes x.c_j for 8 centroids at a time (centroids stored
// transposed in blocks of 8, so one 8-wide FMA per dimension) and for 4 points at a time
// (each centroid vector loaded once feeds 4 FMAs). AVX2+FMA when the CPU has it (checked
// at run time), otherwise the same loop in scalar code.
//
// Determinism: every point's nearest centroid depends only on the point and the centroids
// (ties -> lowest index), and the update runs sequentially in sample order, so the result
// does not depend on the number of threads. (The SIMD and scalar kernels round differently,
// so results can differ between CPUs with and without AVX2.)
#include "scene/scene_compression.h"

#include "scene/parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#define GSR_AVX2_TARGET
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#define GSR_AVX2_TARGET __attribute__((target("avx2,fma")))
#endif

namespace gsr {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool cpuHasAvx2Fma() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0, osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
    if (!(fma && osxsave && avx)) return false;
    if ((_xgetbv(0) & 6) != 6) return false;  // OS saves the YMM state
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;  // AVX2
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// Centroids in the kernel's layout: blocks of 8 centroids, transposed (block b, dimension d
// -> 8 consecutive floats), plus |c|^2 per centroid. Padding centroids get |c|^2 = +inf.
struct CentroidTable {
    uint32_t k = 0, dims = 0, blocks = 0;
    std::vector<float> t;      // blocks * dims * 8
    std::vector<float> norm2;  // blocks * 8

    void build(const std::vector<float>& c, uint32_t k_, uint32_t dims_) {
        k = k_;
        dims = dims_;
        blocks = (k + 7) / 8;
        t.assign(size_t(blocks) * dims * 8, 0.0f);
        norm2.assign(size_t(blocks) * 8, std::numeric_limits<float>::infinity());
        for (uint32_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (uint32_t d = 0; d < dims; ++d) {
                const float v = c[size_t(j) * dims + d];
                t[(size_t(j / 8) * dims + d) * 8 + j % 8] = v;
                s += double(v) * v;
            }
            norm2[j] = float(s);
        }
    }
};

// Nearest centroid of points x[0..m) (m <= 4), scalar version. dist = |c|^2 - 2 x.c.
void nearestScalar(const CentroidTable& ct, const float* const* x, uint32_t m, uint32_t* best, float* bestDist) {
    for (uint32_t p = 0; p < m; ++p) {
        best[p] = 0;
        bestDist[p] = std::numeric_limits<float>::infinity();
    }
    float acc[4][8];
    for (uint32_t b = 0; b < ct.blocks; ++b) {
        for (uint32_t p = 0; p < m; ++p)
            for (int l = 0; l < 8; ++l) acc[p][l] = 0.0f;
        const float* cb = &ct.t[size_t(b) * ct.dims * 8];
        for (uint32_t d = 0; d < ct.dims; ++d) {
            const float* cv = cb + size_t(d) * 8;
            for (uint32_t p = 0; p < m; ++p) {
                const float xd = x[p][d];
                for (int l = 0; l < 8; ++l) acc[p][l] += xd * cv[l];
            }
        }
        for (uint32_t p = 0; p < m; ++p)
            for (int l = 0; l < 8; ++l) {
                const float dist = ct.norm2[b * 8 + l] - 2.0f * acc[p][l];
                if (dist < bestDist[p]) {
                    bestDist[p] = dist;
                    best[p] = b * 8 + l;
                }
            }
    }
}

#if defined(GSR_AVX2_TARGET)
GSR_AVX2_TARGET void nearestAvx2(const CentroidTable& ct, const float* const* x, uint32_t m, uint32_t* best,
                                 float* bestDist) {
    // Always process 4 point slots; unused slots repeat the last point (results ignored).
    const float* xp[4] = {x[0], x[std::min(1u, m - 1)], x[std::min(2u, m - 1)], x[std::min(3u, m - 1)]};
    float bd[4] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    uint32_t bi[4] = {0, 0, 0, 0};
    const __m256 two = _mm256_set1_ps(2.0f);
    alignas(32) float dist[4][8];
    for (uint32_t b = 0; b < ct.blocks; ++b) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float* cb = &ct.t[size_t(b) * ct.dims * 8];
        for (uint32_t d = 0; d < ct.dims; ++d) {
            const __m256 c = _mm256_loadu_ps(cb + size_t(d) * 8);
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(xp[0][d]), c, a0);
            a1 = _mm256_fmadd_ps(_mm256_set1_ps(xp[1][d]), c, a1);
            a2 = _mm256_fmadd_ps(_mm256_set1_ps(xp[2][d]), c, a2);
            a3 = _mm256_fmadd_ps(_mm256_set1_ps(xp[3][d]), c, a3);
        }
        const __m256 n2 = _mm256_loadu_ps(&ct.norm2[size_t(b) * 8]);
        _mm256_store_ps(dist[0], _mm256_fnmadd_ps(two, a0, n2));
        _mm256_store_ps(dist[1], _mm256_fnmadd_ps(two, a1, n2));
        _mm256_store_ps(dist[2], _mm256_fnmadd_ps(two, a2, n2));
        _mm256_store_ps(dist[3], _mm256_fnmadd_ps(two, a3, n2));
        for (int p = 0; p < 4; ++p)
            for (int l = 0; l < 8; ++l)
                if (dist[p][l] < bd[p]) {
                    bd[p] = dist[p][l];
                    bi[p] = b * 8 + uint32_t(l);
                }
    }
    for (uint32_t p = 0; p < m; ++p) {
        best[p] = bi[p];
        bestDist[p] = bd[p];
    }
}
#endif

// Assigns points idx[0..count) (or 0..count when idx == nullptr) to their nearest centroid.
// Writes assignment and the squared distance |x - c|^2 (>= 0) per point.
void assignAll(const CentroidTable& ct, const float* points, uint32_t dims, const uint32_t* idx, uint32_t count,
               uint32_t threads, bool simd, uint32_t* assignment, float* sqDist) {
    parallelChunks(count, threads, [&](uint32_t begin, uint32_t end) {
        for (uint32_t s = begin; s < end; s += 4) {
            const uint32_t m = std::min(4u, end - s);
            const float* x[4];
            float x2[4];
            for (uint32_t p = 0; p < m; ++p) {
                const uint32_t i = idx ? idx[s + p] : s + p;
                x[p] = points + size_t(i) * dims;
                double acc = 0.0;
                for (uint32_t d = 0; d < dims; ++d) acc += double(x[p][d]) * x[p][d];
                x2[p] = float(acc);
            }
            uint32_t best[4];
            float bestDist[4];
#if defined(GSR_AVX2_TARGET)
            if (simd) nearestAvx2(ct, x, m, best, bestDist);
            else nearestScalar(ct, x, m, best, bestDist);
#else
            (void)simd;
            nearestScalar(ct, x, m, best, bestDist);
#endif
            for (uint32_t p = 0; p < m; ++p) {
                assignment[s + p] = best[p];
                sqDist[s + p] = std::max(0.0f, x2[p] + bestDist[p]);
            }
        }
    }, 1024);
}

// Uniform integer in [0, bound) from a 64-bit generator (bias-free enough for bound << 2^32
// and, unlike std::uniform_int_distribution, identical on every standard library).
uint32_t uniformBelow(std::mt19937_64& rng, uint32_t bound) { return uint32_t((rng() >> 32) * bound >> 32); }

}  // namespace

KMeansResult kmeans(const float* points, uint32_t n, uint32_t dims, const KMeansOptions& opt) {
    const auto start = Clock::now();
    KMeansResult r;
    r.stats.dimensions = dims;
    if (n == 0 || dims == 0) return r;
    const uint32_t k = std::max(1u, std::min(opt.k, n));
    const uint32_t threads = opt.threads ? opt.threads : std::max(1u, std::thread::hardware_concurrency());
    const bool simd = opt.allowSimd && cpuHasAvx2Fma();
    r.stats.simd = simd;
    r.stats.entries = k;

    // 1. Training sample (sorted indices).
    const uint32_t S = std::min(n, opt.samples ? std::max(opt.samples, k) : uint32_t(std::min<uint64_t>(n, 32ull * k)));
    std::mt19937_64 rng(opt.seed);
    std::vector<uint32_t> sample;
    if (S == n) {
        sample.resize(n);
        std::iota(sample.begin(), sample.end(), 0u);
    } else {
        std::vector<uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        for (uint32_t i = 0; i < S; ++i) std::swap(perm[i], perm[i + uniformBelow(rng, n - i)]);  // partial Fisher-Yates
        sample.assign(perm.begin(), perm.begin() + S);
        std::sort(sample.begin(), sample.end());
    }
    r.stats.samples = S;

    // 2. Initial centroids: k distinct sample points.
    std::vector<uint32_t> pick(S);
    std::iota(pick.begin(), pick.end(), 0u);
    for (uint32_t i = 0; i < k; ++i) std::swap(pick[i], pick[i + uniformBelow(rng, S - i)]);
    r.centroids.resize(size_t(k) * dims);
    for (uint32_t j = 0; j < k; ++j)
        std::copy_n(points + size_t(sample[pick[j]]) * dims, dims, &r.centroids[size_t(j) * dims]);

    // 3. Lloyd iterations on the sample.
    CentroidTable table;
    std::vector<uint32_t> assign(S, 0u), previous(S, ~0u);
    std::vector<float> dist(S, 0.0f);
    std::vector<double> sums(size_t(k) * dims);
    std::vector<uint32_t> counts(k);
    for (uint32_t it = 0; it < opt.iterations; ++it) {
        table.build(r.centroids, k, dims);
        assignAll(table, points, dims, sample.data(), S, threads, simd, assign.data(), dist.data());
        r.stats.iterations = it + 1;
        const bool converged = assign == previous;
        previous = assign;

        // Update (sequential: deterministic summation order).
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0u);
        for (uint32_t s = 0; s < S; ++s) {
            const float* x = points + size_t(sample[s]) * dims;
            double* sum = &sums[size_t(assign[s]) * dims];
            for (uint32_t d = 0; d < dims; ++d) sum[d] += x[d];
            ++counts[assign[s]];
        }
        // Empty clusters: re-seed with the sample points farthest from their centroid.
        std::vector<uint32_t> empty;
        for (uint32_t j = 0; j < k; ++j)
            if (counts[j] == 0) empty.push_back(j);
        if (!empty.empty()) {
            std::vector<uint32_t> far(S);
            std::iota(far.begin(), far.end(), 0u);
            const size_t m = std::min(empty.size(), far.size());
            std::partial_sort(far.begin(), far.begin() + std::ptrdiff_t(m), far.end(), [&](uint32_t a, uint32_t b) {
                return dist[a] != dist[b] ? dist[a] > dist[b] : a < b;
            });
            for (size_t e = 0; e < m; ++e)
                std::copy_n(points + size_t(sample[far[e]]) * dims, dims, &r.centroids[size_t(empty[e]) * dims]);
            r.stats.emptyReseeded += uint32_t(m);
        }
        for (uint32_t j = 0; j < k; ++j) {
            if (counts[j] == 0) continue;
            for (uint32_t d = 0; d < dims; ++d)
                r.centroids[size_t(j) * dims + d] = float(sums[size_t(j) * dims + d] / double(counts[j]));
        }
        if (converged) break;
    }
    r.stats.trainMs = msSince(start);

    // Training MSE under the final centroids, then 4. assignment of every point.
    table.build(r.centroids, k, dims);
    const auto tf = Clock::now();
    r.assignment.resize(n);
    std::vector<float> fullDist(n);
    assignAll(table, points, dims, nullptr, n, threads, simd, r.assignment.data(), fullDist.data());
    r.stats.assignMs = msSince(tf);
    double sampleErr = 0.0, fullErr = 0.0;
    for (uint32_t s = 0; s < S; ++s) sampleErr += fullDist[sample[s]];
    for (uint32_t i = 0; i < n; ++i) fullErr += fullDist[i];
    r.stats.sampleMse = sampleErr / (double(S) * dims);
    r.stats.fullMse = fullErr / (double(n) * dims);
    r.stats.totalMs = msSince(start);
    return r;
}

}  // namespace gsr
