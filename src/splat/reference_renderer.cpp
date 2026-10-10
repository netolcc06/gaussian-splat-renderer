#include "splat/reference_renderer.h"

#include "scene/gaussian_scene.h"
#include "splat/depth_sort.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace gsr {

namespace {

using Clock = std::chrono::steady_clock;
double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Runs fn(index) for index in [0, count) on `threads` threads. Work is handed out in
// small chunks from an atomic counter, so uneven work (busy tiles) balances itself.
void parallelFor(uint32_t count, unsigned threads, uint32_t chunk, const std::function<void(uint32_t)>& fn) {
    std::atomic<uint32_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const uint32_t begin = next.fetch_add(chunk);
            if (begin >= count) return;
            const uint32_t end = std::min(count, begin + chunk);
            for (uint32_t i = begin; i < end; ++i) fn(i);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (std::thread& t : pool) t.join();
}

}  // namespace

ReferenceImage renderReference(const GaussianScene& scene, const SplatCamera& cam, const ReferenceOptions& options) {
    const auto tStart = Clock::now();
    ReferenceImage image;
    image.width = cam.width;
    image.height = cam.height;
    image.rgb.assign(size_t(cam.width) * cam.height * 3, 0.0f);
    ReferenceStats& stats = image.stats;
    stats.gaussians = scene.count;
    stats.threads = options.threads ? options.threads : std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = stats.threads;

    // --- 1) Preprocess every gaussian (INRIA preprocessCUDA) ----------------------------
    const int degree = std::min(scene.shDegree, options.maxShDegree);
    const uint32_t K = scene.shCoeffsPerChannel();
    std::vector<ProjectedSplat> splats(scene.count);
    parallelFor(scene.count, threads, 4096, [&](uint32_t i) {
        const glm::vec3* sh = reinterpret_cast<const glm::vec3*>(scene.sh.data() + size_t(i) * K * 3);
        splats[i] = preprocessGaussian(scene.posAlpha[i], glm::vec3(scene.scales[i]), scene.rotations[i], sh, degree,
                                       cam);
    });
    // Phase 6 StopThePop: ray-depth parameters of every visible gaussian.
    std::vector<RayDepthParams> rayDepth;
    if (options.sortWindow > 0) {
        rayDepth.resize(scene.count);
        parallelFor(scene.count, threads, 4096, [&](uint32_t i) {
            if (splats[i].cullReason != uint32_t(CullReason::Visible)) return;
            const glm::vec3 tCam = transformPoint(cam.viewInria, glm::vec3(scene.posAlpha[i]));
            rayDepth[i] = computeRayDepthParams(glm::vec3(scene.scales[i]), scene.rotations[i], tCam, cam.viewInria);
        });
    }
    for (const ProjectedSplat& s : splats) {
        switch (CullReason(s.cullReason)) {
            case CullReason::Visible: stats.visible++; break;
            case CullReason::NearPlane: stats.culledNear++; break;
            case CullReason::SingularCov: stats.culledSingular++; break;
            case CullReason::OffScreen: stats.culledOffScreen++; break;
            case CullReason::LowOpacity: break;  // GPU-only cull, never produced by preprocessGaussian
        }
    }
    const auto tPre = Clock::now();

    // --- 2) Depth order: the shared CPU sort (near-culled gaussians are not in it) ------
    DepthSorter sorter;
    sorter.sort(scene.posAlpha.data(), scene.count, cam.viewInria);
    const auto tSort = Clock::now();

    // --- 3) Per-tile lists in depth order (counting sort by tile) ---------------------
    const uint32_t tileCount = cam.tilesX * cam.tilesY;
    std::vector<uint32_t> tileStart(tileCount + 1, 0);
    // Phase 6: the tiles of each splat come from tile_coverage.h (rect = INRIA's getRect).
    const int gx = int(cam.tilesX), gy = int(cam.tilesY);
    auto forEachTile = [&](const ProjectedSplat& s, auto&& fn) {
        const Coverage cv = splatCoverage(options.bounds, s, cam.tilesX, cam.tilesY, kTileSize, gx, gy);
        const auto [r0, r1] = coverageRows(cv, kTileSize);
        for (int ty = r0; ty < r1; ++ty) {
            const auto [x0, x1] = coverageRowSpan(cv, ty, kTileSize);
            for (int tx = x0; tx < x1; ++tx) fn(uint32_t(ty * gx + tx));
        }
    };
    for (uint32_t idx : sorter.order()) {
        const ProjectedSplat& s = splats[idx];
        if (s.cullReason != uint32_t(CullReason::Visible)) continue;
        forEachTile(s, [&](uint32_t t) { tileStart[t + 1]++; });
    }
    for (uint32_t t = 0; t < tileCount; ++t) tileStart[t + 1] += tileStart[t];
    stats.tileInstances = tileStart[tileCount];
    std::vector<uint32_t> tileLists(tileStart[tileCount]);
    {
        std::vector<uint32_t> cursor(tileStart.begin(), tileStart.end() - 1);
        for (uint32_t idx : sorter.order()) {  // depth order is preserved inside each tile
            const ProjectedSplat& s = splats[idx];
            if (s.cullReason != uint32_t(CullReason::Visible)) continue;
            forEachTile(s, [&](uint32_t t) { tileLists[cursor[t]++] = idx; });
        }
    }
    const auto tBin = Clock::now();

    // --- 4) Rasterize: one tile per task, front-to-back per pixel (renderCUDA) ---------
    std::atomic<uint64_t> evaluated{0}, blended{0}, terminated{0};
    parallelFor(tileCount, threads, 1, [&](uint32_t tile) {
        const uint32_t tx = tile % cam.tilesX, ty = tile / cam.tilesX;
        // Gather this tile's splats contiguously (the list is walked by 256 pixels).
        std::vector<ProjectedSplat> local;
        local.reserve(tileStart[tile + 1] - tileStart[tile]);
        for (uint32_t j = tileStart[tile]; j < tileStart[tile + 1]; ++j) local.push_back(splats[tileLists[j]]);
        std::vector<RayDepthParams> localDepth;
        if (options.sortWindow > 0)
            for (uint32_t j = tileStart[tile]; j < tileStart[tile + 1]; ++j) localDepth.push_back(rayDepth[tileLists[j]]);
        struct Pending {
            float depth, alpha;
            glm::vec3 color;
        };
        std::vector<Pending> window;

        uint64_t tileEvaluated = 0, tileBlended = 0, tileTerminated = 0;
        const uint32_t x0 = tx * kTileSize, y0 = ty * kTileSize;
        const uint32_t x1 = std::min(cam.width, x0 + kTileSize), y1 = std::min(cam.height, y0 + kTileSize);
        for (uint32_t y = y0; y < y1; ++y) {
            for (uint32_t x = x0; x < x1; ++x) {
                const float px = float(x), py = float(y);  // INRIA pixel coordinates
                float T = 1.0f;
                glm::vec3 C(0.0f);
                if (options.sortWindow > 0) {
                    // StopThePop-style: same blend rule, different order (tile_render.comp GSR_STOPTHEPOP).
                    const float vx = (px - 0.5f * (float(cam.width) - 1.0f)) / cam.focalX;
                    const float vy = (py - 0.5f * (float(cam.height) - 1.0f)) / cam.focalY;
                    bool done = false;
                    auto blend = [&](float alpha, const glm::vec3& color) {
                        const float testT = T * (1.0f - alpha);
                        if (testT < kMinTransmittance) {
                            done = true;
                            ++tileTerminated;
                            return;
                        }
                        C += color * alpha * T;
                        T = testT;
                        ++tileBlended;
                    };
                    const bool exact = options.sortWindow == ReferenceOptions::kExactPixelSort;
                    const size_t K = options.sortWindow;
                    window.clear();
                    for (size_t j = 0; j < local.size() && !done; ++j) {
                        const ProjectedSplat& s = local[j];
                        if (options.bound == CoverageBound::Radius && !insideRadiusSquare(s, px, py)) continue;
                        ++tileEvaluated;
                        const float alpha = splatAlphaAt(s, px, py);
                        if (alpha == 0.0f) continue;
                        const Pending e{localDepth[j].depthAt(vx, vy), alpha, s.color};
                        if (exact) {
                            window.push_back(e);
                        } else if (window.size() == K) {
                            if (e.depth < window[0].depth) {
                                blend(e.alpha, e.color);
                                continue;
                            }
                            blend(window[0].alpha, window[0].color);
                            size_t i = 0;
                            for (; i + 1 < K && window[i + 1].depth <= e.depth; ++i) window[i] = window[i + 1];
                            window[i] = e;
                        } else {
                            size_t i = window.size();
                            window.push_back(e);
                            for (; i > 0 && window[i - 1].depth > e.depth; --i) window[i] = window[i - 1];
                            window[i] = e;
                        }
                    }
                    if (exact)
                        std::stable_sort(window.begin(), window.end(),
                                         [](const Pending& a, const Pending& b) { return a.depth < b.depth; });
                    for (size_t i = 0; i < window.size() && !done; ++i) blend(window[i].alpha, window[i].color);
                    const glm::vec3 out = C + T * options.background;
                    float* dst = &image.rgb[(size_t(y) * cam.width + x) * 3];
                    dst[0] = out.r;
                    dst[1] = out.g;
                    dst[2] = out.b;
                    continue;
                }
                for (const ProjectedSplat& s : local) {
                    if (options.bound == CoverageBound::Radius && !insideRadiusSquare(s, px, py)) continue;
                    ++tileEvaluated;
                    const float alpha = splatAlphaAt(s, px, py);
                    if (alpha == 0.0f) continue;
                    const float testT = T * (1.0f - alpha);
                    if (options.earlyTermination && testT < kMinTransmittance) {
                        ++tileTerminated;  // INRIA: done = true, this splat is not blended
                        break;
                    }
                    C += s.color * alpha * T;  // same evaluation order as CUDA: (c * alpha) * T
                    T = testT;
                    ++tileBlended;
                }
                const glm::vec3 out = C + T * options.background;
                float* dst = &image.rgb[(size_t(y) * cam.width + x) * 3];
                dst[0] = out.r;
                dst[1] = out.g;
                dst[2] = out.b;
            }
        }
        evaluated += tileEvaluated;
        blended += tileBlended;
        terminated += tileTerminated;
    });
    stats.evaluatedFragments = evaluated;
    stats.blendedFragments = blended;
    stats.terminatedPixels = terminated;
    const auto tEnd = Clock::now();

    stats.preprocessMs = msBetween(tStart, tPre);
    stats.sortMs = msBetween(tPre, tSort);
    stats.binMs = msBetween(tSort, tBin);
    stats.rasterMs = msBetween(tBin, tEnd);
    stats.totalMs = msBetween(tStart, tEnd);
    return image;
}

}  // namespace gsr
