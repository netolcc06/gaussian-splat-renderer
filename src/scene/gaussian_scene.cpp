#include "scene/gaussian_scene.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gsr {

namespace {

// sorted[int(p * (n - 1))] without sorting everything (nth_element is O(n)).
float percentile(std::vector<float>& values, double p) {
    if (values.empty()) return 0.0f;
    const size_t index = size_t(p * double(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + std::ptrdiff_t(index), values.end());
    return values[index];
}

}  // namespace

glm::vec3 GaussianScene::dcColor(uint32_t i) const { return glm::vec3(0.5f) + kShC0 * shCoeff(i, 0); }

size_t GaussianScene::cpuBytes() const {
    return posAlpha.size() * sizeof(glm::vec4) + scales.size() * sizeof(glm::vec4) +
           rotations.size() * sizeof(glm::vec4) + sh.size() * sizeof(float) + rawOpacity.size() * sizeof(float) +
           rawLogScale.size() * sizeof(glm::vec3) + rawRotation.size() * sizeof(glm::vec4);
}

void computePositionStats(const glm::vec4* posAlpha, size_t n, SceneStats& s) {
    if (n == 0) return;
    const float inf = std::numeric_limits<float>::infinity();
    glm::vec3 lo(inf), hi(-inf);
    float aMin = inf, aMax = -inf;
    double aSum = 0.0;
    uint32_t aLow = 0;
    std::vector<float> axis[3];
    for (auto& a : axis) a.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const glm::vec4& pa = posAlpha[i];
        const glm::vec3 p(pa);
        // Non-finite positions would poison the bounds; they are counted by the loader.
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
            for (int c = 0; c < 3; ++c) axis[c].push_back(p[c]);
        }
        if (std::isfinite(pa.w)) {
            aMin = std::min(aMin, pa.w);
            aMax = std::max(aMax, pa.w);
            aSum += pa.w;
            if (pa.w < 1.0f / 255.0f) ++aLow;
        }
    }

    s.bboxMin = lo;
    s.bboxMax = hi;
    for (int c = 0; c < 3; ++c) {
        s.robustMin[c] = percentile(axis[c], SceneStats::kRobustPercentile);
        s.robustMax[c] = percentile(axis[c], 1.0 - SceneStats::kRobustPercentile);
    }
    s.alphaMin = aMin;
    s.alphaMax = aMax;
    s.alphaMean = float(aSum / double(n));
    s.alphaBelowVisible = aLow;
}

void computeSceneStats(GaussianScene& scene) {
    SceneStats& s = scene.stats;
    s = SceneStats{};
    const size_t n = scene.count;
    if (n == 0) return;
    computePositionStats(scene.posAlpha.data(), n, s);

    const float inf = std::numeric_limits<float>::infinity();
    float sMin = inf, sMax = -inf, qMin = inf, qMax = -inf;
    std::vector<float> maxSigma;
    maxSigma.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const glm::vec3 sigma(scene.scales[i]);
        if (std::isfinite(sigma.x) && std::isfinite(sigma.y) && std::isfinite(sigma.z)) {
            const float mn = std::min({sigma.x, sigma.y, sigma.z});
            const float mx = std::max({sigma.x, sigma.y, sigma.z});
            sMin = std::min(sMin, mn);
            sMax = std::max(sMax, mx);
            maxSigma.push_back(mx);
        }
        const float qn = glm::length(scene.rawRotation[i]);
        if (std::isfinite(qn)) {
            qMin = std::min(qMin, qn);
            qMax = std::max(qMax, qn);
        }
    }
    s.sigmaMin = sMin;
    s.sigmaMax = sMax;
    s.sigmaMaxMedian = percentile(maxSigma, 0.5);
    s.quatNormMin = qMin;
    s.quatNormMax = qMax;
}

}  // namespace gsr
