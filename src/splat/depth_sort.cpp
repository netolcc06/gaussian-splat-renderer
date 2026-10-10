#include "splat/depth_sort.h"

#include "splat/splat_math.h"

#include <array>
#include <chrono>
#include <cstring>

namespace gsr {

namespace {
double msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

void DepthSorter::sort(const glm::vec4* posAlpha, uint32_t count, const glm::mat4& viewInria) {
    const auto t0 = std::chrono::steady_clock::now();

    // 1) Keys. Gaussians at or behind the near plane never render: drop them here so the
    //    sort (and later the draw) only handles the visible candidates.
    m_keys.resize(count);
    m_values.resize(count);
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const float z = viewDepth(viewInria, glm::vec3(posAlpha[i]));
        if (!(z > kNearCullDepth)) continue;  // also drops NaN
        uint32_t bits;
        std::memcpy(&bits, &z, sizeof(bits));
        m_keys[n] = bits;
        m_values[n] = i;
        ++n;
    }
    m_keys.resize(n);
    m_values.resize(n);
    const auto t1 = std::chrono::steady_clock::now();
    depthMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2) LSD radix sort, 4 x 8 bits (see radixSortPairs).
    passesRun = radixSortPairs(m_keys, m_values, m_keysTmp, m_valuesTmp);
    sortMs = msSince(t1);
}

uint32_t DepthSorter::radixSortPairs(std::vector<uint32_t>& keys, std::vector<uint32_t>& values,
                                     std::vector<uint32_t>& keysTmp, std::vector<uint32_t>& valuesTmp) {
    // Each pass is a stable counting sort on one byte, so after the last pass the order
    // is by the full key with ties in input order.
    const uint32_t n = uint32_t(keys.size());
    keysTmp.resize(n);
    valuesTmp.resize(n);
    uint32_t passes = 0;
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        std::array<uint32_t, 256> histogram{};
        for (uint32_t i = 0; i < n; ++i) histogram[(keys[i] >> shift) & 0xFFu]++;
        // Every key has the same digit: this pass would not move anything.
        if (n == 0 || histogram[(keys[0] >> shift) & 0xFFu] == n) continue;

        // Exclusive prefix sum: first output slot of each digit.
        uint32_t sum = 0;
        for (uint32_t& h : histogram) {
            const uint32_t c = h;
            h = sum;
            sum += c;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t dst = histogram[(keys[i] >> shift) & 0xFFu]++;
            keysTmp[dst] = keys[i];
            valuesTmp[dst] = values[i];
        }
        keys.swap(keysTmp);
        values.swap(valuesTmp);
        ++passes;
    }
    return passes;
}

}  // namespace gsr
