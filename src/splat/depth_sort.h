// CPU depth sort: the "sort" stage of the Phase 2 baseline (Phase 3 replaces it with a
// GPU radix sort), also used by the CPU reference renderer so both produce the exact
// same order.
//
// Output: the indices of every gaussian in front of the near plane (view z > 0.2, the
// INRIA near cull), NEAREST FIRST. Equal depths keep ascending index order, which is
// what INRIA's (stable) radix sort over (tile | depth) keys yields within a tile.
//
// Algorithm: an LSD radix sort on 32-bit keys. For a positive IEEE-754 float the raw
// bit pattern is monotonic in the value, so the key is simply the depth's bits
// (after the near cull every depth is > 0.2, hence positive). Four stable passes of 8
// bits each; a pass whose digit is identical for every key (common for the top byte,
// since depths share an exponent range) is skipped.
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace gsr {

class DepthSorter {
public:
    // posAlpha: data-space means (xyz) of `count` gaussians; viewInria: data -> INRIA
    // camera space (depth = row 2). Results in order(); timings in the *Ms members.
    void sort(const glm::vec4* posAlpha, uint32_t count, const glm::mat4& viewInria);

    const std::vector<uint32_t>& order() const { return m_values; }
    // The (sorted) keys that go with order(): the IEEE bits of each depth.
    const std::vector<uint32_t>& keys() const { return m_keys; }

    // The radix core on its own: stable LSD sort of (key, value) pairs by key, 4 x 8 bits,
    // in place. `keysTmp`/`valuesTmp` are scratch (resized as needed). Returns the number
    // of passes that actually moved data (all-equal digits are skipped). Also the CPU
    // ground truth for the GPU radix sort (tests/test_gpu_sort.cpp).
    static uint32_t radixSortPairs(std::vector<uint32_t>& keys, std::vector<uint32_t>& values,
                                   std::vector<uint32_t>& keysTmp, std::vector<uint32_t>& valuesTmp);

    double depthMs = 0.0;  // computing keys (one dot product per gaussian) + near cull
    double sortMs = 0.0;   // radix passes
    uint32_t passesRun = 0;

private:
    std::vector<uint32_t> m_keys, m_values;
    std::vector<uint32_t> m_keysTmp, m_valuesTmp;
};

}  // namespace gsr
