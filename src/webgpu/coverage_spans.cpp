#include "webgpu/coverage_spans.h"

#include <algorithm>

namespace gsr::web {

uint32_t encodeCoverageSpans(const Coverage& cv, int S, SpanRecord& w) {
    w.fill(0);
    const auto [r0, r1] = coverageRows(cv, S);
    const uint32_t nRows = uint32_t(r1 - r0);
    const int bx = cv.bins.x0;
    const uint32_t width = uint32_t(std::max(0, cv.bins.x1 - cv.bins.x0));
    w[0] = uint32_t(bx) | (uint32_t(std::max(r0, 0)) << 16);
    if (nRows == 0) return 0;
    if (width > 255) {
        w[1] = nRows | 0x8000u;
        w[2] = width;
        return nRows * width;
    }
    uint32_t count = 0, tailLo = 255, tailHi = 0;
    for (uint32_t i = 0; i < nRows; ++i) {
        const auto [x0, x1] = coverageRowSpan(cv, r0 + int(i), S);
        const uint32_t len = uint32_t(x1 - x0);
        const uint32_t start = len > 0 ? uint32_t(x0 - bx) : 0u;  // empty rows: (0, 0) (start may be > 255)
        if (i < kCoverageSpanRows) {
            w[2 + i / 2] |= (start | (len << 8)) << (16 * (i & 1));
            count += len;
        } else if (len > 0) {
            tailLo = std::min(tailLo, start);
            tailHi = std::max(tailHi, start + len);
        }
    }
    uint32_t tailStart = 0, tailLen = 0;
    if (tailHi > tailLo) {
        tailStart = tailLo;
        tailLen = tailHi - tailLo;
    }
    if (nRows > kCoverageSpanRows) count += (nRows - kCoverageSpanRows) * tailLen;
    w[1] = nRows | (tailStart << 16) | (tailLen << 24);
    return count;
}

uint32_t spanRecordCount(const uint32_t* w) {
    const uint32_t nRows = w[1] & 0x7FFFu;
    if (w[1] & 0x8000u) return nRows * w[2];
    uint32_t count = 0;
    for (uint32_t i = 0; i < std::min(nRows, kCoverageSpanRows); ++i) count += (w[2 + i / 2] >> (16 * (i & 1) + 8)) & 0xFFu;
    if (nRows > kCoverageSpanRows) count += (nRows - kCoverageSpanRows) * (w[1] >> 24);
    return count;
}

}  // namespace gsr::web
