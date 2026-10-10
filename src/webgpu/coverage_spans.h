// CPU twin of shaders/wgsl/coverage_spans.wgsl: a splat's tile coverage stored as integers (web
// phase W2, DESIGN.md section 3 option d). Used by the in-page GPU tests (expected records from the
// CPU coverage of src/splat/tile_coverage.h; expected per-tile lists from the GPU's own records) and
// by the coverage analysis (row histogram, list overhead of the stored tail).
//
// Record (kCoverageSpanWords u32):
//   w0   bx | r0 << 16                       first bin column of the clipping rectangle, first row
//   w1   nRows | wide << 15 | tailStart << 16 | tailLen << 24
//   w2.. rows 0..K-1, 16 bits each: start (relative to bx) | length << 8
// Rows K.. use the tail span (union of their exact spans); wide mode (rectangle wider than 255
// bins) uses the full rectangle for every row, with its width in w2.
#pragma once

#include "splat/tile_coverage.h"
#include "webgpu/shader_constants.h"

#include <array>
#include <cstdint>

namespace gsr::web {

using SpanRecord = std::array<uint32_t, kCoverageSpanWords>;

// The record of a coverage (encodeCoverageSpans); returns the tile count it enumerates.
uint32_t encodeCoverageSpans(const Coverage& cv, int binSize, SpanRecord& out);

// Number of tiles a record enumerates (spanRecordCount).
uint32_t spanRecordCount(const uint32_t* w);

// Calls fn(column, row) for every tile of a record, row by row, left to right: the order of
// spanRecordTile(w, r) for r = 0, 1, ...
template <typename Fn>
void forEachSpanTile(const uint32_t* w, Fn&& fn) {
    const uint32_t bx = w[0] & 0xFFFFu, r0 = w[0] >> 16;
    const uint32_t nRows = w[1] & 0x7FFFu;
    if (w[1] & 0x8000u) {
        for (uint32_t i = 0; i < nRows; ++i)
            for (uint32_t x = 0; x < w[2]; ++x) fn(bx + x, r0 + i);
        return;
    }
    for (uint32_t i = 0; i < nRows; ++i) {
        uint32_t start, len;
        if (i < kCoverageSpanRows) {
            const uint32_t bits = (w[2 + i / 2] >> (16 * (i & 1))) & 0xFFFFu;
            start = bits & 0xFFu;
            len = bits >> 8;
        } else {
            start = (w[1] >> 16) & 0xFFu;
            len = w[1] >> 24;
        }
        for (uint32_t x = 0; x < len; ++x) fn(bx + start + x, r0 + i);
    }
}

}  // namespace gsr::web
