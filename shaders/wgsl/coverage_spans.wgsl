// Stored tile coverage (web phase W2, new; DESIGN.md section 3 option d). Replaces native's
// `precise` count/write agreement: the preprocess evaluates the float coverage of
// tile_coverage.wgsl ONCE and stores it as integers; the count (tileCounts) is computed from those
// integers, and the duplicate pass enumerates the same integers. Agreement holds by construction.
// C++ twin (tests, CPU construction of the expected lists): src/webgpu/coverage_spans.h.
//
// Record of one splat: kSpanWords u32 (kSpanRows = K rows stored exactly):
//   w0          bx (bits 0-15: first bin column of the clipping rectangle) | r0 << 16 (first row)
//   w1          nRows (bits 0-14) | wide flag (bit 15) | tailStart << 16 | tailLen << 24
//   w2 ..       rows 0 .. K-1 of the coverage, 16 bits each (two per word, row i in word 2 + i/2,
//               bits 16 (i & 1) ..): start (8 bits, relative to bx) | length << 8
// Rows K .. nRows-1 all use the TAIL span [bx + tailStart, + tailLen): the union of their exact
// spans, a superset (extra tiles only add (pixel, splat) pairs that every pixel skips with
// alpha < 1/255, so the image is unchanged; only the list grows, see docs/web/W2.md).
// Wide mode (clipping rectangle wider than 255 bins, only at > 4K widths): every row uses the
// full rectangle row [bx, bx + w2).

#include "splat_common.wgsl"
#include "tile_coverage.wgsl"

const kSpanRows: u32 = 8u;                     // C++: kCoverageSpanRows
const kSpanWords: u32 = 2u + kSpanRows / 2u;   // C++: kCoverageSpanWords
const kSpanWideFlag: u32 = 0x8000u;

struct SpanRecord {
    w: array<u32, kSpanWords>,
    count: u32,  // tiles the record enumerates (the splat's tileCounts entry)
}

// The record of a coverage (bin tile size S). Rows are walked in order; each row's span comes from
// coverageRowSpan (float math, evaluated once here).
fn encodeCoverageSpans(cv: Coverage, S: i32) -> SpanRecord {
    var rec: SpanRecord;  // zero-initialized
    let rows = coverageRows(cv, S);
    let nRows = u32(rows.y - rows.x);
    let bx = cv.bins.x;
    let width = u32(max(0, cv.bins.z - cv.bins.x));
    rec.w[0] = u32(bx) | (u32(max(rows.x, 0)) << 16u);
    if (nRows == 0u) {
        return rec;
    }
    if (width > 255u) {  // wide: full rectangle rows
        rec.w[1] = nRows | kSpanWideFlag;
        rec.w[2] = width;
        rec.count = nRows * width;
        return rec;
    }
    var count = 0u;
    var tailLo = 255u;
    var tailHi = 0u;
    for (var i = 0u; i < nRows; i++) {
        let span = coverageRowSpan(cv, rows.x + i32(i), S);
        let len = u32(span.y - span.x);
        // An empty row may report its (clipped) start anywhere, even far right of the rectangle:
        // store it as (0, 0), or a start > 255 would spill into the length bits.
        let start = select(0u, u32(span.x - bx), len > 0u);
        if (i < kSpanRows) {
            rec.w[2u + i / 2u] |= (start | (len << 8u)) << (16u * (i & 1u));
            count += len;
        } else if (len > 0u) {
            tailLo = min(tailLo, start);
            tailHi = max(tailHi, start + len);
        }
    }
    var tailStart = 0u;
    var tailLen = 0u;
    if (tailHi > tailLo) {
        tailStart = tailLo;
        tailLen = tailHi - tailLo;
    }
    if (nRows > kSpanRows) {
        count += (nRows - kSpanRows) * tailLen;
    }
    rec.w[1] = nRows | (tailStart << 16u) | (tailLen << 24u);
    rec.count = count;
    return rec;
}

// Number of tiles a stored record enumerates (integer math only).
fn spanRecordCount(w: array<u32, kSpanWords>) -> u32 {
    let nRows = w[1] & 0x7FFFu;
    if ((w[1] & kSpanWideFlag) != 0u) {
        return nRows * w[2];
    }
    var count = 0u;
    for (var i = 0u; i < min(nRows, kSpanRows); i++) {
        count += (w[2u + i / 2u] >> (16u * (i & 1u) + 8u)) & 0xFFu;
    }
    if (nRows > kSpanRows) {
        count += (nRows - kSpanRows) * (w[1] >> 24u);
    }
    return count;
}

// The r-th tile (row by row, left to right) of a stored record as (column, row) in bin tiles;
// (-1, -1) if r >= the record's count.
fn spanRecordTile(w: array<u32, kSpanWords>, r: u32) -> vec2<i32> {
    let bx = w[0] & 0xFFFFu;
    let r0 = w[0] >> 16u;
    let nRows = w[1] & 0x7FFFu;
    if ((w[1] & kSpanWideFlag) != 0u) {
        let width = w[2];
        if (width == 0u || r / width >= nRows) {
            return vec2<i32>(-1, -1);
        }
        return vec2<i32>(i32(bx + r % width), i32(r0 + r / width));
    }
    var before = 0u;
    for (var i = 0u; i < min(nRows, kSpanRows); i++) {
        let bits = (w[2u + i / 2u] >> (16u * (i & 1u))) & 0xFFFFu;
        let len = bits >> 8u;
        if (r < before + len) {
            return vec2<i32>(i32(bx + (bits & 0xFFu) + (r - before)), i32(r0 + i));
        }
        before += len;
    }
    let tailLen = w[1] >> 24u;
    if (nRows > kSpanRows && tailLen > 0u) {
        let q = r - before;
        let row = kSpanRows + q / tailLen;
        if (row < nRows) {
            return vec2<i32>(i32(bx + ((w[1] >> 16u) & 0xFFu) + q % tailLen), i32(r0 + row));
        }
    }
    return vec2<i32>(-1, -1);
}
