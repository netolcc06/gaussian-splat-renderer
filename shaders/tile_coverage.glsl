// Phase 6: which screen tiles a splat must be binned into ("bounds", `--bounds`).
// C++ twin (same expressions, used by the CPU reference and the tests):
// src/splat/tile_coverage.h/.cpp. Keep the two in sync. Derivation and the exactness
// argument: docs/notes/phase6.md section 1.
//
//   rect         (Phase 2-5) every tile of INRIA's 3-sigma tile rectangle (getRect, 16-px tiles)
//   tight        the rectangle INTERSECTED with the opacity-aware cutoff ellipse: a tile is
//                kept only if the ellipse { e : q(e) <= t } overlaps it, where
//                  q(e) = A ex^2 + 2 B ex ey + C ey^2   (the conic, e = pixel - centre)
//                  t    = 2 ln(255 o) + margin          (o * exp(-q/2) >= 1/255  <=>  q <= 2 ln(255 o))
//                Every dropped (pixel, splat) pair has alpha < 1/255, which the render loop
//                skips anyway: the image is IDENTICAL to rect.
//   cutoff-only  the cutoff ellipse alone, NOT clipped to the 3-sigma rectangle (only to the
//                screen). Not INRIA: opaque splats (o > e^4.5/255 = 0.35) reach beyond their
//                3-sigma rectangle. Educational, measured against the reference.
//
// The tile test is done row by row: the intersection of the (convex) ellipse with the
// horizontal strip of one tile row is convex, so the tiles it overlaps in that row are ONE
// contiguous run [x0, x1). rowSpan() computes that run exactly (up to a conservative pad):
// the extreme x of the ellipse inside the strip is reached either at the ellipse's own
// leftmost/rightmost point (if that lies inside the strip) or on the strip's edge.
//
// Count/write agreement: the preprocess (count) and the duplicate pass (write) evaluate
// these functions on the same stored inputs (centre, conic, opacity as the render pass sees
// them). Every float operation is `precise` (no fused multiply-add, no reassociation), so
// both pipelines compute the same bits; tile_duplicate.comp also re-checks the total and
// counts disagreements (TileControl.coverageMismatch, 0 in every test and run).

#ifndef TILE_COVERAGE_GLSL
#define TILE_COVERAGE_GLSL

const uint kBoundsRect = 0u;
const uint kBoundsTight = 1u;
const uint kBoundsCutoff = 2u;  // cutoff-only

// The cutoff ellipse of one splat.
struct CutoffEllipse {
    vec2 c;       // centre (pixel coordinates)
    float A, B, C, D;  // conic and its determinant A C - B^2
    float t;      // q threshold incl. the numerical margin (< 0: no pixel can reach 1/255)
    float exPad;  // half extent in x / y of { q <= t }, padded
    float eyPad;
    bool valid;   // false: the conic is not positive definite (fall back to the rectangle)
};

CutoffEllipse makeCutoffEllipse(vec2 center, vec3 conic, float opacity) {
    CutoffEllipse e;
    e.c = center;
    precise float A = conic.x;
    precise float B = conic.y;
    precise float C = conic.z;
    precise float D = A * C - B * B;
    e.A = A;
    e.B = B;
    e.C = C;
    e.D = D;
    e.t = -1.0;
    e.exPad = 0.0;
    e.eyPad = 0.0;
    e.valid = A > 0.0 && C > 0.0 && D > 0.0;
    if (!e.valid) return e;
    if (!(opacity >= kMinAlpha)) return e;  // alpha = min(.99, o exp(power)) <= o < 1/255 everywhere
    // q_max = 2 ln(255 o) >= 0. Margin: the render loop evaluates power = -q/2 in fp32 with
    // ~8 roundings on terms whose magnitude is bounded by K q / 2, K = 2 / (1 - |rho|),
    // rho = B / sqrt(A C) (K = 2 for a circle, large for needles); exp/log are a few ulps.
    // The margin covers all of it with a factor > 25 (phase6.md section 1).
    precise float qmax = 2.0 * log(255.0 * opacity);
    precise float rho = abs(B) / sqrt(A * C);
    precise float K = 2.0 / max(1.0 - rho, 1e-30);
    precise float t = qmax + (1e-3 + 2e-5 * K) * max(qmax, 1.0) + 2e-3;
    if (!(t < 1e30)) {  // degenerate (rho ~ 1): no useful ellipse, use the rectangle
        e.valid = false;
        return e;
    }
    e.t = t;
    precise float ex = sqrt(t * C / D);
    precise float ey = sqrt(t * A / D);
    precise float exPad = ex + (0.02 + 2e-3 * ex);
    precise float eyPad = ey + (0.02 + 2e-3 * ey);
    e.exPad = exPad;
    e.eyPad = eyPad;
    return e;
}

// floor(v / S) for a float holding an integer value, clamped so the int conversion is safe.
int floorDivClamped(float v, int S) { return int(floor(clamp(v, -65536.0, 1048576.0) / float(S))); }

// INRIA's 16-px tile rectangle [lo, hi) -> bin tiles of size S (S divides 16), clamped to the grid.
ivec4 rectToBins(ivec4 rect16, int S, ivec2 grid) {
    int f = 16 / S;
    return ivec4(rect16.x * f, rect16.y * f, min(rect16.z * f, grid.x), min(rect16.w * f, grid.y));
}

// Rows [lo, hi) (bin tiles) the ellipse can touch, intersected with `bins` (x, y, z, w = x0, y0, x1, y1).
ivec2 ellipseRows(CutoffEllipse e, int S, ivec4 bins) {
    if (e.t < 0.0) return ivec2(0, 0);
    precise float y0 = ceil(e.c.y - e.eyPad);
    precise float y1 = floor(e.c.y + e.eyPad);
    int lo = max(bins.y, floorDivClamped(y0, S));
    int hi = min(bins.w, floorDivClamped(y1, S) + 1);
    return ivec2(lo, max(lo, hi));
}

// The run of bin tiles [x0, x1) of tile row `row` that the ellipse overlaps, clipped to
// `bins` (empty: x1 <= x0). Pixel rows of the tile row: row*S .. row*S + S-1 (integers).
ivec2 ellipseRowSpan(CutoffEllipse e, int row, int S, ivec4 bins) {
    precise float ey0 = float(row * S) - e.c.y;
    precise float ey1 = float(row * S + S - 1) - e.c.y;
    precise float lo = max(ey0, -e.eyPad);
    precise float hi = min(ey1, e.eyPad);
    if (!(lo <= hi)) return ivec2(bins.x, bins.x);
    // The ellipse's rightmost point is at ey = -B ex / C, its leftmost at +B ex / C
    // (ex = its x half extent); inside the strip the extreme is at that ey clamped to it.
    precise float exTrue = sqrt(e.t * e.C / e.D);
    precise float eyRight = clamp(-e.B * exTrue / e.C, lo, hi);
    precise float eyLeft = clamp(e.B * exTrue / e.C, lo, hi);
    precise float At = e.A * e.t;
    precise float xRight = (-e.B * eyRight + sqrt(max(0.0, At - e.D * eyRight * eyRight))) / e.A;
    precise float xLeft = (-e.B * eyLeft - sqrt(max(0.0, At - e.D * eyLeft * eyLeft))) / e.A;
    precise float pad = 0.02 + 2e-3 * exTrue;
    precise float px0 = ceil(e.c.x + xLeft - pad);
    precise float px1 = floor(e.c.x + xRight + pad);
    int x0 = max(bins.x, floorDivClamped(px0, S));
    int x1 = min(bins.z, floorDivClamped(px1, S) + 1);
    return ivec2(x0, max(x0, x1));
}

// Everything the binning needs to enumerate a splat's tiles.
struct Coverage {
    ivec4 bins;        // the clipping rectangle in bin tiles (rect: also the result)
    CutoffEllipse e;
    bool useEllipse;   // false: every tile of `bins`
};

// rect16: INRIA's tile rectangle (16-px tiles); grid: the bin grid (ceil(size / S)).
Coverage makeCoverage(uint mode, ivec4 rect16, int S, ivec2 grid, vec2 center, vec3 conic, float opacity) {
    Coverage cv;
    cv.bins = mode == kBoundsCutoff ? ivec4(0, 0, grid) : rectToBins(rect16, S, grid);
    cv.useEllipse = false;
    cv.e.valid = false;
    cv.e.t = -1.0;
    if (mode != kBoundsRect) {
        cv.e = makeCutoffEllipse(center, conic, opacity);
        cv.useEllipse = cv.e.valid;
        // cutoff-only with an unusable conic: keep INRIA's rectangle (never the whole screen).
        if (!cv.e.valid && mode == kBoundsCutoff) cv.bins = rectToBins(rect16, S, grid);
    }
    return cv;
}

ivec2 coverageRows(Coverage cv, int S) {
    if (!cv.useEllipse) return ivec2(cv.bins.y, max(cv.bins.y, cv.bins.w));
    return ellipseRows(cv.e, S, cv.bins);
}

ivec2 coverageRowSpan(Coverage cv, int row, int S) {
    if (!cv.useEllipse) return ivec2(cv.bins.x, max(cv.bins.x, cv.bins.z));
    return ellipseRowSpan(cv.e, row, S, cv.bins);
}

// Number of bin tiles the splat is binned into (the preprocess's count).
uint coverageCount(Coverage cv, int S) {
    ivec2 rows = coverageRows(cv, S);
    if (!cv.useEllipse) return uint(max(0, cv.bins.z - cv.bins.x) * (rows.y - rows.x));
    uint n = 0u;
    for (int r = rows.x; r < rows.y; ++r) {
        ivec2 s = ellipseRowSpan(cv.e, r, S, cv.bins);
        n += uint(s.y - s.x);
    }
    return n;
}

#endif  // TILE_COVERAGE_GLSL
