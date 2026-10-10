// Port of shaders/tile_coverage.glsl (web phase W2): which screen tiles a splat is binned into.
// C++ twin: src/splat/tile_coverage.cpp (same expressions, same order). Derivation and the
// exactness argument: docs/notes/phase6.md section 1. Only the modes the web uses are ported:
// `rect` (INRIA's 3-sigma tile rectangle) as the fallback of `tight` (that rectangle intersected
// with the alpha >= 1/255 ellipse). `cutoff-only` is not ported.
//
// The big difference from native: WGSL has no `precise`, so two pipelines evaluating these
// functions on the same inputs may round differently (FMA contraction). Native relies on
// `precise` for the count (preprocess) and the write (duplicate) to agree. The web port
// evaluates them ONCE, in the preprocess, and stores the result as integers (coverage_spans.wgsl,
// DESIGN.md section 3, option d); the duplicate pass reads those integers. The margins below
// (> 25x the float error, phase6.md) keep every dropped pair at alpha < 1/255 whatever the
// compiler fuses, so the image stays exact.

const kBoundsRect: u32 = 0u;
const kBoundsTight: u32 = 1u;

struct CutoffEllipse {
    c: vec2<f32>,  // centre (pixel coordinates)
    A: f32,        // conic and its determinant A C - B^2
    B: f32,
    C: f32,
    D: f32,
    t: f32,        // q threshold incl. the numerical margin (< 0: no pixel can reach 1/255)
    exPad: f32,    // half extent in x / y of { q <= t }, padded
    eyPad: f32,
    valid: bool,   // false: the conic is not positive definite (fall back to the rectangle)
}

fn makeCutoffEllipse(center: vec2<f32>, conic: vec3<f32>, opacity: f32) -> CutoffEllipse {
    var e: CutoffEllipse;
    e.c = center;
    let A = conic.x;
    let B = conic.y;
    let C = conic.z;
    let D = A * C - B * B;
    e.A = A;
    e.B = B;
    e.C = C;
    e.D = D;
    e.t = -1.0;
    e.exPad = 0.0;
    e.eyPad = 0.0;
    e.valid = A > 0.0 && C > 0.0 && D > 0.0;
    if (!e.valid) {
        return e;
    }
    if (!(opacity >= kMinAlpha)) {
        return e;  // alpha = min(.99, o exp(power)) <= o < 1/255 everywhere
    }
    // q_max = 2 ln(255 o) >= 0, plus the margin of phase6.md section 1.
    let qmax = 2.0 * log(255.0 * opacity);
    let rho = abs(B) / sqrt(A * C);
    let K = 2.0 / max(1.0 - rho, 1e-30);
    let t = qmax + (1e-3 + 2e-5 * K) * max(qmax, 1.0) + 2e-3;
    if (!(t < 1e30)) {  // degenerate (rho ~ 1): no useful ellipse, use the rectangle
        e.valid = false;
        return e;
    }
    e.t = t;
    let ex = sqrt(t * C / D);
    let ey = sqrt(t * A / D);
    e.exPad = ex + (0.02 + 2e-3 * ex);
    e.eyPad = ey + (0.02 + 2e-3 * ey);
    return e;
}

// floor(v / S) for a float holding an integer value, clamped so the int conversion is safe.
fn floorDivClamped(v: f32, S: i32) -> i32 {
    return i32(floor(clamp(v, -65536.0, 1048576.0) / f32(S)));
}

// INRIA's 16-px tile rectangle [lo, hi) -> bin tiles of size S (S divides 16), clamped to the grid.
fn rectToBins(rect16: vec4<i32>, S: i32, grid: vec2<i32>) -> vec4<i32> {
    let f = 16 / S;
    return vec4<i32>(rect16.x * f, rect16.y * f, min(rect16.z * f, grid.x), min(rect16.w * f, grid.y));
}

// Rows [lo, hi) (bin tiles) the ellipse can touch, intersected with `bins` (x0, y0, x1, y1).
fn ellipseRows(e: CutoffEllipse, S: i32, bins: vec4<i32>) -> vec2<i32> {
    if (e.t < 0.0) {
        return vec2<i32>(0, 0);
    }
    let y0 = ceil(e.c.y - e.eyPad);
    let y1 = floor(e.c.y + e.eyPad);
    let lo = max(bins.y, floorDivClamped(y0, S));
    let hi = min(bins.w, floorDivClamped(y1, S) + 1);
    return vec2<i32>(lo, max(lo, hi));
}

// The run of bin tiles [x0, x1) of tile row `row` that the ellipse overlaps, clipped to `bins`
// (empty: x1 <= x0). Pixel rows of the tile row: row*S .. row*S + S-1 (integers).
fn ellipseRowSpan(e: CutoffEllipse, row: i32, S: i32, bins: vec4<i32>) -> vec2<i32> {
    let ey0 = f32(row * S) - e.c.y;
    let ey1 = f32(row * S + S - 1) - e.c.y;
    let lo = max(ey0, -e.eyPad);
    let hi = min(ey1, e.eyPad);
    if (!(lo <= hi)) {
        return vec2<i32>(bins.x, bins.x);
    }
    // The ellipse's rightmost point is at ey = -B ex / C, its leftmost at +B ex / C (ex = its x
    // half extent); inside the strip the extreme is at that ey clamped to it.
    let exTrue = sqrt(e.t * e.C / e.D);
    let eyRight = clamp(-e.B * exTrue / e.C, lo, hi);
    let eyLeft = clamp(e.B * exTrue / e.C, lo, hi);
    let At = e.A * e.t;
    let xRight = (-e.B * eyRight + sqrt(max(0.0, At - e.D * eyRight * eyRight))) / e.A;
    let xLeft = (-e.B * eyLeft - sqrt(max(0.0, At - e.D * eyLeft * eyLeft))) / e.A;
    let pad = 0.02 + 2e-3 * exTrue;
    let px0 = ceil(e.c.x + xLeft - pad);
    let px1 = floor(e.c.x + xRight + pad);
    let x0 = max(bins.x, floorDivClamped(px0, S));
    let x1 = min(bins.z, floorDivClamped(px1, S) + 1);
    return vec2<i32>(x0, max(x0, x1));
}

// Everything the binning needs to enumerate a splat's tiles.
struct Coverage {
    bins: vec4<i32>,  // the clipping rectangle in bin tiles (rect: also the result)
    e: CutoffEllipse,
    useEllipse: bool, // false: every tile of `bins`
}

// rect16: INRIA's tile rectangle (16-px tiles); grid: the bin grid (ceil(size / S)).
fn makeCoverage(mode: u32, rect16: vec4<i32>, S: i32, grid: vec2<i32>, center: vec2<f32>, conic: vec3<f32>,
                opacity: f32) -> Coverage {
    var cv: Coverage;
    cv.bins = rectToBins(rect16, S, grid);
    cv.useEllipse = false;
    cv.e.valid = false;
    cv.e.t = -1.0;
    if (mode != kBoundsRect) {
        cv.e = makeCutoffEllipse(center, conic, opacity);
        cv.useEllipse = cv.e.valid;
    }
    return cv;
}

fn coverageRows(cv: Coverage, S: i32) -> vec2<i32> {
    if (!cv.useEllipse) {
        return vec2<i32>(cv.bins.y, max(cv.bins.y, cv.bins.w));
    }
    return ellipseRows(cv.e, S, cv.bins);
}

fn coverageRowSpan(cv: Coverage, row: i32, S: i32) -> vec2<i32> {
    if (!cv.useEllipse) {
        return vec2<i32>(cv.bins.x, max(cv.bins.x, cv.bins.z));
    }
    return ellipseRowSpan(cv.e, row, S, cv.bins);
}
