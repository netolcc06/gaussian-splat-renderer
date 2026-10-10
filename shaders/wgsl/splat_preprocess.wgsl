// Port of shaders/splat_preprocess.comp, GSR_TILES build (web phase W2): one thread per gaussian,
// the INRIA preprocessCUDA plus the splat-tiles bookkeeping. Reads the scene (group 0,
// scene_decode.wgsl, any web scene format) and writes, for EVERY gaussian i:
//   tileCounts[i]   the number of bin tiles it is binned into (0 = culled)
// and for visible gaussians:
//   splats[i]       the 48-byte Splat (pixel centre, conic, opacity, SH colour, radius, depth)
//   spans[i]        its tile coverage as integers (coverage_spans.wgsl; NEW on the web)
// and counts every cull reason in the control block.
//
// Differences from the GLSL:
//   - The coverage (tight bounds: the 3-sigma rectangle clipped to the alpha >= 1/255 ellipse) is
//     evaluated once and STORED (DESIGN.md section 3); the count is the integer sum of the stored
//     spans, so the duplicate pass reproduces it exactly without `precise`.
//   - Cull counters: portable build = shared-memory atomics, then one global atomic per reason per
//     workgroup; GSR_SUBGROUPS = native's ballot count per subgroup.
//   - No packed render splats (p32/p24) and no StopThePop ray-depth parameters in web v1.
//   - The depth is computed with the CPU's operation order (viewDepth), but WGSL cannot forbid FMA
//     contraction, so it may differ from the CPU in the last bit (only near-ties can reorder; far
//     below the 8-bit output, DESIGN.md section 3).

#ifdef GSR_SUBGROUPS
enable subgroups;
#endif

#include "workgroup_scan.wgsl"
#include "splat_common.wgsl"
#include "scene_decode.wgsl"
#include "tile_common.wgsl"
#include "tile_coverage.wgsl"
#include "coverage_spans.wgsl"

// Bounds mode (tile_coverage.wgsl): tight by default, rect for tests and comparisons.
override kBoundsMode: u32 = 1u;
const kBinTileSize: i32 = 16;  // binning tile = render tile = INRIA's getRect tile

// C++ twin: PreprocessParams in src/webgpu/wgpu_tile_renderer.cpp (128 bytes).
struct PreprocessParams {
    viewInria: mat4x4<f32>,  // data space -> INRIA camera space
    camPos: vec4<f32>,       // xyz: camera centre in data space
    focal: vec2<f32>,        // pixels
    tanHalfFov: vec2<f32>,
    imageSize: vec2<u32>,    // pixels
    count: u32,              // gaussians
    shDegree: u32,           // bands to evaluate (<= stored degree)
    tiles16: vec2<u32>,      // INRIA's 16-px tile grid (getRect)
    grid: vec2<u32>,         // the bin grid
}

@group(1) @binding(0) var<uniform> params: PreprocessParams;
@group(1) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2) var<storage, read_write> tileCounts: array<u32>;
@group(1) @binding(3) var<storage, read_write> spans: array<u32>;
@group(1) @binding(4) var<storage, read_write> control: array<atomic<u32>>;

// The per-gaussian preprocess. Always sets depth and cullReason.
fn preprocessGaussian(i: u32) -> Splat {
    var s: Splat;  // zero-initialized

    // 1) Near-plane cull (the only frustum test INRIA does here). The depth is the sort key:
    //    the CPU's expression (viewDepth() in splat_math.h, same operation order).
    let pa = loadPosAlpha(i);
    let V = params.viewInria;
    var t = (V * vec4<f32>(pa.xyz, 1.0)).xyz;
    let depth = V[0][2] * pa.x + V[1][2] * pa.y + V[2][2] * pa.z + V[3][2];
    t.z = depth;
    s.depth = depth;
    // !(z > near) also drops a NaN depth, which would otherwise become a NaN sort key.
    if (!(t.z > kNearCullDepth)) {
        s.cullReason = kCullNearPlane;
        return s;
    }

    // 1b) Opacity cull: alpha = min(0.99, o exp(power)) <= o < 1/255 everywhere.
    if (pa.w < kMinAlpha) {
        s.cullReason = kCullLowOpacity;
        return s;
    }

    // 2) 3D covariance -> EWA 2D covariance -> conic and radius.
    let shape = loadShape(i);
    let cov3D = computeCov3D(shape.sigma, shape.q);
    let W = mat3x3<f32>(V[0].xyz, V[1].xyz, V[2].xyz);
    let cov2D = computeCov2D(t, cov3D, W, params.focal, params.tanHalfFov);
    let cr = conicAndRadius(cov2D);
    if (!cr.ok) {
        s.cullReason = kCullSingularCov;
        return s;
    }

    // 3) Pixel centre and tile rectangle; nothing on screen -> culled (the frustum cull).
    let center = projectToPixel(t, params.imageSize, params.tanHalfFov);
    let rect = computeTileRect(center, cr.radius, params.tiles16);
    if ((rect.z - rect.x) * (rect.w - rect.y) == 0) {
        s.center = center;
        s.cullReason = kCullOffScreen;
        return s;
    }

    // 4) SH colour, view direction from the camera to the mean in DATA space. Only the
    //    coefficients of the evaluated bands are fetched.
    let degree = min(params.shDegree, gaussianShDegree(i));
    let used = (degree + 1u) * (degree + 1u);
    let coeffs = loadSH(i, used);
    let dir = normalize(pa.xyz - params.camPos.xyz);

    s.center = center;
    s.conicOpacity = vec4<f32>(cr.conic, pa.w);
    s.color = evalSH(degree, coeffs, dir);
    s.radius = cr.radius;
    s.cullReason = kCullVisible;
    return s;
}

const kNoGaussian: u32 = 0xFFFFFFFFu;  // threads past the end of the scene

#ifndef GSR_SUBGROUPS
var<workgroup> sCullCounts: array<atomic<u32>, kCullReasonCount>;
#endif

@compute @workgroup_size(256)
fn main(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    initLanes(lanes);
#ifndef GSR_SUBGROUPS
    if (lanes.index < kCullReasonCount) {
        atomicStore(&sCullCounts[lanes.index], 0u);
    }
    workgroupBarrier();
#endif
    let i = linearWorkgroup(wid, nwg) * 256u + lanes.index;
    var reason = kNoGaussian;
    if (i < params.count) {
        let s = preprocessGaussian(i);
        reason = s.cullReason;
        var touched = 0u;
        if (reason == kCullVisible) {
            // The bin tiles of tile_coverage.wgsl from the centre, conic and opacity exactly as the
            // render pass will see them, stored once as integers.
            let rect = computeTileRect(s.center, s.radius, params.tiles16);
            let cv = makeCoverage(kBoundsMode, rect, kBinTileSize, vec2<i32>(params.grid), s.center,
                                  s.conicOpacity.xyz, s.conicOpacity.w);
            let rec = encodeCoverageSpans(cv, kBinTileSize);
            touched = rec.count;
            for (var w = 0u; w < kSpanWords; w++) {
                spans[i * kSpanWords + w] = rec.w[w];
            }
            splats[i] = s;  // culled splats are never read (tileCounts = 0)
        }
        tileCounts[i] = touched;
    }

    // Cull statistics. No early return above: every thread takes part.
#ifdef GSR_SUBGROUPS
    // One atomic per subgroup and reason, with the ballot count (native).
    for (var r = 0u; r < kCullReasonCount; r++) {
        let bits = countOneBits(subgroupBallot(reason == r));
        let n = bits.x + bits.y + bits.z + bits.w;
        if (subgroupElect() && n > 0u) {
            atomicAdd(&control[kCtlCullCounts + r], n);
        }
    }
#else
    // Shared-memory counts, then one global atomic per reason per workgroup.
    if (reason < kCullReasonCount) {
        atomicAdd(&sCullCounts[reason], 1u);
    }
    workgroupBarrier();
    if (lanes.index < kCullReasonCount) {
        let n = atomicLoad(&sCullCounts[lanes.index]);
        if (n > 0u) {
            atomicAdd(&control[kCtlCullCounts + lanes.index], n);
        }
    }
#endif
}
