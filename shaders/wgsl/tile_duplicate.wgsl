// Port of shaders/tile_duplicate.comp, binning mode + GSR_DUPLICATE_PER_ENTRY (the native default;
// web phase W2): DUPLICATION. One thread per OUTPUT entry k of the duplicated list (indirect
// dispatch over min(M, capacity) entries). The list of visible splats is already in (depth, index)
// order (depth sort), so a STABLE sort by tile afterwards gives per-tile lists in (depth, index)
// order: key = tile id only.
//
// The thread finds its splat by a binary search in the offsets (the last j with offsets[j] <= k: an
// "upper bound" search; splats with zero tiles share the next one's offset and are skipped), then
// its tile as entry r = k - offsets[j] of the splat's STORED coverage (coverage_spans.wgsl): integer
// math only, no float recomputation. That is the web replacement for native's `precise`: the count
// (preprocess) and this enumeration read the same integers, so they agree by construction. The
// thread of a splat's first entry still re-checks the total against tileCounts and counts a
// disagreement (must stay 0; copied into the control block after this pass); a thread that finds no
// tile writes the sentinel tile id = #tiles, which sorts last and which tile_ranges ignores.
//
// The control block is READ-ONLY here, and the mismatch counter has its own small buffer: WGSL
// atomics on the control block would turn every thread's read of `entries` / `visible` into an
// atomic read-modify-write (Tint lowers atomicLoad to InterlockedOr on D3D12), all on two
// addresses: millions of serialized atomics per frame.
//
// Entries at offsets >= capacity are not written (the overflow is counted by tile_setup).

#include "workgroup_scan.wgsl"
#include "tile_common.wgsl"
#include "coverage_spans.wgsl"

// C++ twin: DuplicateParams in src/webgpu/wgpu_tile_renderer.cpp.
struct DuplicateParams {
    tilesX: u32,     // the bin tile grid
    tileCount: u32,  // tilesX * tilesY = the sentinel tile id
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> sortedOrder: array<u32>;  // depth-sorted visible indices
@group(0) @binding(1) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;      // exclusive scan, depth order
@group(0) @binding(3) var<storage, read> spans: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;    // tile id
@group(0) @binding(5) var<storage, read_write> valuesOut: array<u32>;  // gaussian index
@group(0) @binding(6) var<storage, read> control: array<u32>;
@group(0) @binding(7) var<uniform> params: DuplicateParams;
// [0] splats whose stored record disagrees with tileCounts; [1..3] one example (splat, record count,
// tileCounts) for diagnostics. Copied to the control block words kCtlCoverageMismatch.. afterwards.
@group(0) @binding(8) var<storage, read_write> mismatch: array<atomic<u32>, 4>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) index: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = linearWorkgroup(wid, nwg) * kTileGroupSize + index;
    if (k >= control[kCtlEntries]) {
        return;
    }
    let n = control[kCtlVisible];  // offsets of the depth-sorted visible splats
    // Last j with offsets[j] <= k. offsets[0] = 0 <= k, so the answer is in [lo, hi).
    var lo = 0u;
    var hi = n;
    while (hi - lo > 1u) {
        let mid = (lo + hi) >> 1u;
        if (offsets[mid] <= k) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let g = sortedOrder[lo];
    let r = k - offsets[lo];  // my entry within the splat's tiles, row by row

    var w: array<u32, kSpanWords>;
    for (var i = 0u; i < kSpanWords; i++) {
        w[i] = spans[g * kSpanWords + i];
    }
    let xy = spanRecordTile(w, r);
    var tile = params.tileCount;  // sentinel: not found (never seen)
    if (xy.x >= 0) {
        tile = u32(xy.y) * params.tilesX + u32(xy.x);
    }
    // Count/write agreement check, once per splat (by its first entry).
    if (r == 0u) {
        let recordCount = spanRecordCount(w);
        if (recordCount != tileCounts[g] && atomicAdd(&mismatch[0], 1u) == 0u) {  // the first one: its example
            atomicStore(&mismatch[1], g);
            atomicStore(&mismatch[2], recordCount);
            atomicStore(&mismatch[3], tileCounts[g]);
        }
    }
    keysOut[k] = tile;
    valuesOut[k] = g;
}
