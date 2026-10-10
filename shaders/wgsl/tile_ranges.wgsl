// Port of shaders/tile_ranges.comp (web phase W2): per-tile ranges (INRIA identifyTileRanges).
// After the sort, the entries of each tile are contiguous; one thread per entry k looks at its
// neighbour:
//
//   tile(k-1) != tile(k)  ->  ranges[tile(k-1)].y = k   and   ranges[tile(k)].x = k
//
// plus the two ends of the list. Tiles without any entry keep the (0, 0) of the per-frame clear.
// Launched with dispatchWorkgroupsIndirect (the 2D-clamped grid of tile_setup).

#include "workgroup_scan.wgsl"
#include "tile_common.wgsl"

// C++ twin: RangesParams in src/webgpu/wgpu_tile_renderer.cpp.
struct RangesParams {
    tileCount: u32,  // tiles in the grid: keys >= tileCount are the sentinel
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> tileKeys: array<u32>;  // tile id per entry
@group(0) @binding(1) var<storage, read_write> ranges: array<vec2<u32>>;  // [start, end) per tile
@group(0) @binding(2) var<storage, read> control: array<u32>;
@group(0) @binding(3) var<uniform> params: RangesParams;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) index: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = linearWorkgroup(wid, nwg) * kTileGroupSize + index;
    let n = control[kCtlEntries];
    if (k >= n) {
        return;
    }
    let tile = tileKeys[k];
    if (tile >= params.tileCount) {
        // Sentinel (sorted to the end, never seen): closes the last real tile's range.
        if (k > 0u && tileKeys[k - 1u] < params.tileCount) {
            ranges[tileKeys[k - 1u]].y = k;
        }
        return;
    }
    if (k == 0u) {
        ranges[tile].x = 0u;
    } else {
        let prev = tileKeys[k - 1u];
        if (prev != tile) {
            if (prev < params.tileCount) {
                ranges[prev].y = k;
            }
            ranges[tile].x = k;
        }
    }
    if (k == n - 1u) {
        ranges[tile].y = n;
    }
}
