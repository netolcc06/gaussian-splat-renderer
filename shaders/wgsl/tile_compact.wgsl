// Port of shaders/tile_compact.comp (web phase W2): ORDER-PRESERVING stream compaction of the
// visible splats. One thread per gaussian i. The exclusive scan of the visibility flags (scan.wgsl
// with kScanFlagNonZero over tileCounts) gave slot[i] = number of visible gaussians with a smaller
// index, so the compact list is in INDEX order, and the stable depth sort that follows orders ties
// by index, exactly like the reference.
//
//   depthKeys[slot[i]]   = float bits of the view depth (positive, so they sort as u32)
//   depthValues[slot[i]] = i
// Only the full 48-byte Splat is ported (no packed depths array).

#include "workgroup_scan.wgsl"
#include "splat_common.wgsl"
#include "tile_common.wgsl"

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> depthValues: array<u32>;
@group(0) @binding(5) var<storage, read> control: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) index: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linearWorkgroup(wid, nwg) * kTileGroupSize + index;
    if (i >= control[kCtlGaussianCount]) {
        return;  // past the end
    }
    if (tileCounts[i] == 0u) {
        return;  // culled
    }
    let slot = slots[i];
    depthKeys[slot] = bitcast<u32>(splats[i].depth);
    depthValues[slot] = i;
}
