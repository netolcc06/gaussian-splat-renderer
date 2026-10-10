// Port of shaders/tile_setup.comp (web phase W2), one thread: clamp the number of (tile, splat)
// entries to the list capacity, record the overflow, and write the indirect dispatch arguments of
// the passes over the list (duplicate, ranges: one thread per entry). NEW on the web: the workgroup
// count is split into a 2D grid when it exceeds maxComputeWorkgroupsPerDimension (65535); WebGPU
// would otherwise reject or skip the indirect dispatch. The passes compute their linear workgroup
// index from @builtin(num_workgroups) (linearWorkgroup in workgroup_scan.wgsl).

#include "tile_common.wgsl"

// C++ twin: SetupParams in src/webgpu/wgpu_tile_renderer.cpp.
struct SetupParams {
    capacity: u32,    // entries the duplicated list can hold
    maxGroupsX: u32,  // maxComputeWorkgroupsPerDimension (tests: smaller)
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> control: array<u32>;
@group(0) @binding(1) var<uniform> params: SetupParams;
// The same arguments again in their own buffer: the duplicate pass binds the control block as
// writable storage (its mismatch counter), and WebGPU forbids using one buffer as writable storage
// and as the indirect buffer of the same dispatch (Vulkan allows it; native reads them from
// TileControl).
@group(0) @binding(2) var<storage, read_write> listArgs: array<u32, 3>;

@compute @workgroup_size(1)
fn main() {
    let m = control[kCtlDuplicates];
    let entries = min(m, params.capacity);
    control[kCtlEntries] = entries;
    control[kCtlOverflow] = m - entries;
    let groups = (entries + kTileGroupSize - 1u) / kTileGroupSize;
    let gx = min(groups, params.maxGroupsX);
    var gy = 1u;
    if (gx > 0u) {
        gy = (groups + gx - 1u) / gx;
    }
    control[kCtlListGroupsX] = gx;  // 0: an empty dispatch
    control[kCtlListGroupsY] = gy;
    control[kCtlListGroupsZ] = 1u;
    listArgs[0] = gx;
    listArgs[1] = gy;
    listArgs[2] = 1u;
}
