// Port of shaders/scan_common.glsl + scan_reduce.comp + scan_blocks.comp + scan_downsweep.comp (web
// phase W2): GPU exclusive prefix sum, three entry points of one module. C++ side:
// src/webgpu/wgpu_scan.cpp (the web GpuScan).
//
//   output[j] = value(0) + ... + value(j - 1)      for j < count,   total = sum of all
//
// "Reduce-then-scan" in three dispatches over blocks of kScanBlockSize elements:
//   1. scan_reduce     one workgroup per block: blockSums[b] = sum of the block's values
//   2. scan_blocks     ONE workgroup: exclusive scan of blockSums in place (+ the total)
//   3. scan_downsweep  one workgroup per block: scan of the block + blockSums[b] -> output
//
// The element count lives in a GPU buffer (control[countOffset], clamped to maxCount), so the CPU
// records the scan without knowing it: it dispatches enough workgroups for maxCount and the
// workgroups past the actual count exit at once. Differences from the GLSL: push constants are a
// uniform buffer; the count source and the total destination are one read_write binding (WebGPU
// forbids binding one buffer as read-only and writable storage in the same dispatch); the count is
// made workgroup-uniform with workgroupUniformLoad (WGSL rejects barriers in loops whose bounds
// come from read_write storage); workgroups may come as a 2D grid (65535 limit).
//
// value(j) (params.flags):
//   default            input[j]
//   kScanFlagGather    input[index[j]]  (e.g. "tiles touched" of the j-th DEPTH-SORTED splat)
//   kScanFlagNonZero   1 if the value is != 0, else 0 (visibility flags: order-preserving compaction)

#ifdef GSR_SUBGROUPS
enable subgroups;
#endif

#include "workgroup_scan.wgsl"

const kScanItemsPerThread: u32 = 8u;
const kScanBlockSize: u32 = 2048u;  // kScanThreads * kScanItemsPerThread

const kScanFlagGather: u32 = 1u;
const kScanFlagNonZero: u32 = 2u;

// C++ twin: ScanParams in src/webgpu/wgpu_scan.cpp.
struct ScanParams {
    countOffset: u32,  // index (in u32) of the element count in `control`
    maxCount: u32,     // capacity: the count is clamped to it
    totalOffset: u32,  // index (in u32) in `control` where scan_blocks writes the total
    flags: u32,        // kScanFlag*
}

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read> scanInput: array<u32>;
@group(0) @binding(2) var<storage, read> scanIndex: array<u32>;
@group(0) @binding(3) var<storage, read_write> scanOutput: array<u32>;
@group(0) @binding(4) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(5) var<storage, read_write> control: array<u32>;

var<workgroup> sCount: u32;

// The element count, the same value in every thread (workgroup-uniform).
fn scanCount() -> u32 {
    if (gLane == 0u) {
        sCount = min(control[params.countOffset], params.maxCount);
    }
    return workgroupUniformLoad(&sCount);
}

fn scanNumBlocks(count: u32) -> u32 {
    return (count + kScanBlockSize - 1u) / kScanBlockSize;
}

fn scanValue(j: u32) -> u32 {
    var v: u32;
    if ((params.flags & kScanFlagGather) != 0u) {
        v = scanInput[scanIndex[j]];
    } else {
        v = scanInput[j];
    }
    if ((params.flags & kScanFlagNonZero) != 0u) {
        v = select(0u, 1u, v != 0u);
    }
    return v;
}

// Step 1: blockSums[block] = sum of the block's values.
@compute @workgroup_size(256)
fn scan_reduce(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    initLanes(lanes);
    let count = scanCount();
    let block = linearWorkgroup(wid, nwg);
    let blockStart = block * kScanBlockSize;
    if (blockStart >= count) {
        return;  // uniform: the workgroup is past the end
    }
    // Coalesced: in iteration i the workgroup reads 256 consecutive elements.
    var sum = 0u;
    for (var i = 0u; i < kScanItemsPerThread; i++) {
        let j = blockStart + i * kScanThreads + gLane;
        if (j < count) {
            sum += scanValue(j);
        }
    }
    let r = workgroupExclusiveScan(sum);  // only the total is needed here
    if (gLane == 0u) {
        blockSums[block] = r.total;
    }
}

// Step 2 (one workgroup): blockSums -> its exclusive prefix sum in place, total -> control.
@compute @workgroup_size(256)
fn scan_blocks(lanes: LaneInput) {
    initLanes(lanes);
    let numBlocks = scanNumBlocks(scanCount());
    var running = 0u;
    for (var chunk = 0u; chunk < numBlocks; chunk += kScanThreads) {
        let b = chunk + gLane;
        var v = 0u;
        if (b < numBlocks) {
            v = blockSums[b];
        }
        let r = workgroupExclusiveScan(v);
        if (b < numBlocks) {
            blockSums[b] = running + r.exclusive;
        }
        running += r.total;
    }
    if (gLane == 0u) {
        control[params.totalOffset] = running;  // 0 for an empty input
    }
}

// Step 3: thread t owns the kScanItemsPerThread CONSECUTIVE elements [t*8, t*8 + 8) of the block;
// a workgroup scan of the per-thread sums gives each thread the sum before its run, then it walks
// the run sequentially.
@compute @workgroup_size(256)
fn scan_downsweep(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    initLanes(lanes);
    let count = scanCount();
    let block = linearWorkgroup(wid, nwg);
    let blockStart = block * kScanBlockSize;
    if (blockStart >= count) {
        return;  // uniform
    }
    let first = blockStart + gLane * kScanItemsPerThread;
    var values: array<u32, kScanItemsPerThread>;
    var sum = 0u;
    for (var i = 0u; i < kScanItemsPerThread; i++) {
        let j = first + i;
        if (j < count) {
            values[i] = scanValue(j);
        } else {
            values[i] = 0u;
        }
        sum += values[i];
    }
    var running = blockSums[block] + workgroupExclusiveScan(sum).exclusive;
    for (var i = 0u; i < kScanItemsPerThread; i++) {
        let j = first + i;
        if (j < count) {
            scanOutput[j] = running;
        }
        running += values[i];
    }
}
