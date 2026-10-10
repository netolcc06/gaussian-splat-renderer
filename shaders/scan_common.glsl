// GPU exclusive prefix sum (Phase 4), declarations shared by scan_reduce.comp,
// scan_blocks.comp and scan_downsweep.comp. C++ side: src/render/gpu_scan.cpp.
//
//   output[j] = value(0) + ... + value(j - 1)      for j < count,   total = sum of all
//
// "Reduce-then-scan" in three dispatches over blocks of SCAN_BLOCK_SIZE elements:
//   1. reduce     one workgroup per block: blockSums[b] = sum of the block's values
//   2. blocks     ONE workgroup: exclusive scan of blockSums in place (+ the total)
//   3. downsweep  one workgroup per block: scan of the block + blockSums[b] -> output
//
// The element count lives in a GPU buffer (countSource[countOffset], clamped to maxCount),
// so the CPU can record the scan without knowing it. The C++ side dispatches enough
// workgroups for maxCount, and the workgroups past the actual count exit immediately.
//
// value(j) (push constant flags):
//   default            input[j]
//   kScanFlagGather    input[index[j]]  (e.g. "tiles touched" of the j-th DEPTH-SORTED splat)
//   kScanFlagNonZero   1 if the value is != 0, else 0 (turns counts into flags: the scan
//                      of visibility flags is an order-preserving stream compaction)

#ifndef SCAN_COMMON_GLSL
#define SCAN_COMMON_GLSL

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include "workgroup_scan.glsl"

#define SCAN_WG_SIZE 256u
#define SCAN_ITEMS_PER_THREAD 8u
#define SCAN_BLOCK_SIZE 2048u  // SCAN_WG_SIZE * SCAN_ITEMS_PER_THREAD

const uint kScanFlagGather = 1u;
const uint kScanFlagNonZero = 2u;

// C++ twin: ScanPushConstants in src/render/gpu_scan.cpp.
layout(push_constant) uniform ScanPushConstants {
    uint countOffset;  // index (in uints) of the element count in CountSource
    uint maxCount;     // capacity: the count is clamped to it
    uint totalOffset;  // index (in uints) in TotalOut where scan_blocks writes the total
    uint flags;        // kScanFlag*
} pc;

layout(std430, set = 0, binding = 0) readonly buffer ScanInput { uint scanInput[]; };
layout(std430, set = 0, binding = 1) readonly buffer ScanIndex { uint scanIndex[]; };
layout(std430, set = 0, binding = 2) writeonly buffer ScanOutput { uint scanOutput[]; };
layout(std430, set = 0, binding = 3) buffer BlockSums { uint blockSums[]; };
layout(std430, set = 0, binding = 4) readonly buffer CountSource { uint countSource[]; };
layout(std430, set = 0, binding = 5) buffer TotalOut { uint totalOut[]; };

uint scanCount() { return min(countSource[pc.countOffset], pc.maxCount); }
uint scanNumBlocks(uint count) { return (count + SCAN_BLOCK_SIZE - 1u) / SCAN_BLOCK_SIZE; }

uint scanValue(uint j) {
    uint v = (pc.flags & kScanFlagGather) != 0u ? scanInput[scanIndex[j]] : scanInput[j];
    if ((pc.flags & kScanFlagNonZero) != 0u) v = v != 0u ? 1u : 0u;
    return v;
}

#endif  // SCAN_COMMON_GLSL
