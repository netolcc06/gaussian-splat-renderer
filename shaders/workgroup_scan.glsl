// Workgroup-wide building blocks shared by the radix sort (radix_*.comp, Phase 3) and the
// prefix-sum shaders (scan_*.comp, Phase 4). Requires the subgroup basic + arithmetic
// extensions to be enabled by the including file.
//
// Index order: subgroup operations (ballots, scans) work in subgroup-lane order, and
// Vulkan does not promise that lanes map to gl_LocalInvocationIndex in order. Every
// shader that scans therefore numbers its threads with linearLane() = subgroup * size +
// lane, which is the order the ballots and scans see, and derives element indices from it.

#ifndef WORKGROUP_SCAN_GLSL
#define WORKGROUP_SCAN_GLSL

// Upper bound on subgroups per workgroup: workgroup size (256) / the smallest subgroup
// size the device may use (8 on NVIDIA: 256 / 32). Set by the C++ side from minSubgroupSize.
layout(constant_id = 0) const uint kMaxSubgroups = 8u;

// Thread index in subgroup-lane order (see the file comment).
uint linearLane() { return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID; }

// Exclusive prefix sum over the whole workgroup of one value per thread, in linearLane()
// order. Two levels: a subgroup scan (hardware), then the subgroup totals (at most
// kMaxSubgroups, via shared memory). Must be called by every thread (it has barriers).
shared uint sScanSubgroupTotals[kMaxSubgroups];
uint workgroupExclusiveScan(uint value, out uint total) {
    uint inclusive = subgroupInclusiveAdd(value);
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1u) sScanSubgroupTotals[gl_SubgroupID] = inclusive;
    barrier();
    uint before = 0u, sum = 0u;
    for (uint s = 0u; s < gl_NumSubgroups; ++s) {
        uint t = sScanSubgroupTotals[s];
        if (s < gl_SubgroupID) before += t;
        sum += t;
    }
    total = sum;
    barrier();  // sScanSubgroupTotals may be reused by the next call
    return before + inclusive - value;
}

#endif  // WORKGROUP_SCAN_GLSL
