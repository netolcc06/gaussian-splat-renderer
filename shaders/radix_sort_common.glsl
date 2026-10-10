// GPU LSD radix sort (Phase 3): declarations shared by the four sort shaders.
//
// Sorts (key, value) pairs of 32-bit uints by key, ascending and STABLE (equal keys keep
// their input order), so the result is bit-identical to the CPU DepthSorter
// (src/splat/depth_sort.cpp). C++ side: src/render/gpu_radix_sort.cpp. The algorithm is
// described in detail in docs/notes/phase3.md; in short, one pass per 8-bit digit
// (shift = 0, 8, 16, 24), each pass being three dispatches:
//
//   1. histogram  (one workgroup per block of 2048 keys): count each digit in the block
//                 -> blockHist[digit * numBlocks + block]
//   2. scan       (one workgroup per digit): exclusive prefix sum of blockHist along the
//                 blocks, in place -> "how many keys with this digit come from earlier
//                 blocks"; plus digitCounts[digit] = total keys with this digit
//   3. scatter    (one workgroup per block): output position of a key =
//                   (keys with a smaller digit, anywhere)        exclusive scan of digitCounts
//                 + (keys with the same digit in earlier blocks) blockHist after the scan
//                 + (keys with the same digit earlier in this block) stable local rank
//
// The local rank is what makes the sort stable, and the subtle part (radix_scatter.comp).
//
// Index order: subgroup operations (ballot, scans) work in subgroup-lane order, and
// Vulkan does not promise that lanes map to gl_LocalInvocationIndex in order. Every
// shader here therefore numbers its threads with linearLane() = subgroup * size + lane,
// which is the order the ballots and scans see, and derives element indices from it.

#ifndef RADIX_SORT_COMMON_GLSL
#define RADIX_SORT_COMMON_GLSL

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define RADIX_BITS 8u
#define RADIX_SIZE 256u                 // 1 << RADIX_BITS buckets per pass
#define SORT_WG_SIZE 256u               // threads per workgroup (== RADIX_SIZE: one thread per digit)
#define SORT_ITEMS_PER_THREAD 8u        // keys per thread per block
#define SORT_BLOCK_SIZE 2048u           // SORT_WG_SIZE * SORT_ITEMS_PER_THREAD keys per block

// kMaxSubgroups (specialization constant 0), linearLane() and workgroupExclusiveScan()
// live in workgroup_scan.glsl (Phase 4: shared with the prefix-sum shaders scan_*.comp).
#include "workgroup_scan.glsl"

// Indirect-dispatch arguments + sizes, written by radix_setup.comp (C++ twin: SortArgs).
struct SortArgs {
    uint groupsX;    // VkDispatchIndirectCommand for histogram and scatter: numBlocks
    uint groupsY;    // 1
    uint groupsZ;    // 1
    uint count;      // number of keys to sort (clamped to the buffer capacity)
    uint numBlocks;  // ceil(count / SORT_BLOCK_SIZE)
};

layout(push_constant) uniform SortPushConstants {
    uint shift;        // first bit of this pass's digit (0, 8, 16, 24; up to 56 with GSR_KEY64)
    uint countOffset;  // radix_setup.comp: index of the count (in uints) in CountSource
    uint maxCount;     // radix_setup.comp: buffer capacity in keys
} pc;

uint digitOf(uint key) { return (key >> pc.shift) & (RADIX_SIZE - 1u); }

#ifdef GSR_KEY64
// Phase 4 build (radix_*_key64.comp.spv): 64-bit keys stored as two words, the low word
// in the usual key buffers and the high word in KeysHi buffers (bindings 8 and 9).
// Passes with shift < 32 sort on the low word, passes with shift >= 32 on the high word.
uint digitOf64(uint lo, uint hi) {
    return pc.shift >= 32u ? ((hi >> (pc.shift - 32u)) & (RADIX_SIZE - 1u)) : digitOf(lo);
}
#endif

#endif  // RADIX_SORT_COMMON_GLSL
