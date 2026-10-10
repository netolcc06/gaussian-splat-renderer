// Port of shaders/radix_sort_common.glsl + radix_setup.comp + radix_histogram.comp + radix_scan.comp
// + radix_scatter.comp (web phase W2):
// a GPU LSD radix sort of (key, value) pairs of u32, ascending and STABLE (equal keys keep their
// input order), so the result equals the CPU DepthSorter::radixSortPairs bit for bit. C++ side:
// src/webgpu/wgpu_radix_sort.cpp.
//
// What changed from the GLSL (DESIGN.md section 2.5): native sorts 8-bit digits in blocks of 2048
// keys, and its stable local rank uses subgroup ballots plus 1 KB of shared counters per subgroup
// (26 KB of shared memory with the reorder variant; with subgroup size 4, allowed by WebGPU, 64 KB).
// That does not fit WebGPU's default 16 KB of workgroup storage, and needs subgroups. The web sort:
//   - 4-bit digits (16 buckets): 8 passes for 32-bit depth keys, ceil(bits / 4) rounded up to even
//     for tile keys (4 passes at 1080p);
//   - blocks of 1024 keys, 256 threads x 4 CONSECUTIVE keys;
//   - stable local rank from PACKED COUNTERS: each thread counts its 4 keys' digits in 16 counters,
//     two per u32 (16-bit halves: at most 1024 per block, no carry between halves), i.e. 8 u32 =
//     two vec4; ONE workgroup exclusive scan of those 8 words (workgroup_scan.wgsl, portable or
//     subgroup) gives each thread, for every digit, the number of keys with that digit in the
//     threads before it. Rank of a key = that prefix + the same-digit keys earlier in its own 4.
//     Shared memory: 8 KB (portable scan) or 2 KB (subgroup scan).
//
// Per pass (shift = 0, 4, 8, ...), three dispatches, as native:
//   1. radix_histogram  (one workgroup per block): digit counts -> blockHist[digit * numBlocks + block]
//   2. radix_scan       (one workgroup per digit, 16): exclusive scan along the blocks in place,
//                       digitCounts[digit] = total
//   3. radix_scatter    (one workgroup per block): dst = (keys with a smaller digit, anywhere)
//                       + (same digit, earlier blocks) + (same digit, earlier in this block)
// plus radix_setup (one thread, once per sort): count -> indirect dispatch arguments, with the
// workgroup count split into a 2D grid when it exceeds maxComputeWorkgroupsPerDimension.

#ifdef GSR_SUBGROUPS
enable subgroups;
#endif

#include "workgroup_scan.wgsl"

const kRadixBits: u32 = 4u;
const kRadixSize: u32 = 16u;          // buckets per pass
const kSortItemsPerThread: u32 = 4u;
const kSortBlockSize: u32 = 1024u;    // kScanThreads * kSortItemsPerThread

// C++ twin: SortPassParams in wgpu_radix_sort.cpp (one 256-byte uniform slot per pass, selected
// with a dynamic offset: the WebGPU stand-in for native's push constant `shift`).
struct SortPassParams {
    shift: u32,        // first bit of this pass's digit
    countOffset: u32,  // radix_setup: index (in u32) of the key count in countSource
    maxCount: u32,     // radix_setup: buffer capacity in keys
    maxGroupsX: u32,   // radix_setup: maxComputeWorkgroupsPerDimension (tests: smaller)
}

// Indirect-dispatch arguments + sizes, written by radix_setup (C++ twin: SortArgs).
struct SortArgs {
    groupsX: u32,  // dispatchWorkgroupsIndirect arguments for histogram and scatter:
    groupsY: u32,  // numBlocks workgroups as a (groupsX, groupsY) grid
    groupsZ: u32,
    count: u32,    // number of keys to sort (clamped to the capacity)
    numBlocks: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> sortPass: SortPassParams;

// --- radix_setup: its own bind group layout {0, 8, 9}, because the arguments it writes are a
// read-only binding (and the indirect buffer) of the passes; one buffer may not be bound as both.
@group(0) @binding(8) var<storage, read> countSource: array<u32>;
@group(0) @binding(9) var<storage, read_write> setupArgs: SortArgs;

@compute @workgroup_size(1)
fn radix_setup() {
    let count = min(countSource[sortPass.countOffset], sortPass.maxCount);
    let blocks = (count + kSortBlockSize - 1u) / kSortBlockSize;
    var gx = min(blocks, sortPass.maxGroupsX);
    var gy = 1u;
    if (gx > 0u) {
        gy = (blocks + gx - 1u) / gx;
    }
    setupArgs.groupsX = gx;  // 0 is a valid (empty) indirect dispatch
    setupArgs.groupsY = gy;
    setupArgs.groupsZ = 1u;
    setupArgs.count = count;
    setupArgs.numBlocks = blocks;
}

// --- The passes --------------------------------------------------------------------------------
@group(0) @binding(1) var<storage, read> keysIn: array<u32>;
@group(0) @binding(2) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(4) var<storage, read_write> valuesOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> blockHist: array<u32>;
@group(0) @binding(6) var<storage, read_write> digitCounts: array<u32>;
@group(0) @binding(7) var<storage, read> args: SortArgs;

fn digitOf(key: u32) -> u32 {
    return (key >> sortPass.shift) & (kRadixSize - 1u);
}

var<workgroup> sHistogram: array<atomic<u32>, kRadixSize>;

// Step 1: count each digit in the block (order does not matter: shared atomics).
@compute @workgroup_size(256)
fn radix_histogram(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    initLanes(lanes);
    let block = linearWorkgroup(wid, nwg);
    let numBlocks = args.numBlocks;
    if (block >= numBlocks) {
        return;  // uniform: the padding of the 2D grid
    }
    let count = args.count;
    if (gLane < kRadixSize) {
        atomicStore(&sHistogram[gLane], 0u);
    }
    workgroupBarrier();
    // Coalesced: in iteration i the workgroup reads 256 consecutive keys.
    for (var i = 0u; i < kSortItemsPerThread; i++) {
        let index = block * kSortBlockSize + i * kScanThreads + gLane;
        if (index < count) {
            atomicAdd(&sHistogram[digitOf(keysIn[index])], 1u);
        }
    }
    workgroupBarrier();
    if (gLane < kRadixSize) {
        blockHist[gLane * numBlocks + block] = atomicLoad(&sHistogram[gLane]);
    }
}

// Step 2: one workgroup per digit: row `digit` of blockHist -> its exclusive scan along the blocks,
// in chunks of 256 blocks with a carried running total; digitCounts[digit] = the row's total.
@compute @workgroup_size(256)
fn radix_scan(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>) {
    initLanes(lanes);
    let digit = wid.x;
    let numBlocks = args.numBlocks;
    let row = digit * numBlocks;
    var running = 0u;
    for (var chunk = 0u; chunk < numBlocks; chunk += kScanThreads) {
        let b = chunk + gLane;
        var c = 0u;
        if (b < numBlocks) {
            c = blockHist[row + b];
        }
        let r = workgroupExclusiveScan(c);
        if (b < numBlocks) {
            blockHist[row + b] = running + r.exclusive;
        }
        running += r.total;
    }
    if (gLane == 0u) {
        digitCounts[digit] = running;
    }
}

var<workgroup> sDigitBase: array<u32, kRadixSize>;

// Packed counter of digit d in (a, b): word d / 2, half d & 1.
fn packedCount(s: Scan8, d: u32) -> u32 {
    let w = d >> 1u;
    var word: u32;
    if (w < 4u) {
        word = s.a[w];
    } else {
        word = s.b[w - 4u];
    }
    return (word >> (16u * (d & 1u))) & 0xFFFFu;
}

// Step 3: move every key/value of the block to its sorted position for this digit.
@compute @workgroup_size(256)
fn radix_scatter(lanes: LaneInput, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    initLanes(lanes);
    let block = linearWorkgroup(wid, nwg);
    let numBlocks = args.numBlocks;
    if (block >= numBlocks) {
        return;  // uniform
    }
    let count = args.count;

    // Where this block's keys of digit d start: keys with a smaller digit (16-element scan of
    // digitCounts, recomputed per block: cheaper than another dispatch) + keys with digit d in
    // earlier blocks (radix_scan).
    if (gLane < kRadixSize) {
        var start = 0u;
        for (var e = 0u; e < gLane; e++) {
            start += digitCounts[e];
        }
        sDigitBase[gLane] = start + blockHist[gLane * numBlocks + block];
    }

    // This thread's 4 consecutive keys, their digits, their rank among the thread's own keys with
    // the same digit, and the packed per-digit counts.
    let first = block * kSortBlockSize + gLane * kSortItemsPerThread;
    var keys: array<u32, kSortItemsPerThread>;
    var values: array<u32, kSortItemsPerThread>;
    var digits: array<u32, kSortItemsPerThread>;
    var ranks: array<u32, kSortItemsPerThread>;
    var counts = Scan8(vec4<u32>(0u), vec4<u32>(0u));
    for (var i = 0u; i < kSortItemsPerThread; i++) {
        let index = first + i;
        if (index < count) {
            keys[i] = keysIn[index];
            values[i] = valuesIn[index];
            let d = digitOf(keys[i]);
            digits[i] = d;
            ranks[i] = packedCount(counts, d);
            let inc = 1u << (16u * (d & 1u));
            let w = d >> 1u;
            if (w < 4u) {
                counts.a[w] += inc;
            } else {
                counts.b[w - 4u] += inc;
            }
        }
    }

    // Keys with each digit in the threads before this one (barriers inside: sDigitBase is visible
    // afterwards).
    let before = workgroupExclusiveScan8(counts.a, counts.b);

    for (var i = 0u; i < kSortItemsPerThread; i++) {
        let index = first + i;
        if (index < count) {
            let d = digits[i];
            let dst = sDigitBase[d] + packedCount(before, d) + ranks[i];
            keysOut[dst] = keys[i];
            valuesOut[dst] = values[i];
        }
    }
}
