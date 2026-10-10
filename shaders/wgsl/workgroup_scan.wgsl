// Port of shaders/workgroup_scan.glsl (web phase W2): workgroup-wide exclusive prefix sums for the
// scan (scan.wgsl) and the radix sort (radix_sort.wgsl). Workgroup size: 256 threads (kScanThreads).
//
// Two builds (DESIGN.md D7):
//   portable (default)  shared memory only (Hillis-Steele over the 256 threads); runs on every
//                       WebGPU implementation, no `subgroups` feature needed. The reference path.
//   GSR_SUBGROUPS       native's algorithm: a subgroup scan per subgroup, then the subgroup totals
//                       through shared memory. Needs the `subgroups` feature and the `subgroup_id`
//                       / `num_subgroups` builtins; chosen at run time.
// Both give identical results (integer sums).
//
// Thread numbering: subgroup operations work in subgroup-lane order, and WebGPU (like Vulkan)
// does not promise that lanes follow local_invocation_index. Every shader that scans therefore
// numbers its threads with gLane = subgroup_id * subgroup_size + subgroup_invocation_id (subgroup
// build; native's linearLane()) or local_invocation_index (portable), set by initLanes() at the
// start of each entry point, and derives element indices from it.

const kScanThreads: u32 = 256u;

var<private> gLane: u32;
#ifdef GSR_SUBGROUPS
var<private> gSubgroupId: u32;
var<private> gSubgroupInvocation: u32;
var<private> gSubgroupSize: u32;
var<private> gNumSubgroups: u32;

struct LaneInput {
    @builtin(local_invocation_index) index: u32,
    @builtin(subgroup_invocation_id) sgInvocation: u32,
    @builtin(subgroup_size) sgSize: u32,
    @builtin(subgroup_id) sgId: u32,
    @builtin(num_subgroups) numSubgroups: u32,
}

fn initLanes(l: LaneInput) {
    gSubgroupId = l.sgId;
    gSubgroupInvocation = l.sgInvocation;
    gSubgroupSize = l.sgSize;
    gNumSubgroups = l.numSubgroups;
    gLane = l.sgId * l.sgSize + l.sgInvocation;
}
#else
struct LaneInput {
    @builtin(local_invocation_index) index: u32,
}

fn initLanes(l: LaneInput) {
    gLane = l.index;
}
#endif

struct ScanResult {
    exclusive: u32,  // sum of the values of the threads before this one (gLane order)
    total: u32,      // sum over the workgroup
}

// The 8 packed counters of the radix scatter (two vec4).
struct Scan8 {
    a: vec4<u32>,
    b: vec4<u32>,
}

#ifdef GSR_SUBGROUPS

// At most 256 / 4 subgroups (WebGPU allows subgroup sizes 4 .. 128).
var<workgroup> sScanTotals: array<u32, 64>;
var<workgroup> sScanTotals8: array<Scan8, 64>;

// Exclusive prefix sum over the workgroup of one value per thread. Every thread must call it
// (barriers).
fn workgroupExclusiveScan(value: u32) -> ScanResult {
    let inclusive = subgroupInclusiveAdd(value);
    if (gSubgroupInvocation == gSubgroupSize - 1u) {
        sScanTotals[gSubgroupId] = inclusive;
    }
    workgroupBarrier();
    var before = 0u;
    var sum = 0u;
    for (var s = 0u; s < gNumSubgroups; s++) {
        let t = sScanTotals[s];
        if (s < gSubgroupId) {
            before += t;
        }
        sum += t;
    }
    workgroupBarrier();  // sScanTotals may be reused by the next call
    return ScanResult(before + inclusive - value, sum);
}

// Exclusive prefix sum of 8 counters per thread (no total).
fn workgroupExclusiveScan8(a: vec4<u32>, b: vec4<u32>) -> Scan8 {
    let ea = subgroupExclusiveAdd(a);
    let eb = subgroupExclusiveAdd(b);
    if (gSubgroupInvocation == gSubgroupSize - 1u) {
        sScanTotals8[gSubgroupId] = Scan8(ea + a, eb + b);
    }
    workgroupBarrier();
    var before = Scan8(ea, eb);
    for (var s = 0u; s < gSubgroupId; s++) {
        let t = sScanTotals8[s];
        before.a += t.a;
        before.b += t.b;
    }
    workgroupBarrier();
    return before;
}

#else

var<workgroup> sScan: array<u32, kScanThreads>;
var<workgroup> sScan8: array<Scan8, kScanThreads>;

// Exclusive prefix sum over the workgroup of one value per thread (Hillis-Steele: log2(256) = 8
// steps of "add the value `offset` threads back"). Every thread must call it (barriers).
fn workgroupExclusiveScan(value: u32) -> ScanResult {
    var inclusive = value;
    sScan[gLane] = inclusive;
    workgroupBarrier();
    for (var offset = 1u; offset < kScanThreads; offset <<= 1u) {
        var add = 0u;
        if (gLane >= offset) {
            add = sScan[gLane - offset];
        }
        workgroupBarrier();  // everybody read before anybody writes
        inclusive += add;
        sScan[gLane] = inclusive;
        workgroupBarrier();
    }
    let total = sScan[kScanThreads - 1u];
    workgroupBarrier();  // sScan may be reused by the next call
    return ScanResult(inclusive - value, total);
}

fn workgroupExclusiveScan8(a: vec4<u32>, b: vec4<u32>) -> Scan8 {
    var inclusive = Scan8(a, b);
    sScan8[gLane] = inclusive;
    workgroupBarrier();
    for (var offset = 1u; offset < kScanThreads; offset <<= 1u) {
        var add = Scan8(vec4<u32>(0u), vec4<u32>(0u));
        if (gLane >= offset) {
            add = sScan8[gLane - offset];
        }
        workgroupBarrier();
        inclusive.a += add.a;
        inclusive.b += add.b;
        sScan8[gLane] = inclusive;
        workgroupBarrier();
    }
    workgroupBarrier();
    return Scan8(inclusive.a - a, inclusive.b - b);
}

#endif

// The linear index of this workgroup in a 1D dispatch that may be split into a 2D grid because
// of maxComputeWorkgroupsPerDimension (65535): x fastest.
fn linearWorkgroup(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}
