// Port of shaders/tile_render.comp (web phase W2): the RENDER pass (INRIA renderCUDA). One
// workgroup per 16x16 tile, one thread per pixel. The tile's splats are ranges[tile] = [start, end)
// of the sorted list, nearest first. The workgroup walks them in batches of 256 (one per thread):
//
//   1. the threads fetch the batch's splats (centre, conic + opacity, colour) into shared memory,
//      so the pixels read each splat from there instead of once per pixel from VRAM
//   2. each thread blends the batch front to back into its own pixel:
//        alpha = min(0.99, o * exp(power))   (skip power > 0, skip alpha < 1/255)
//        test_T = T * (1 - alpha); if test_T < 1e-4: done, and this splat is NOT blended
//        C += colour * alpha * T; T = test_T
//   3. vote: how many pixels of the tile are done? all of them -> the whole tile stops
//
// exactly as the CPU reference (src/splat/reference_renderer.cpp) and INRIA. C + T * background
// (black) and alpha = 1 - T are stored once per pixel into the rgba16float target (rgba32float in
// the GSR_OUT_RGBA32F test build); the accumulation is fp32 in registers.
//
// Differences from the GLSL:
//   - the whole-tile vote is at the END of each batch (native: at the start of the next batch; the
//     same decisions, since nothing changes in between). WGSL rejects a `break` out of a loop with
//     barriers unless the condition is provably uniform, so the done count is read back with
//     workgroupUniformLoad;
//   - the vote counts with one shared atomic per finished pixel (portable) or with a ballot per
//     subgroup (GSR_SUBGROUPS, native's way): the same count, so the same image;
//   - no stats / StopThePop / packed-splat / radius-bound / fp16 / AoS variants (DESIGN.md 4.2).

#ifdef GSR_SUBGROUPS
enable subgroups;
#endif

#include "splat_common.wgsl"

const kTile: u32 = 16u;
const kThreads: u32 = 256u;  // kTile * kTile
const kBatch: u32 = 256u;    // splats per shared-memory batch (one per thread)

// C++ twin: RenderParams in src/webgpu/wgpu_tile_renderer.cpp.
struct RenderParams {
    imageSize: vec2<u32>,
    tilesX: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> tileValues: array<u32>;  // gaussian index per entry
@group(0) @binding(2) var<storage, read> ranges: array<vec2<u32>>;
#ifdef GSR_OUT_RGBA32F
@group(0) @binding(3) var outImage: texture_storage_2d<rgba32float, write>;
#else
@group(0) @binding(3) var outImage: texture_storage_2d<rgba16float, write>;
#endif
@group(0) @binding(4) var<uniform> params: RenderParams;

// --- Shared-memory batch: 256 x (8 + 16 + 16) bytes = 10 KB ---------------------------------------
var<workgroup> sCenter: array<vec2<f32>, kBatch>;
var<workgroup> sConicOpacity: array<vec4<f32>, kBatch>;
var<workgroup> sColor: array<vec4<f32>, kBatch>;  // xyz colour, w radius (unused: tile bound only)
var<workgroup> sDoneCount: atomic<u32>;
var<workgroup> sDone: u32;

// Loads list entry `entry` into batch slot `slot`.
fn fetchSplat(entry: u32, slot: u32) {
    let s = splats[tileValues[entry]];
    sCenter[slot] = s.center;
    sConicOpacity[slot] = s.conicOpacity;
    sColor[slot] = vec4<f32>(s.color, s.radius);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {
    let tileXY = wid.xy;
    let tile = tileXY.y * params.tilesX + tileXY.x;
    let pix = tileXY * kTile + lid.xy;
    let pixf = vec2<f32>(pix);  // INRIA pixel coordinates: the centre of pixel (i, j) is (i, j)
    let inside = pix.x < params.imageSize.x && pix.y < params.imageSize.y;
    // Pixels outside the image (right/bottom edge tiles) still help fetch, but never blend.
    var done = !inside;
    var T = 1.0;
    var C = vec3<f32>(0.0);

    let range = ranges[tile];  // read-only storage at a uniform index: uniform
    let total = range.y - range.x;
    let rounds = (total + kBatch - 1u) / kBatch;

    if (t == 0u) {
        atomicStore(&sDoneCount, 0u);
    }
    workgroupBarrier();

    for (var round = 0u; round < rounds; round++) {
        // 1. Cooperative fetch of this batch into shared memory.
        let first = range.x + round * kBatch;
        if (first + t < range.y) {
            fetchSplat(first + t, t);
        }
        workgroupBarrier();

        // 2. Front-to-back blending of the batch into this thread's pixel.
        let batch = min(kBatch, total - round * kBatch);
        for (var j = 0u; !done && j < batch; j++) {
            let d = sCenter[j] - pixf;
            let co = sConicOpacity[j];
            let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.0) {
                continue;
            }
            let alpha = min(kMaxAlpha, co.w * exp(power));
            if (alpha < kMinAlpha) {
                continue;
            }
            // INRIA's test-before-accept: the splat that would make T < 1e-4 is NOT blended.
            let testT = T * (1.0 - alpha);
            if (testT < 0.0001) {
                done = true;
                break;
            }
            C += sColor[j].xyz * alpha * T;
            T = testT;
        }

        // 3. Whole-tile early exit (INRIA: __syncthreads_count(done) == BLOCK_SIZE).
#ifdef GSR_SUBGROUPS
        let bits = countOneBits(subgroupBallot(done));
        let n = bits.x + bits.y + bits.z + bits.w;
        if (subgroupElect()) {
            atomicAdd(&sDoneCount, n);
        }
#else
        if (done) {
            atomicAdd(&sDoneCount, 1u);
        }
#endif
        workgroupBarrier();  // the batch is consumed (it may be overwritten now) and the count complete
        if (t == 0u) {
            sDone = atomicLoad(&sDoneCount);
            atomicStore(&sDoneCount, 0u);  // re-armed: the next adds come after the barrier below
        }
        if (workgroupUniformLoad(&sDone) == kThreads) {
            break;  // the same value in every thread: uniform exit
        }
    }

    if (inside) {
        textureStore(outImage, vec2<i32>(pix), vec4<f32>(C, 1.0 - T));  // + T * background (black)
    }
}
