// Port of shaders/tile_common.glsl (web phase W2): declarations shared by the tile_*.wgsl passes.
// C++ side and the whole pipeline: src/webgpu/wgpu_tile_renderer.h/.cpp.
//
// The control block (native TileControl, 80 bytes) is bound as an array of u32 (or of atomic<u32>
// in the passes that count with atomics): WGSL cannot read atomic struct members non-atomically,
// and different pipelines may view one buffer with different types. Word indices (C++ twin:
// WebTileControl in src/webgpu/shader_constants.h, same layout as native):
const kCtlGaussianCount: u32 = 0u;  // N, written by the per-frame reset
const kCtlVisible: u32 = 1u;        // V: scan total of the visibility flags
const kCtlDuplicates: u32 = 2u;     // M: scan total of the touched-tile counts (unclamped)
const kCtlEntries: u32 = 3u;        // min(M, capacity): entries written / sorted / rendered
const kCtlCullCounts: u32 = 4u;     // 8 words: per cull reason, [0] = visible
const kCtlOverflow: u32 = 12u;      // M - entries
const kCtlListGroupsX: u32 = 13u;   // dispatchWorkgroupsIndirect arguments of the passes over the
const kCtlListGroupsY: u32 = 14u;   //   list (duplicate, ranges): one thread per entry, 2D grid
const kCtlListGroupsZ: u32 = 15u;
const kCtlCoverageMismatch: u32 = 16u;  // splats whose stored spans disagree with their count (0)
                                        // 17..19: one example (splat, record count, tileCounts)
const kCtlWords: u32 = 20u;

const kTileGroupSize: u32 = 256u;  // threads per workgroup of the one-thread-per-item passes
