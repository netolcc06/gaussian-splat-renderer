// In-page GPU self-tests of the web splat-tiles renderer (web phase W2): the web counterparts of
// tests/test_gpu_sort.cpp and tests/test_tile_raster.cpp. CPU twins compiled into the wasm
// (DepthSorter, tile_coverage, coverage_spans, reference_renderer) are compared with GPU buffers read
// back asynchronously (the W1 decode self-test pattern):
//
//   scan      WgpuScan vs a CPU prefix sum, bit for bit: sizes around the workgroup (256) and block
//             (2048) sizes and large ones, the flags and gather modes, the GPU-side count (smaller
//             than the data, clamped to the capacity), the total, untouched memory past the count, a
//             forced 2D workgroup grid; portable and subgroup kernels
//   sort      WgpuRadixSort vs DepthSorter::radixSortPairs, bit for bit incl. stability: random,
//             equal, sorted, reverse, few distinct and 13-bit keys, odd sizes, GPU-side count + clamp,
//             a forced 2D grid; portable and subgroup kernels
//   coverage  the stored coverage spans of a synthetic scene vs the CPU tile_coverage + coverage_spans
//             on the GPU's own splats (exact except float-borderline splats, as native
//             coverageIsBorderline), counts = record sums, every exact tile inside the stored set
//   tiles     the sorted (tile, splat) list, the depth order and the tile ranges vs a CPU construction
//             from the GPU's own splats and spans; overflow (clamped list, exact counter, the CPU's
//             truncated list); a forced 2D grid; rect vs tight and portable vs subgroups give
//             identical images
//   image32   the rgba32float render vs the CPU reference renderer (PSNR > 60 dB, as native)
//   scene     analysis of the app's current scene/frame: the coverage checks above on real data, the
//             histogram of tile rows per splat and the list overhead of the stored tail per K
//
// Asynchronous: run(name) starts a test (once) and returns {"pending": true} until it is done.
#pragma once

#include "splat/splat_math.h"

#include <webgpu/webgpu.h>

#include <functional>
#include <memory>
#include <string>

namespace gsr {

class WgslShaderCache;
class WgpuTileRenderer;
class WgpuScene;

class WgpuTileSelftest {
public:
    // What the "scene" analysis needs from the app: the renderer, scene and camera of the last frame
    // (null renderer: no scene).
    struct SceneAccess {
        const WgpuTileRenderer* renderer = nullptr;
        SplatCamera camera;
        uint32_t width = 0, height = 0;
    };

    WgpuTileSelftest(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders, WGPUBindGroupLayout sceneLayout,
                     bool subgroupsAvailable, std::function<SceneAccess()> sceneAccess);
    ~WgpuTileSelftest();
    WgpuTileSelftest(const WgpuTileSelftest&) = delete;
    WgpuTileSelftest& operator=(const WgpuTileSelftest&) = delete;

    // JSON result of test `name` ("scan", "sort", "coverage", "tiles", "image32", "scene"), starting
    // it on the first call; {"pending": true} while running.
    std::string run(const std::string& name);
    static bool known(const std::string& name);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gsr
