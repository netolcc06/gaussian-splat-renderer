// Constants and struct layouts shared by the C++ side of the web splat-tiles renderer and its WGSL
// shaders (DESIGN.md section 4.3: the single source on the web side). Every value here has a WGSL
// twin named in the comment; the shader self-test and the in-page GPU tests catch disagreements.
#pragma once

#include "splat/splat_math.h"

#include <cstddef>
#include <cstdint>

namespace gsr::web {

// Threads of the one-thread-per-item passes and of the workgroup scans
// (kTileGroupSize in tile_common.wgsl, kScanThreads in workgroup_scan.wgsl).
inline constexpr uint32_t kGroupSize = 256;

// scan.wgsl: kScanBlockSize.
inline constexpr uint32_t kScanBlockSize = 2048;

// radix_sort.wgsl: kRadixBits, kRadixSize, kSortBlockSize.
inline constexpr uint32_t kRadixBits = 4;
inline constexpr uint32_t kRadixSize = 1u << kRadixBits;
inline constexpr uint32_t kSortBlockSize = 1024;

// coverage_spans.wgsl: kSpanRows, kSpanWords (record of the stored tile coverage).
inline constexpr uint32_t kCoverageSpanRows = 8;
inline constexpr uint32_t kCoverageSpanWords = 2 + kCoverageSpanRows / 2;

// The render tile (tile_render.wgsl kTile) = binning tile = INRIA's getRect tile.
inline constexpr uint32_t kTileSize = 16;

// The control block (tile_common.wgsl kCtl*; native TileControl in splat_tile_renderer.h has the
// same layout and size, so readbacks and tools can share code).
struct WebTileControl {
    uint32_t gaussianCount;  // N
    uint32_t visible;        // V
    uint32_t duplicates;     // M, unclamped
    uint32_t entries;        // min(M, capacity)
    uint32_t cullCounts[8];  // per cull reason, [0] = visible
    uint32_t overflow;       // M - entries
    uint32_t listGroups[3];  // indirect dispatch arguments of the passes over the list (2D grid)
    uint32_t coverageMismatch;  // splats whose stored spans disagree with their count (always 0)
    uint32_t mismatchExample[3];  // one disagreeing splat: index, record count, tileCounts (diagnostics)
};
static_assert(sizeof(WebTileControl) == 80, "must match tile_common.wgsl (kCtlWords = 20)");
static_assert(offsetof(WebTileControl, overflow) == 12 * 4, "kCtlOverflow");
static_assert(offsetof(WebTileControl, listGroups) == 13 * 4, "kCtlListGroupsX");
static_assert(offsetof(WebTileControl, coverageMismatch) == 16 * 4, "kCtlCoverageMismatch");

// struct Splat in splat_common.wgsl.
static_assert(sizeof(ProjectedSplat) == 48, "must match struct Splat in splat_common.wgsl");

}  // namespace gsr::web
