// Phase 4 tile rasterizer (`--renderer splat-tiles`): declarations shared by the tile_*.comp
// shaders. C++ side and the whole pipeline: src/render/splat_tile_renderer.h/.cpp.

#ifndef TILE_COMMON_GLSL
#define TILE_COMMON_GLSL

// The control block, C++ twin: TileControl in splat_tile_renderer.h (80 bytes). Declared
// by each shader at its own binding with TILE_CONTROL_MEMBERS.
#define TILE_CONTROL_MEMBERS                                                                  \
    uint gaussianCount;  /* N, written by the per-frame reset (count of the scans over N) */  \
    uint visible;        /* V: scan total of the visibility flags (binning mode)         */  \
    uint duplicates;     /* M: scan total of the touched-tile counts (unclamped)         */  \
    uint entries;        /* min(M, capacity): entries actually written/sorted/rendered   */  \
    uint cullCounts[8];  /* per cull reason (preprocess), [0] = visible                  */  \
    uint overflow;       /* M - entries: (tile, splat) pairs dropped for lack of space   */  \
    uint rangesGroupsX;  /* VkDispatchIndirectCommand of the tile-ranges pass            */  \
    uint rangesGroupsY;                                                                       \
    uint rangesGroupsZ;                                                                       \
    uint coverageMismatch; /* Phase 6: splats whose written tiles != their count (0)     */  \
    uint controlPad0;                                                                         \
    uint controlPad1;                                                                         \
    uint controlPad2;

#define TILE_RANGES_WG_SIZE 256u

// Phase 6: the bounds mode (tile_coverage.glsl, `--bounds`) and the tile size of the binning
// grid (`--tile-size 8|16`, a divisor of INRIA's 16-px getRect tiles). Specialization
// constants of the preprocess (tiles build), duplicate and render passes.
layout(constant_id = 12) const uint kBoundsMode = 0u;
layout(constant_id = 13) const uint kBinTileSize = 16u;

#endif  // TILE_COMMON_GLSL
