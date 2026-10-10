// Phase 6: which screen tiles a splat is binned into (`--bounds rect|tight|cutoff-only`).
// CPU twin of shaders/tile_coverage.glsl (same expressions, same order); used by the CPU
// reference renderer (its binning) and by the tests (GPU lists vs a CPU construction).
// Derivation and exactness argument: docs/notes/phase6.md section 1.
//
//   rect         INRIA: every tile of the 3-sigma tile rectangle (computeTileRect)
//   tight        rect INTERSECTED with the opacity-aware cutoff ellipse
//                  { e : A ex^2 + 2 B ex ey + C ey^2 <= 2 ln(255 o) + margin }:
//                drops only (pixel, splat) pairs with alpha < 1/255 -> identical image
//   cutoff-only  the cutoff ellipse clipped to the screen only (NOT INRIA: no 3-sigma clip)
#pragma once

#include "splat/splat_math.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace gsr {

enum class TileBounds : uint32_t { Rect = 0, Tight = 1, CutoffOnly = 2 };
const char* tileBoundsName(TileBounds b);
std::optional<TileBounds> parseTileBounds(std::string_view text);

struct CutoffEllipse {
    glm::vec2 c{0.0f};
    float A = 0, B = 0, C = 0, D = 0;
    float t = -1.0f;           // q threshold incl. margin; < 0: no pixel reaches 1/255
    float exPad = 0, eyPad = 0;  // padded half extents
    bool valid = false;        // conic positive definite (else: fall back to the rectangle)
};

CutoffEllipse makeCutoffEllipse(const glm::vec2& center, const glm::vec3& conic, float opacity);

// A tile range in bin tiles: [x0, x1) x [y0, y1).
struct BinRect {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
BinRect rectToBins(const TileRect& rect16, int binSize, int gridX, int gridY);

struct Coverage {
    BinRect bins;
    CutoffEllipse e;
    bool useEllipse = false;
};

Coverage makeCoverage(TileBounds mode, const TileRect& rect16, int binSize, int gridX, int gridY,
                      const glm::vec2& center, const glm::vec3& conic, float opacity);
// Tile rows [first, second) the splat may touch.
std::pair<int, int> coverageRows(const Coverage& cv, int binSize);
// The contiguous run [first, second) of bin tiles of `row` the splat is binned into.
std::pair<int, int> coverageRowSpan(const Coverage& cv, int row, int binSize);
uint32_t coverageCount(const Coverage& cv, int binSize);

// Convenience for a ProjectedSplat (render-visible centre/conic/opacity), the camera's
// 16-px tile grid and a bin grid of size binSize.
Coverage splatCoverage(TileBounds mode, const ProjectedSplat& s, uint32_t tilesX16, uint32_t tilesY16, int binSize,
                       int gridX, int gridY);

// Tests: true if any float decision of this splat's coverage (a ceil/floor of a pixel
// bound) lies within `eps` pixels of an integer, i.e. where GPU and CPU rounding (sqrt,
// division are not correctly rounded on GPUs) could legitimately pick different tiles.
bool coverageIsBorderline(const Coverage& cv, int binSize, float eps = 2e-3f);

}  // namespace gsr
