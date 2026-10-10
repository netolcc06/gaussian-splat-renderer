#include "splat/tile_coverage.h"

#include <algorithm>
#include <cmath>

namespace gsr {

const char* tileBoundsName(TileBounds b) {
    switch (b) {
        case TileBounds::Rect: return "rect";
        case TileBounds::Tight: return "tight";
        case TileBounds::CutoffOnly: return "cutoff-only";
    }
    return "?";
}

std::optional<TileBounds> parseTileBounds(std::string_view text) {
    if (text == "rect") return TileBounds::Rect;
    if (text == "tight") return TileBounds::Tight;
    if (text == "cutoff-only" || text == "cutoff") return TileBounds::CutoffOnly;
    return std::nullopt;
}

CutoffEllipse makeCutoffEllipse(const glm::vec2& center, const glm::vec3& conic, float opacity) {
    CutoffEllipse e;
    e.c = center;
    const float A = conic.x, B = conic.y, C = conic.z;
    const float D = A * C - B * B;
    e.A = A;
    e.B = B;
    e.C = C;
    e.D = D;
    e.valid = A > 0.0f && C > 0.0f && D > 0.0f;
    if (!e.valid) return e;
    if (!(opacity >= kMinAlpha)) return e;
    const float qmax = 2.0f * std::log(255.0f * opacity);
    const float rho = std::fabs(B) / std::sqrt(A * C);
    const float K = 2.0f / std::max(1.0f - rho, 1e-30f);
    const float t = qmax + (1e-3f + 2e-5f * K) * std::max(qmax, 1.0f) + 2e-3f;
    if (!(t < 1e30f)) {
        e.valid = false;
        return e;
    }
    e.t = t;
    const float ex = std::sqrt(t * C / D);
    const float ey = std::sqrt(t * A / D);
    e.exPad = ex + (0.02f + 2e-3f * ex);
    e.eyPad = ey + (0.02f + 2e-3f * ey);
    return e;
}

namespace {

int floorDivClamped(float v, int S) { return int(std::floor(std::clamp(v, -65536.0f, 1048576.0f) / float(S))); }

std::pair<int, int> ellipseRows(const CutoffEllipse& e, int S, const BinRect& b) {
    if (e.t < 0.0f) return {0, 0};
    const float y0 = std::ceil(e.c.y - e.eyPad);
    const float y1 = std::floor(e.c.y + e.eyPad);
    const int lo = std::max(b.y0, floorDivClamped(y0, S));
    const int hi = std::min(b.y1, floorDivClamped(y1, S) + 1);
    return {lo, std::max(lo, hi)};
}

struct RowBounds {  // the float pixel bounds before rounding (for the borderline test)
    float x0 = 0, x1 = 0;
    bool empty = true;
};

RowBounds ellipseRowBounds(const CutoffEllipse& e, int row, int S) {
    RowBounds rb;
    const float ey0 = float(row * S) - e.c.y;
    const float ey1 = float(row * S + S - 1) - e.c.y;
    const float lo = std::max(ey0, -e.eyPad);
    const float hi = std::min(ey1, e.eyPad);
    if (!(lo <= hi)) return rb;
    const float exTrue = std::sqrt(e.t * e.C / e.D);
    const float eyRight = std::clamp(-e.B * exTrue / e.C, lo, hi);
    const float eyLeft = std::clamp(e.B * exTrue / e.C, lo, hi);
    const float At = e.A * e.t;
    const float xRight = (-e.B * eyRight + std::sqrt(std::max(0.0f, At - e.D * eyRight * eyRight))) / e.A;
    const float xLeft = (-e.B * eyLeft - std::sqrt(std::max(0.0f, At - e.D * eyLeft * eyLeft))) / e.A;
    const float pad = 0.02f + 2e-3f * exTrue;
    rb.x0 = e.c.x + xLeft - pad;
    rb.x1 = e.c.x + xRight + pad;
    rb.empty = false;
    return rb;
}

std::pair<int, int> ellipseRowSpan(const CutoffEllipse& e, int row, int S, const BinRect& b) {
    const RowBounds rb = ellipseRowBounds(e, row, S);
    if (rb.empty) return {b.x0, b.x0};
    const int x0 = std::max(b.x0, floorDivClamped(std::ceil(rb.x0), S));
    const int x1 = std::min(b.x1, floorDivClamped(std::floor(rb.x1), S) + 1);
    return {x0, std::max(x0, x1)};
}

bool nearInteger(float v, float eps) { return std::fabs(v - std::round(v)) < eps; }

}  // namespace

BinRect rectToBins(const TileRect& r, int S, int gridX, int gridY) {
    const int f = 16 / S;
    return {r.minX * f, r.minY * f, std::min(r.maxX * f, gridX), std::min(r.maxY * f, gridY)};
}

Coverage makeCoverage(TileBounds mode, const TileRect& rect16, int S, int gridX, int gridY, const glm::vec2& center,
                      const glm::vec3& conic, float opacity) {
    Coverage cv;
    cv.bins = mode == TileBounds::CutoffOnly ? BinRect{0, 0, gridX, gridY} : rectToBins(rect16, S, gridX, gridY);
    if (mode != TileBounds::Rect) {
        cv.e = makeCutoffEllipse(center, conic, opacity);
        cv.useEllipse = cv.e.valid;
        if (!cv.e.valid && mode == TileBounds::CutoffOnly) cv.bins = rectToBins(rect16, S, gridX, gridY);
    }
    return cv;
}

std::pair<int, int> coverageRows(const Coverage& cv, int S) {
    if (!cv.useEllipse) return {cv.bins.y0, std::max(cv.bins.y0, cv.bins.y1)};
    return ellipseRows(cv.e, S, cv.bins);
}

std::pair<int, int> coverageRowSpan(const Coverage& cv, int row, int S) {
    if (!cv.useEllipse) return {cv.bins.x0, std::max(cv.bins.x0, cv.bins.x1)};
    return ellipseRowSpan(cv.e, row, S, cv.bins);
}

uint32_t coverageCount(const Coverage& cv, int S) {
    const auto [r0, r1] = coverageRows(cv, S);
    if (!cv.useEllipse) return uint32_t(std::max(0, cv.bins.x1 - cv.bins.x0) * (r1 - r0));
    uint32_t n = 0;
    for (int r = r0; r < r1; ++r) {
        const auto [x0, x1] = ellipseRowSpan(cv.e, r, S, cv.bins);
        n += uint32_t(x1 - x0);
    }
    return n;
}

Coverage splatCoverage(TileBounds mode, const ProjectedSplat& s, uint32_t tilesX16, uint32_t tilesY16, int S, int gridX,
                       int gridY) {
    const TileRect rect = computeTileRect(s.center, s.radius, tilesX16, tilesY16);
    return makeCoverage(mode, rect, S, gridX, gridY, s.center, glm::vec3(s.conicOpacity), s.conicOpacity.w);
}

bool coverageIsBorderline(const Coverage& cv, int S, float eps) {
    if (!cv.useEllipse || cv.e.t < 0.0f) return false;
    // GPU sqrt/division/log are not correctly rounded; for needles (K = 2 / (1 - |rho|) large)
    // a 1-ulp difference in sqrt(A C) moves the margin of t, and thus the extents, by more.
    const float rho = std::fabs(cv.e.B) / std::sqrt(cv.e.A * cv.e.C);
    const float K = 2.0f / std::max(1.0f - rho, 1e-30f);
    eps += (1e-7f * K + 1e-12f * K * K) * (cv.e.exPad + cv.e.eyPad);
    if (nearInteger(cv.e.c.y - cv.e.eyPad, eps) || nearInteger(cv.e.c.y + cv.e.eyPad, eps)) return true;
    const auto [r0, r1] = coverageRows(cv, S);
    for (int r = r0; r < r1; ++r) {
        const RowBounds rb = ellipseRowBounds(cv.e, r, S);
        if (rb.empty) continue;
        if (nearInteger(rb.x0, eps) || nearInteger(rb.x1, eps)) return true;
        // The strip test (lo <= hi) itself: an edge of the strip close to +-eyPad.
        const float ey0 = float(r * S) - cv.e.c.y, ey1 = float(r * S + S - 1) - cv.e.c.y;
        if (std::fabs(ey1 + cv.e.eyPad) < eps || std::fabs(ey0 - cv.e.eyPad) < eps) return true;
    }
    return false;
}

}  // namespace gsr
