// imgdiff: compares two images (PNG or PFM) the way the reference checks need.
//
//   imgdiff <test> <reference> [--min-psnr <dB>] [--threshold <v>] [--heatmap <out.png>]
//           [--heatmap-max <v>] [--json <out.json>]
//
// Both images are clamped to [0, 1] first: that is what is displayed, and a GPU UNORM
// target can not hold anything else. Errors are per channel, on the [0, 1] scale:
//   PSNR = 10 log10(1 / MSE)            (peak = 1.0; "inf" for identical images)
//   mean / max absolute error           (also printed in 8-bit levels, x255)
//   % pixels whose largest channel error exceeds --threshold (default 2/255)
// Heatmap: per-pixel max channel error, 0 = black ... --heatmap-max (default 16/255)
// and above = white, through blue, magenta, orange, yellow.
//
// Exit code: 0 = ok, 1 = PSNR below --min-psnr, 2 = usage or I/O error.
#include "app/image_io.h"

#include <nlohmann/json.hpp>

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <stb_image_write.h>  // implementation compiled in app/image_io.cpp
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

void printUsage() {
    std::printf(
        "Usage: imgdiff <test> <reference> [options]   (PNG or PFM; sizes must match)\n"
        "  --min-psnr <dB>       exit 1 if the PSNR is below this\n"
        "  --threshold <v>       per-pixel error threshold for the %% statistic (default 0.00784 = 2/255)\n"
        "  --heatmap <out.png>   write a per-pixel error heatmap\n"
        "  --heatmap-max <v>     error shown as white in the heatmap (default 0.0627 = 16/255)\n"
        "  --json <out.json>     write the metrics as JSON\n");
}

// Black -> blue -> magenta -> orange -> yellow -> white.
void heatColor(float t, uint8_t* rgb) {
    static const float stops[6][3] = {{0, 0, 0}, {0.1f, 0.1f, 0.8f}, {0.8f, 0.1f, 0.7f},
                                      {1.0f, 0.5f, 0.1f}, {1.0f, 0.95f, 0.2f}, {1, 1, 1}};
    t = std::clamp(t, 0.0f, 1.0f) * 5.0f;
    const int i = std::min(4, int(t));
    const float f = t - float(i);
    for (int c = 0; c < 3; ++c)
        rgb[c] = uint8_t(std::lround(255.0f * (stops[i][c] + f * (stops[i + 1][c] - stops[i][c]))));
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> files;
    double minPsnr = -std::numeric_limits<double>::infinity();
    float threshold = 2.0f / 255.0f, heatMax = 16.0f / 255.0f;
    std::string heatmapPath, jsonPath;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "--min-psnr" || arg == "--threshold" || arg == "--heatmap-max" || arg == "--heatmap" ||
                   arg == "--json") {
            const char* v = value();
            if (!v) {
                std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
                return 2;
            }
            if (arg == "--min-psnr") minPsnr = std::atof(v);
            else if (arg == "--threshold") threshold = float(std::atof(v));
            else if (arg == "--heatmap-max") heatMax = float(std::atof(v));
            else if (arg == "--heatmap") heatmapPath = v;
            else jsonPath = v;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "error: unknown option %s\n", argv[i]);
            printUsage();
            return 2;
        } else {
            files.emplace_back(arg);
        }
    }
    if (files.size() != 2) {
        printUsage();
        return 2;
    }

    try {
        const gsr::ImageRgbF a = gsr::readImageRgbF(files[0]);
        const gsr::ImageRgbF b = gsr::readImageRgbF(files[1]);
        if (a.width != b.width || a.height != b.height) {
            std::fprintf(stderr, "error: size mismatch %ux%u vs %ux%u\n", a.width, a.height, b.width, b.height);
            return 2;
        }
        const size_t pixels = size_t(a.width) * a.height;
        double sumSq = 0.0, sumAbs = 0.0, maxAbs = 0.0;
        size_t over = 0;
        std::vector<float> pixelErr(pixels);
        for (size_t p = 0; p < pixels; ++p) {
            float worst = 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float va = std::clamp(a.rgb[p * 3 + c], 0.0f, 1.0f);
                const float vb = std::clamp(b.rgb[p * 3 + c], 0.0f, 1.0f);
                const double d = std::fabs(double(va) - double(vb));
                sumSq += d * d;
                sumAbs += d;
                maxAbs = std::max(maxAbs, d);
                worst = std::max(worst, float(d));
            }
            pixelErr[p] = worst;
            if (worst > threshold) ++over;
        }
        const double n = double(pixels) * 3.0;
        const double mse = sumSq / n;
        const double psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : std::numeric_limits<double>::infinity();
        const double meanAbs = sumAbs / n;
        const double overPct = 100.0 * double(over) / double(pixels);

        std::printf("PSNR %.2f dB | mean abs %.5f (%.3f/255) | max abs %.4f (%.1f/255) | %.3f%% pixels > %.4f "
                    "(%.1f/255)\n",
                    psnr, meanAbs, meanAbs * 255.0, maxAbs, maxAbs * 255.0, overPct, threshold, threshold * 255.0);

        if (!heatmapPath.empty()) {
            std::vector<uint8_t> rgb(pixels * 3);
            for (size_t p = 0; p < pixels; ++p) heatColor(pixelErr[p] / heatMax, &rgb[p * 3]);
            const std::filesystem::path hp(heatmapPath);
            if (hp.has_parent_path()) std::filesystem::create_directories(hp.parent_path());
            if (!stbi_write_png(heatmapPath.c_str(), int(a.width), int(a.height), 3, rgb.data(), int(a.width * 3))) {
                std::fprintf(stderr, "error: cannot write %s\n", heatmapPath.c_str());
                return 2;
            }
        }
        if (!jsonPath.empty()) {
            nlohmann::json j = {{"test", files[0]},
                                {"reference", files[1]},
                                {"width", a.width},
                                {"height", a.height},
                                {"psnr_db", std::isinf(psnr) ? 999.0 : psnr},
                                {"mse", mse},
                                {"mean_abs", meanAbs},
                                {"max_abs", maxAbs},
                                {"threshold", threshold},
                                {"pixels_over_threshold_pct", overPct}};
            const std::filesystem::path jp(jsonPath);
            if (jp.has_parent_path()) std::filesystem::create_directories(jp.parent_path());
            std::ofstream(jp) << j.dump(2) << "\n";
        }
        if (psnr < minPsnr) {
            std::printf("FAIL: PSNR %.2f dB is below the minimum %.2f dB\n", psnr, minPsnr);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
