// flicker: temporal-popping metric of a rendered frame sequence against a ground-truth
// sequence of the same camera path (Phase 6, StopThePop).
//
//   flicker --test <prefix> --gt <prefix> --frames N [--gt-ext .pfm|.png] [--json out.json]
//
// Frames are <prefix>_0000.png ... (gsplat --sequence) and <gt prefix>_0000<ext> (gsplat
// --sequence, or gsplat_ref --orbit-frames i --pfm). Both are clamped to [0, 1] and rounded
// to 8 bits (the GPU output's quantization, so float GT frames do not look "smoother").
//
// Metrics, over all pixels and channels (8-bit levels):
//   psnr_mean          mean over frames of the PSNR of frame t vs GT frame t (static error)
//   temporal_rmse      RMS over t >= 1 of (I_t - I_{t-1}) - (G_t - G_{t-1}): the part of the
//                      frame-to-frame change that the ground truth does not have. Smooth,
//                      view-consistent differences cancel; popping (a splat jumping in front
//                      of another between two frames) does not.
//   pop_fraction       fraction of pixels (t >= 1) whose largest channel of that difference
//                      exceeds --pop-threshold (default 8/255): visible pops
//   change_rms         RMS of I_t - I_{t-1} (how much the image moves anyway, for scale)
#include "app/image_io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> load8(const std::string& path, uint32_t& w, uint32_t& h) {
    const gsr::ImageRgbF img = gsr::readImageRgbF(path);
    w = img.width;
    h = img.height;
    std::vector<uint8_t> out(img.rgb.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = uint8_t(std::lround(std::clamp(img.rgb[i], 0.0f, 1.0f) * 255.0f));
    return out;
}

std::string frameName(const std::string& prefix, uint32_t i, const std::string& ext) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%04u", i);
    return prefix + buf + ext;
}

}  // namespace

int main(int argc, char** argv) {
    std::string test, gt, gtExt = ".png", json;
    uint32_t frames = 0;
    double popThreshold = 8.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--test") test = v;
        else if (a == "--gt") gt = v;
        else if (a == "--gt-ext") gtExt = v;
        else if (a == "--frames") frames = uint32_t(std::stoul(v));
        else if (a == "--json") json = v;
        else if (a == "--pop-threshold") popThreshold = std::stod(v);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (test.empty() || gt.empty() || frames < 2) {
        std::fprintf(stderr, "usage: flicker --test <prefix> --gt <prefix> --frames N [--gt-ext .pfm] [--json f]\n");
        return 2;
    }
    try {
        std::vector<uint8_t> prevI, prevG;
        double psnrSum = 0.0, tempSe = 0.0, changeSe = 0.0, gtChangeSe = 0.0;
        uint64_t tempN = 0, pops = 0, popPixels = 0;
        std::vector<double> perFrameTemporal;
        for (uint32_t f = 0; f < frames; ++f) {
            uint32_t w = 0, h = 0, w2 = 0, h2 = 0;
            const std::vector<uint8_t> I = load8(frameName(test, f, ".png"), w, h);
            const std::vector<uint8_t> G = load8(frameName(gt, f, gtExt), w2, h2);
            if (w != w2 || h != h2) throw std::runtime_error("size mismatch at frame " + std::to_string(f));
            double se = 0.0;
            for (size_t i = 0; i < I.size(); ++i) se += double(int(I[i]) - int(G[i])) * double(int(I[i]) - int(G[i]));
            const double mse = se / double(I.size()) / (255.0 * 255.0);
            psnrSum += mse > 0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
            if (f > 0) {
                double frameSe = 0.0;
                for (size_t p = 0; p < size_t(w) * h; ++p) {
                    int worst = 0;
                    for (int c = 0; c < 3; ++c) {
                        const size_t i = p * 3 + c;
                        const int dI = int(I[i]) - int(prevI[i]), dG = int(G[i]) - int(prevG[i]);
                        const int e = dI - dG;
                        frameSe += double(e) * e;
                        changeSe += double(dI) * dI;
                        gtChangeSe += double(dG) * dG;
                        worst = std::max(worst, std::abs(e));
                    }
                    pops += worst > popThreshold;
                    ++popPixels;
                }
                tempSe += frameSe;
                tempN += I.size();
                perFrameTemporal.push_back(std::sqrt(frameSe / double(I.size())));
            }
            prevI = I;
            prevG = G;
        }
        const double temporal = std::sqrt(tempSe / double(tempN));
        const double change = std::sqrt(changeSe / double(tempN));
        const double gtChange = std::sqrt(gtChangeSe / double(tempN));
        const double popFraction = double(pops) / double(popPixels);
        std::sort(perFrameTemporal.begin(), perFrameTemporal.end());
        const double p95 = perFrameTemporal[std::min(perFrameTemporal.size() - 1, perFrameTemporal.size() * 95 / 100)];
        std::printf("frames %u: PSNR vs GT %.2f dB | temporal RMSE %.3f/255 (worst-5%% frame %.3f) | pops > %.0f/255: "
                    "%.4f%% px | change RMS %.3f/255 (GT %.3f)\n",
                    frames, psnrSum / frames, temporal, p95, popThreshold, 100.0 * popFraction, change, gtChange);
        if (!json.empty()) {
            nlohmann::json j = {{"test", test},          {"gt", gt},
                                {"frames", frames},      {"psnr_mean_db", psnrSum / frames},
                                {"temporal_rmse_255", temporal}, {"temporal_rmse_p95_frame_255", p95},
                                {"pop_threshold_255", popThreshold}, {"pop_fraction", popFraction},
                                {"change_rms_255", change}, {"gt_change_rms_255", gtChange}};
            const std::filesystem::path p(json);
            if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
            std::ofstream(p) << j.dump(2) << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
