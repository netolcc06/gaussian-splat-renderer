// scene_precision: the numbers behind the Phase 5 precision analysis (docs/notes/phase5.md).
//
//   scene_precision --scene <file.ply> [--json <out.json>] [--codebook-size K] [--threads N]
//
// For one real scene it prints, per attribute, the dynamic range (percentiles) and what each
// storage format does to it:
//   positions   |x| range, fp16 spacing at that magnitude, and the projected error in pixels
//               at the training cameras' distances (why positions stay fp32)
//   sigma       range in linear and log space; relative error of linear-fp16 vs log-fp16
//               storage; how many values are fp16 subnormals / round to zero
//   alpha       range, fp16 absolute error, unorm8/unorm16 for comparison
//   rotation    angle error of the fp16 quaternion (after renormalization)
//   SH          DC and per-band rest ranges; per-format coefficient error (f16, q8, cb);
//               and the error of the EVALUATED colour (0.5 + sum Y_lm c_lm, what the
//               rasterizer blends) over random view directions, in 1/255 units.
// No GPU. The encoders are the ones the renderer uses (scene/scene_compression.h).
#include "scene/half.h"
#include "scene/ply_loader.h"
#include "scene/scene_compression.h"
#include "scene/training_cameras.h"
#include "splat/splat_math.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace gsr;
using nlohmann::json;

namespace {

struct Pct {
    double p0, p001, p01, p50, p99, p999, p100;
};
Pct percentiles(std::vector<double> v) {
    Pct p{};
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1)))]; };
    p = {v.front(), at(0.001), at(0.01), at(0.5), at(0.99), at(0.999), v.back()};
    return p;
}
json pctJson(const Pct& p) {
    return {{"min", p.p0}, {"p0.1", p.p001}, {"p1", p.p01}, {"median", p.p50}, {"p99", p.p99}, {"p99.9", p.p999}, {"max", p.p100}};
}
void printPct(const char* name, const Pct& p) {
    std::printf("  %-28s min %10.3g  p0.1 %10.3g  p1 %10.3g  median %10.3g  p99 %10.3g  p99.9 %10.3g  max %10.3g\n", name,
                p.p0, p.p001, p.p01, p.p50, p.p99, p.p999, p.p100);
}

struct ErrStats {
    double sum = 0.0, max = 0.0;
    uint64_t n = 0;
    void add(double e) {
        sum += e;
        max = std::max(max, e);
        ++n;
    }
    double mean() const { return n ? sum / double(n) : 0.0; }
};

// Evaluated SH colour for direction d (splat_math evalSH without the clamp).
glm::vec3 shColor(const glm::vec3* c, int degree, const glm::vec3& d) { return evalSH(degree, c, d); }

}  // namespace

int main(int argc, char** argv) {
    std::string scenePath, jsonPath;
    uint32_t codebookSize = 4096, threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--scene") scenePath = value() ? argv[i] : "";
        else if (a == "--json") jsonPath = value() ? argv[i] : "";
        else if (a == "--codebook-size") codebookSize = uint32_t(std::atoi(value() ? argv[i] : "4096"));
        else if (a == "--threads") threads = uint32_t(std::atoi(value() ? argv[i] : "0"));
        else {
            std::fprintf(stderr, "usage: scene_precision --scene <file.ply> [--json <out.json>] [--codebook-size K]\n");
            return 2;
        }
    }
    if (scenePath.empty()) {
        std::fprintf(stderr, "usage: scene_precision --scene <file.ply> [--json <out.json>] [--codebook-size K]\n");
        return 2;
    }
    try {
        const GaussianScene sc = loadGaussianPly(scenePath);
        const uint32_t n = sc.count;
        std::printf("Scene %s: %u gaussians, SH degree %d\n", scenePath.c_str(), n, sc.shDegree);
        json j;
        j["scene"] = scenePath;
        j["gaussians"] = n;

        // --- Positions -------------------------------------------------------------------
        std::vector<double> absPos;
        absPos.reserve(size_t(n) * 3);
        double maxAbs = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) {
                const double v = std::abs(sc.posAlpha[i][a]);
                absPos.push_back(v);
                maxAbs = std::max(maxAbs, v);
            }
        const Pct pPos = percentiles(absPos);
        std::printf("positions (data space)\n");
        printPct("|coordinate|", pPos);
        // fp16 spacing (ulp) at a magnitude: 2^(floor(log2 x) - 10).
        auto ulp16 = [](double x) { return x < std::ldexp(1.0, -14) ? std::ldexp(1.0, -24) : std::ldexp(1.0, int(std::floor(std::log2(x))) - 10); };
        ErrStats posErr;
        for (uint32_t i = 0; i < n; ++i) {
            const glm::vec3 p(sc.posAlpha[i]);
            const glm::vec3 r(roundToHalf(p.x), roundToHalf(p.y), roundToHalf(p.z));
            posErr.add(glm::length(r - p));
        }
        std::printf("  fp16 spacing at the median |x| %.3g: %.3g; at p99 %.3g: %.3g; at max %.3g: %.3g\n", pPos.p50,
                    ulp16(pPos.p50), pPos.p99, ulp16(pPos.p99), pPos.p100, ulp16(pPos.p100));
        std::printf("  fp16 rounding moves gaussians by %.3g on average, at most %.3g (data units)\n", posErr.mean(),
                    posErr.max);
        // Projected error at the training cameras: |dx| / z * focal (focal from the camera).
        json posJson = {{"abs_coordinate", pctJson(pPos)},
                        {"fp16_move_mean", posErr.mean()},
                        {"fp16_move_max", posErr.max}};
        const std::filesystem::path camsPath = std::filesystem::path(scenePath).parent_path() / "cameras.json";
        if (std::filesystem::exists(camsPath)) {
            const auto cams = loadTrainingCameras(camsPath.string());
            if (!cams.empty()) {
                const TrainingCamera& c0 = cams[0];
                // Pixel error of fp16 positions as seen from camera 0 (1920 px wide output).
                const float focal = c0.fx * (1920.0f / float(c0.width));
                std::vector<double> px;
                for (uint32_t i = 0; i < n; i += 7) {
                    const glm::vec3 p(sc.posAlpha[i]);
                    const glm::vec3 r(roundToHalf(p.x), roundToHalf(p.y), roundToHalf(p.z));
                    const glm::vec3 t = glm::transpose(c0.cameraToWorld) * (p - c0.position);  // world -> camera
                    if (t.z <= 0.2f) continue;
                    px.push_back(double(glm::length(r - p)) / t.z * focal);
                }
                const Pct pPx = percentiles(px);
                std::printf("  fp16 position error projected at training camera 0 (1080p-equivalent focal %.0f px):\n", focal);
                printPct("  pixels", pPx);
                posJson["fp16_error_px_camera0"] = pctJson(pPx);
            }
        }
        j["positions"] = posJson;

        // --- Scales ------------------------------------------------------------------------
        std::vector<double> sig, logSig;
        ErrStats linRel, logRel;
        uint64_t subnormal = 0, toZero = 0;
        for (uint32_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) {
                const float s = sc.scales[i][a];
                sig.push_back(s);
                logSig.push_back(std::log(double(s)));
                if (s < std::ldexp(1.0f, -14)) ++subnormal;
                if (floatToHalf(s) == 0) ++toZero;
                linRel.add(std::abs(double(roundToHalf(s)) - s) / s);
                const float l = std::log(s);
                logRel.add(std::abs(std::exp(double(roundToHalf(l))) - s) / s);
            }
        std::printf("sigma (activated scale, data units)\n");
        const Pct pSig = percentiles(sig), pLog = percentiles(logSig);
        printPct("sigma", pSig);
        printPct("log(sigma)", pLog);
        std::printf("  linear fp16: relative error mean %.3g max %.3g; %llu of %u values are fp16 subnormals "
                    "(< 6.1e-5), %llu round to 0\n",
                    linRel.mean(), linRel.max, static_cast<unsigned long long>(subnormal), n * 3,
                    static_cast<unsigned long long>(toZero));
        std::printf("  log fp16:    relative error mean %.3g max %.3g\n", logRel.mean(), logRel.max);
        j["sigma"] = {{"linear", pctJson(pSig)},
                      {"log", pctJson(pLog)},
                      {"fp16_linear_rel_err_mean", linRel.mean()},
                      {"fp16_linear_rel_err_max", linRel.max},
                      {"fp16_log_rel_err_mean", logRel.mean()},
                      {"fp16_log_rel_err_max", logRel.max},
                      {"fp16_subnormal", subnormal},
                      {"fp16_to_zero", toZero}};

        // --- Opacity -----------------------------------------------------------------------
        std::vector<double> alpha;
        ErrStats aH, a8, a16;
        for (uint32_t i = 0; i < n; ++i) {
            const float a = sc.posAlpha[i].w;
            alpha.push_back(a);
            aH.add(std::abs(double(roundToHalf(a)) - a));
            a8.add(std::abs(std::round(double(a) * 255.0) / 255.0 - a));
            a16.add(std::abs(std::round(double(a) * 65535.0) / 65535.0 - a));
        }
        std::printf("alpha (activated opacity)\n");
        const Pct pA = percentiles(alpha);
        printPct("alpha", pA);
        std::printf("  absolute error: fp16 mean %.3g max %.3g | unorm8 mean %.3g max %.3g | unorm16 mean %.3g max %.3g\n",
                    aH.mean(), aH.max, a8.mean(), a8.max, a16.mean(), a16.max);
        j["alpha"] = {{"range", pctJson(pA)},
                      {"fp16_abs_err_mean", aH.mean()},
                      {"fp16_abs_err_max", aH.max},
                      {"unorm8_abs_err_max", a8.max},
                      {"unorm16_abs_err_max", a16.max}};

        // --- Rotation ----------------------------------------------------------------------
        ErrStats angle;
        for (uint32_t i = 0; i < n; ++i) {
            const glm::vec4 q = sc.rotations[i];
            glm::vec4 r(roundToHalf(q.x), roundToHalf(q.y), roundToHalf(q.z), roundToHalf(q.w));
            r /= glm::length(r);
            const double d = std::min(1.0, std::abs(double(glm::dot(q, r))));
            angle.add(2.0 * std::acos(d));
        }
        std::printf("rotation: fp16 quaternion (renormalized) angle error mean %.3g rad, max %.3g rad\n", angle.mean(),
                    angle.max);
        j["rotation"] = {{"fp16_angle_err_mean_rad", angle.mean()}, {"fp16_angle_err_max_rad", angle.max}};

        // --- SH ------------------------------------------------------------------------------
        const uint32_t K = sc.shCoeffsPerChannel();
        std::vector<double> dc, band[4];
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < K; ++k)
                for (int c = 0; c < 3; ++c) {
                    const double v = sc.sh[(size_t(i) * K + k) * 3 + c];
                    if (k == 0) dc.push_back(v);
                    else band[k < 4 ? 1 : k < 9 ? 2 : 3].push_back(std::abs(v));
                }
        std::printf("SH coefficients\n");
        printPct("DC (f_dc)", percentiles(dc));
        json shJson = {{"dc", pctJson(percentiles(dc))}};
        for (int b = 1; b <= sc.shDegree; ++b) {
            const std::string name = "|rest|, band " + std::to_string(b);
            printPct(name.c_str(), percentiles(band[b]));
            shJson["rest_abs_band" + std::to_string(b)] = pctJson(percentiles(band[b]));
        }

        // Per-format: coefficient error and evaluated-colour error over random directions.
        std::mt19937 rng(1234);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<glm::vec3> dirs(64);
        for (auto& d : dirs) d = glm::normalize(glm::vec3(g(rng), g(rng), g(rng)));
        json formats = json::object();
        for (SceneFormat f : {SceneFormat::F16, SceneFormat::Q8, SceneFormat::Codebook}) {
            CompressionOptions o;
            o.format = f;
            o.layout = SceneLayout::Split;
            o.codebookSize = codebookSize;
            o.threads = threads;
            const EncodedScene e = encodeScene(sc, o);
            ErrStats coef, colErr, dcErr;
            DecodedGaussian d;
            std::vector<glm::vec3> orig(K);
            for (uint32_t i = 0; i < n; i += 3) {
                decodeGaussian(e, i, d);
                for (uint32_t k = 0; k < K; ++k) {
                    orig[k] = sc.shCoeff(i, k);
                    for (int c = 0; c < 3; ++c) {
                        const double err = std::abs(double(d.sh[k][c]) - orig[k][c]);
                        if (k == 0) dcErr.add(err);
                        else coef.add(err);
                    }
                }
                for (size_t t = i % 8; t < dirs.size(); t += 8) {
                    const glm::vec3 a = shColor(orig.data(), sc.shDegree, dirs[t]);
                    const glm::vec3 b = shColor(d.sh.data(), sc.shDegree, dirs[t]);
                    for (int c = 0; c < 3; ++c)
                        colErr.add(std::abs(double(std::clamp(a[c], 0.0f, 1.0f)) - std::clamp(b[c], 0.0f, 1.0f)) * 255.0);
                }
            }
            std::printf("  %-3s %.1f B/gaussian (encode %.0f ms): DC abs err mean %.3g max %.3g; rest abs err mean %.3g "
                        "max %.3g; evaluated colour error mean %.3f/255 max %.1f/255\n",
                        sceneFormatName(f), e.bytesPerGaussian(), e.encodeMs, dcErr.mean(), dcErr.max, coef.mean(),
                        coef.max, colErr.mean(), colErr.max);
            json fj = {{"bytes_per_gaussian", e.bytesPerGaussian()},
                       {"encode_ms", e.encodeMs},
                       {"dc_abs_err_mean", dcErr.mean()},
                       {"dc_abs_err_max", dcErr.max},
                       {"rest_abs_err_mean", coef.mean()},
                       {"rest_abs_err_max", coef.max},
                       {"colour_err_255_mean", colErr.mean()},
                       {"colour_err_255_max", colErr.max}};
            if (f == SceneFormat::Codebook)
                fj["codebook"] = {{"entries", e.codebook.entries}, {"total_ms", e.codebook.totalMs},
                                  {"train_ms", e.codebook.trainMs}, {"assign_ms", e.codebook.assignMs},
                                  {"mse", e.codebook.fullMse}};
            formats[sceneFormatName(f)] = fj;
        }
        shJson["formats"] = formats;
        j["sh"] = shJson;

        if (!jsonPath.empty()) {
            const std::filesystem::path p(jsonPath);
            if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
            std::ofstream(p) << j.dump(2) << "\n";
            std::printf("Wrote %s\n", jsonPath.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
