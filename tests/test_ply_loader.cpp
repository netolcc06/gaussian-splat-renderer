// Tests for the 3DGS PLY loader (src/scene/ply_loader.*). Run through CTest.
//
//   test_ply_loader --synthetic-dir data/synthetic --expected <json> --scratch-dir <dir>
//                   [--scenes-dir data/scenes]
//
// 1. Every file in the synthetic dir is loaded and compared against the values that
//    tools/gen_synthetic_ply.py --expected-json says it wrote: count, SH degree, full
//    sample records (bit-exact raw values, SH transpose), per-property sums over all
//    records, and the activations.
// 2. Hand-written files exercise the header parser: other scalar types, unknown
//    properties, an element before `vertex`, zero quaternions, NaN.
// 3. Error paths: truncated files, ASCII, big-endian, compressed PLY, wrong f_rest
//    counts, missing properties, ... must throw PlyError with a helpful message.
// 4. Optional: the real scenes in data/scenes (if present): counts and SH degree.
#include "scene/ply_loader.h"
#include "scene/scene_transform.h"
#include "scene/training_cameras.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using gsr::GaussianScene;
using gsr::PlyError;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// --- Synthetic files vs. generator ------------------------------------------------

// Raw value of PLY property `name` for gaussian i, as the loader stored it.
bool rawValue(const GaussianScene& s, uint32_t i, const std::string& name, float& out) {
    const uint32_t K = s.shCoeffsPerChannel();
    if (name == "x" || name == "y" || name == "z") {
        out = s.posAlpha[i][name[0] - 'x'];
    } else if (name == "opacity") {
        out = s.rawOpacity[i];
    } else if (name.rfind("scale_", 0) == 0) {
        out = s.rawLogScale[i][std::stoi(name.substr(6))];
    } else if (name.rfind("rot_", 0) == 0) {
        out = s.rawRotation[i][std::stoi(name.substr(4))];
    } else if (name.rfind("f_dc_", 0) == 0) {
        out = s.shCoeff(i, 0)[std::stoi(name.substr(5))];
    } else if (name.rfind("f_rest_", 0) == 0) {
        // Channel-major in the file: f_rest[c*(K-1) + (k-1)] -> sh[k][c].
        const int j = std::stoi(name.substr(7));
        const uint32_t c = uint32_t(j) / (K - 1);
        const uint32_t k = uint32_t(j) % (K - 1) + 1;
        out = s.shCoeff(i, k)[c];
    } else {
        return false;  // nx/ny/nz: not kept
    }
    return true;
}

void checkSyntheticFile(const fs::path& path, const nlohmann::json& exp) {
    std::printf("[synthetic] %s\n", path.filename().string().c_str());
    GaussianScene s;
    try {
        s = gsr::loadGaussianPly(path);
    } catch (const std::exception& e) {
        CHECK(false, "load failed: %s", e.what());
        return;
    }
    const uint32_t n = exp["count"].get<uint32_t>();
    CHECK(s.count == n, "count %u, expected %u", s.count, n);
    CHECK(s.shDegree == exp["sh_degree"].get<int>(), "SH degree %d", s.shDegree);
    CHECK(s.sh.size() == size_t(n) * s.shCoeffsPerChannel() * 3, "sh array size %zu", s.sh.size());
    CHECK(s.warnings.empty(), "unexpected warning: %s", s.warnings.empty() ? "" : s.warnings[0].c_str());
    CHECK(gsr::defaultUpAxis(s.comments) == gsr::UpAxis::PosY, "synthetic files should default to +y up");
    if (s.count != n) return;

    // Full sample records: raw values must be bit-exact (float32 survives JSON round trips).
    for (const nlohmann::json& sample : exp["samples"]) {
        const uint32_t i = sample["index"].get<uint32_t>();
        for (const auto& [name, value] : sample["values"].items()) {
            float got = 0.0f;
            if (!rawValue(s, i, name, got)) continue;
            const float want = value.get<float>();
            CHECK(got == want, "gaussian %u %s = %.9g, expected %.9g", i, name.c_str(), got, want);
        }
        // Activations.
        const float op = s.rawOpacity[i];
        CHECK(near(s.posAlpha[i].w, 1.0 / (1.0 + std::exp(-double(op))), 1e-6), "alpha of %u", i);
        for (int a = 0; a < 3; ++a)
            CHECK(near(s.scales[i][a], std::exp(double(s.rawLogScale[i][a])), 1e-6 * s.scales[i][a]), "sigma of %u", i);
        const glm::vec4 q = s.rawRotation[i] / glm::length(s.rawRotation[i]);
        CHECK(near(glm::length(s.rotations[i]), 1.0, 1e-6), "|q| of %u", i);
        CHECK(glm::length(s.rotations[i] - q) < 1e-6f, "normalized q of %u", i);
    }

    // Sums over ALL records (catches errors outside the sampled records).
    const nlohmann::json& sums = exp["sums"];
    auto sumOf = [&](const std::function<double(uint32_t)>& f) {
        double total = 0.0, absTotal = 0.0;
        for (uint32_t i = 0; i < s.count; ++i) {
            const double v = f(i);
            total += v;
            absTotal += std::fabs(v);
        }
        return std::pair{total, absTotal};
    };
    for (const auto& [name, value] : sums.items()) {
        std::pair<double, double> got;
        if (name == "f_rest_all") {
            const uint32_t K = s.shCoeffsPerChannel();
            got = sumOf([&](uint32_t i) {
                double t = 0.0;
                for (uint32_t k = 1; k < K; ++k)
                    for (int c = 0; c < 3; ++c) t += s.shCoeff(i, k)[c];
                return t;
            });
        } else {
            got = sumOf([&](uint32_t i) {
                float v = 0.0f;
                rawValue(s, i, name, v);
                return double(v);
            });
        }
        const double want = value.get<double>();
        CHECK(near(got.first, want, 1e-9 * (1.0 + got.second)), "sum of %s = %.12g, expected %.12g", name.c_str(),
              got.first, want);
    }

    // Stats sanity.
    CHECK(s.stats.robustMin.x >= s.stats.bboxMin.x && s.stats.robustMax.x <= s.stats.bboxMax.x, "robust bbox inside bbox");
    CHECK(s.stats.alphaMin > 0.0f && s.stats.alphaMax < 1.0f, "alpha range");
    CHECK(s.stats.zeroQuaternions == 0 && s.stats.nonFiniteValues == 0, "no bad values");
}

void checkSyntheticSemantics(const fs::path& dir) {
    // Values the generator documents in words, independently of the JSON.
    const fs::path single = dir / "single_sh0.ply";
    if (fs::exists(single)) {
        std::printf("[semantics] single_sh0.ply\n");
        const GaussianScene s = gsr::loadGaussianPly(single);
        CHECK(near(s.posAlpha[0].w, 0.8, 1e-5), "alpha %.6f", s.posAlpha[0].w);
        CHECK(near(s.scales[0].x, 0.5, 1e-5) && near(s.scales[0].z, 0.5, 1e-5), "sigma");
        const glm::vec3 c = s.dcColor(0);
        CHECK(near(c.r, 1.0, 1e-5) && near(c.g, 0.5, 1e-5) && near(c.b, 0.1, 1e-5), "dc colour %f %f %f", c.r, c.g, c.b);
    }
    const fs::path shdir = dir / "shdir_sh3.ply";
    if (fs::exists(shdir)) {
        std::printf("[semantics] shdir_sh3.ply (SH transpose)\n");
        const GaussianScene s = gsr::loadGaussianPly(shdir);
        const uint32_t ks[] = {1, 2, 3, 4, 6, 9};
        for (uint32_t i = 0; i < 6 && i < s.count; ++i) {
            for (uint32_t k = 1; k < 16; ++k) {
                const glm::vec3 coeff = s.shCoeff(i, k);
                const glm::vec3 want = k == ks[i] ? glm::vec3(1.2f, 0.0f, -1.2f) : glm::vec3(0.0f);
                CHECK(glm::length(coeff - want) < 1e-6f, "gaussian %u coefficient %u = (%g %g %g)", i, k, coeff.r,
                      coeff.g, coeff.b);
            }
        }
    }
    const fs::path overlap = dir / "overlap_sh0.ply";
    if (fs::exists(overlap)) {
        std::printf("[semantics] overlap_sh0.ply (unnormalized quaternions)\n");
        const GaussianScene s = gsr::loadGaussianPly(overlap);
        CHECK(near(s.stats.quatNormMin, 2.0, 1e-5) && near(s.stats.quatNormMax, 2.0, 1e-5), "raw |q| = 2");
        CHECK(near(glm::length(s.rotations[3]), 1.0, 1e-6), "normalized");
    }
}

// --- Hand-written files ------------------------------------------------------------

struct PlyBuilder {
    std::string header;
    std::vector<uint8_t> payload;
    template <typename T>
    void put(T v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        payload.insert(payload.end(), p, p + sizeof(T));
    }
    fs::path write(const fs::path& path, size_t truncateTo = SIZE_MAX) const {
        std::vector<uint8_t> bytes(header.begin(), header.end());
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        if (truncateTo < bytes.size()) bytes.resize(truncateTo);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        return path;
    }
};

const char* kStandardDeg0Props =
    "property float x\nproperty float y\nproperty float z\n"
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";

std::string deg0Header(uint32_t count, const char* format = "binary_little_endian") {
    return std::string("ply\nformat ") + format + " 1.0\nelement vertex " + std::to_string(count) + "\n" +
           kStandardDeg0Props + "end_header\n";
}

void putDeg0Record(PlyBuilder& b, float base) {
    for (int j = 0; j < 14; ++j) b.put<float>(base + float(j));
}

void checkMixedTypes(const fs::path& scratch) {
    std::printf("[parser] mixed scalar types, unknown properties, element before vertex\n");
    PlyBuilder b;
    b.header =
        "ply\nformat binary_little_endian 1.0\ncomment written by test_ply_loader\n"
        "element extra_info 2\nproperty int a\nproperty uchar b\n"  // skipped: 2 x 5 bytes
        "element vertex 3\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property float nxx\nproperty uchar flags\nproperty ushort id\n"  // unknown names: skipped, counted
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property int rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "property float filter_3D\n"
        "end_header\n";
    for (int e = 0; e < 2; ++e) {
        b.put<int32_t>(-7);
        b.put<uint8_t>(0xAB);
    }
    for (int i = 0; i < 3; ++i) {
        b.put<double>(1.5 + i);
        b.put<double>(-2.25);
        b.put<double>(1e-3 * i);
        b.put<float>(99.0f);   // nxx
        b.put<uint8_t>(0xFF);  // flags
        b.put<uint16_t>(uint16_t(1000 + i));
        b.put<float>(0.1f);
        b.put<float>(0.2f);
        b.put<float>(0.3f);
        b.put<float>(0.0f);  // opacity logit 0 -> alpha 0.5
        b.put<float>(0.0f);
        b.put<float>(std::log(2.0f));
        b.put<float>(-1.0f);
        b.put<int32_t>(i == 2 ? 0 : 2);  // rot_0 as int; record 2 is a zero quaternion
        b.put<float>(0.0f);
        b.put<float>(0.0f);
        b.put<float>(0.0f);
        b.put<float>(i == 1 ? std::numeric_limits<float>::quiet_NaN() : 0.5f);  // filter_3D: skipped
    }
    const fs::path path = b.write(scratch / "mixed_types.ply");
    GaussianScene s;
    try {
        s = gsr::loadGaussianPly(path);
    } catch (const std::exception& e) {
        CHECK(false, "load failed: %s", e.what());
        return;
    }
    CHECK(s.count == 3 && s.shDegree == 0, "count %u degree %d", s.count, s.shDegree);
    CHECK(s.posAlpha[1].x == 2.5f && s.posAlpha[1].y == -2.25f, "double positions (%g %g)", s.posAlpha[1].x,
          s.posAlpha[1].y);
    CHECK(s.shCoeff(2, 0) == glm::vec3(0.1f, 0.2f, 0.3f), "dc after skipped properties");
    CHECK(near(s.posAlpha[0].w, 0.5, 1e-7), "alpha");
    CHECK(near(s.scales[0].y, 2.0, 1e-6) && near(s.scales[0].z, std::exp(-1.0), 1e-6), "sigma");
    CHECK(s.rawRotation[0].x == 2.0f && s.rotations[0] == glm::vec4(1, 0, 0, 0), "int quaternion normalized");
    CHECK(s.stats.zeroQuaternions == 1 && s.rotations[2] == glm::vec4(1, 0, 0, 0), "zero quaternion -> identity");
    CHECK(s.stats.nonFiniteValues == 0, "NaN in a skipped property must not count (%llu)",
          static_cast<unsigned long long>(s.stats.nonFiniteValues));
    CHECK(s.comments.size() == 1 && s.comments[0] == "written by test_ply_loader", "comments");
    CHECK(gsr::defaultUpAxis(s.comments) == gsr::UpAxis::NegY, "non-synthetic files default to -y up");

    // NaN in a used property is counted (and reported as a warning).
    PlyBuilder nan;
    nan.header = deg0Header(2);
    putDeg0Record(nan, 0.0f);
    for (int j = 0; j < 14; ++j) nan.put<float>(j == 1 ? std::numeric_limits<float>::infinity() : 0.5f);
    const GaussianScene sn = gsr::loadGaussianPly(nan.write(scratch / "nan.ply"));
    CHECK(sn.stats.nonFiniteValues == 1 && !sn.warnings.empty(), "non-finite count %llu",
          static_cast<unsigned long long>(sn.stats.nonFiniteValues));
}

void expectError(const char* what, const fs::path& path, const char* needle) {
    std::printf("[error] %s\n", what);
    try {
        gsr::loadGaussianPly(path);
        CHECK(false, "%s: loaded without error", what);
    } catch (const PlyError& e) {
        const bool found = std::string(e.what()).find(needle) != std::string::npos;
        CHECK(found, "message '%s' does not mention '%s'", e.what(), needle);
        std::printf("    -> %s\n", e.what());
    } catch (const std::exception& e) {
        CHECK(false, "%s: wrong exception type: %s", what, e.what());
    }
}

void checkErrors(const fs::path& scratch, const fs::path& syntheticDir) {
    // Truncated copy of a real synthetic file: keep the header and half the payload.
    const fs::path grid = syntheticDir / "grid_512_sh3.ply";
    if (fs::exists(grid)) {
        std::ifstream in(grid, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const size_t header = std::string_view(bytes.data(), bytes.size()).find("end_header\n") + 11;
        const size_t half = header + (bytes.size() - header) / 2;
        std::ofstream(scratch / "truncated_payload.ply", std::ios::binary).write(bytes.data(), std::streamsize(half));
        expectError("truncated payload", scratch / "truncated_payload.ply", "truncated");
        std::ofstream(scratch / "truncated_header.ply", std::ios::binary).write(bytes.data(), 200);
        expectError("truncated inside the header", scratch / "truncated_header.ply", "end_header");
        // One byte short.
        std::ofstream(scratch / "one_byte_short.ply", std::ios::binary)
            .write(bytes.data(), std::streamsize(bytes.size() - 1));
        expectError("one byte short", scratch / "one_byte_short.ply", "511 complete records");
    } else {
        CHECK(false, "missing %s", grid.string().c_str());
    }

    PlyBuilder b;
    b.header = deg0Header(1, "ascii");
    expectError("ascii", b.write(scratch / "ascii.ply"), "ASCII");

    b = {};
    b.header = deg0Header(1, "binary_big_endian");
    putDeg0Record(b, 0.0f);
    expectError("big endian", b.write(scratch / "big_endian.ply"), "big_endian");

    b = {};
    b.header =
        "ply\nformat binary_little_endian 1.0\nelement chunk 1\nproperty float min_x\n"
        "element vertex 1\nproperty uint packed_position\nproperty uint packed_rotation\n"
        "property uint packed_scale\nproperty uint packed_color\nend_header\n";
    for (int j = 0; j < 5; ++j) b.put<uint32_t>(0);
    expectError("PlayCanvas compressed PLY", b.write(scratch / "compressed.ply"), "compressed");

    // 10 f_rest properties: not 0/9/24/45.
    b = {};
    b.header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + std::string(kStandardDeg0Props);
    for (int j = 0; j < 10; ++j) b.header += "property float f_rest_" + std::to_string(j) + "\n";
    b.header += "end_header\n";
    for (int j = 0; j < 24; ++j) b.put<float>(0.0f);
    expectError("wrong f_rest count", b.write(scratch / "f_rest_10.ply"), "f_rest");

    // 9 f_rest properties but with a gap (f_rest_4 missing, f_rest_9 present).
    b = {};
    b.header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + std::string(kStandardDeg0Props);
    for (int j = 0; j < 10; ++j)
        if (j != 4) b.header += "property float f_rest_" + std::to_string(j) + "\n";
    b.header += "end_header\n";
    for (int j = 0; j < 23; ++j) b.put<float>(0.0f);
    expectError("f_rest gap", b.write(scratch / "f_rest_gap.ply"), "contiguous");

    b = {};
    b.header = deg0Header(1);
    b.header.erase(b.header.find("property float opacity\n"), 23);
    for (int j = 0; j < 13; ++j) b.put<float>(0.0f);
    expectError("missing property", b.write(scratch / "no_opacity.ply"), "'opacity'");

    b = {};
    b.header = "PLY\nformat binary_little_endian 1.0\nend_header\n";
    expectError("bad magic", b.write(scratch / "bad_magic.ply"), "not a PLY");

    b = {};
    b.header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty list uchar int vertex_indices\n" +
               std::string(kStandardDeg0Props) + "end_header\n";
    expectError("list property in vertex", b.write(scratch / "list.ply"), "list");

    b = {};
    b.header = deg0Header(0);
    expectError("zero gaussians", b.write(scratch / "empty.ply"), "empty");

    b = {};
    b.header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float16 x\nend_header\n";
    expectError("unknown type", b.write(scratch / "unknown_type.ply"), "unknown property type");

    b = {};
    b.header = "ply\nformat binary_little_endian 1.0\nelement face 1\nproperty float a\nend_header\n";
    b.put<float>(0.0f);
    expectError("no vertex element", b.write(scratch / "no_vertex.ply"), "no 'vertex'");
}

// --- Scene transform and training-camera helpers ------------------------------------

void checkTransforms() {
    std::printf("[transform] up axes, camera alignment, focus point\n");
    using gsr::UpAxis;
    const struct {
        UpAxis up;
        glm::vec3 dataUp;
    } cases[] = {{UpAxis::PosY, {0, 1, 0}}, {UpAxis::NegY, {0, -1, 0}}, {UpAxis::PosZ, {0, 0, 1}}, {UpAxis::NegZ, {0, 0, -1}}};
    for (const auto& c : cases) {
        const glm::mat4 m = gsr::sceneToWorld(c.up);
        CHECK(near(glm::determinant(glm::mat3(m)), 1.0, 1e-6), "%s is a rotation", gsr::upAxisName(c.up));
        CHECK(glm::length(glm::mat3(m) * c.dataUp - glm::vec3(0, 1, 0)) < 1e-6f, "%s maps up to +Y",
              gsr::upAxisName(c.up));
        CHECK(gsr::parseUpAxis(gsr::upAxisName(c.up)) == c.up, "name round trip");
    }
    // -y flip: (x, y, z) -> (x, -y, -z).
    CHECK(glm::vec3(gsr::sceneToWorld(UpAxis::NegY) * glm::vec4(1, 2, 3, 1)) == glm::vec3(1, -2, -3), "NegY flip");
    // Camera alignment: a level capture gives exactly the NegY flip; a tilted one maps its up to +Y.
    CHECK(gsr::sceneToWorldAligned({0, -1, 0}) == gsr::sceneToWorld(UpAxis::NegY), "level capture = NegY");
    const glm::vec3 tilted = glm::normalize(glm::vec3(0.075f, -0.765f, -0.639f));  // bonsai_30k
    const glm::mat4 a = gsr::sceneToWorldAligned(tilted);
    CHECK(glm::length(glm::mat3(a) * tilted - glm::vec3(0, 1, 0)) < 1e-5f, "tilted up -> +Y");
    CHECK(near(glm::determinant(glm::mat3(a)), 1.0, 1e-5), "aligned is a rotation");

    // Focus point: cameras on a circle looking at (1, 2, 3).
    std::vector<gsr::TrainingCamera> cams;
    for (int i = 0; i < 8; ++i) {
        gsr::TrainingCamera c;
        const float ang = float(i) * 0.785398f;
        c.position = glm::vec3(1, 2, 3) + glm::vec3(4.0f * std::cos(ang), 0.5f * float(i % 2), 4.0f * std::sin(ang));
        const glm::vec3 f = glm::normalize(glm::vec3(1, 2, 3) - c.position);
        const glm::vec3 r = glm::normalize(glm::cross(f, glm::vec3(0, -1, 0)));
        c.cameraToWorld = glm::mat3(r, glm::cross(f, r), f);  // columns: right, down, forward
        cams.push_back(c);
    }
    const glm::vec3 focus = gsr::cameraFocusPoint(cams);
    CHECK(glm::length(focus - glm::vec3(1, 2, 3)) < 1e-3f, "focus point (%g %g %g)", focus.x, focus.y, focus.z);
    CHECK(glm::length(gsr::meanCameraUp(cams) - glm::vec3(0, -1, 0)) < 0.1f, "mean up of y-down rig is -y");
}

// --- Real scenes (optional) -----------------------------------------------------------

void checkRealScenes(const fs::path& scenesDir) {
    struct Expected {
        const char* dir;
        uint32_t count;
    };
    const Expected scenes[] = {{"utsuho_plush", 53671}, {"train_7k", 559263}, {"bonsai_30k", 1244819}};
    for (const Expected& e : scenes) {
        const fs::path path = scenesDir / e.dir / "point_cloud.ply";
        if (!fs::exists(path)) {
            std::printf("[scene] %s: not present, skipped\n", e.dir);
            continue;
        }
        const GaussianScene s = gsr::loadGaussianPly(path);
        std::printf("[scene] %s: %u gaussians, SH %d, %.1f ms (decode %.1f ms)\n", e.dir, s.count, s.shDegree,
                    s.timings.totalMs, s.timings.decodeMs);
        CHECK(s.count == e.count && s.shDegree == 3, "%s: count %u degree %d", e.dir, s.count, s.shDegree);
        CHECK(s.warnings.empty(), "%s: warning %s", e.dir, s.warnings.empty() ? "" : s.warnings[0].c_str());
        CHECK(s.stats.zeroQuaternions == 0 && s.stats.nonFiniteValues == 0, "%s: bad values", e.dir);
        CHECK(gsr::defaultUpAxis(s.comments) == gsr::UpAxis::NegY, "%s: default up should be -y", e.dir);
        const fs::path camerasPath = path.parent_path() / "cameras.json";
        if (fs::exists(camerasPath)) {
            const auto cams = gsr::loadTrainingCameras(camerasPath);
            CHECK(!cams.empty() && cams[0].id == 0 && cams[0].fy > 0.0f, "%s: cameras.json", e.dir);
            CHECK(gsr::defaultUpAxis(s.comments, true) == gsr::UpAxis::Cameras, "%s: cameras default", e.dir);
            std::printf("        %zu training cameras\n", cams.size());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    fs::path syntheticDir, expectedPath, scratch, scenesDir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--synthetic-dir") syntheticDir = argv[i + 1];
        else if (arg == "--expected") expectedPath = argv[i + 1];
        else if (arg == "--scratch-dir") scratch = argv[i + 1];
        else if (arg == "--scenes-dir") scenesDir = argv[i + 1];
    }
    if (syntheticDir.empty() || expectedPath.empty() || scratch.empty()) {
        std::printf("usage: test_ply_loader --synthetic-dir D --expected J --scratch-dir S [--scenes-dir D]\n");
        return 2;
    }
    fs::create_directories(scratch);

    std::ifstream expectedFile(expectedPath);
    if (!expectedFile) {
        std::printf("cannot open %s\n", expectedPath.string().c_str());
        return 1;
    }
    const nlohmann::json expected = nlohmann::json::parse(expectedFile);

    // Every .ply in the directory must be known, and every known file must exist.
    std::set<std::string> seen;
    if (fs::is_directory(syntheticDir)) {
        for (const fs::directory_entry& entry : fs::directory_iterator(syntheticDir)) {
            if (entry.path().extension() != ".ply") continue;
            const std::string name = entry.path().filename().string();
            seen.insert(name);
            if (!expected.contains(name)) {
                CHECK(false, "%s is not part of the generator's standard set", name.c_str());
                continue;
            }
            checkSyntheticFile(entry.path(), expected[name]);
        }
    }
    for (const auto& [name, _] : expected.items())
        CHECK(seen.count(name) == 1, "%s missing from %s (regenerate: python -I tools/gen_synthetic_ply.py --all)",
              name.c_str(), syntheticDir.string().c_str());

    checkSyntheticSemantics(syntheticDir);
    checkMixedTypes(scratch);
    checkErrors(scratch, syntheticDir);
    checkTransforms();
    if (!scenesDir.empty()) checkRealScenes(scenesDir);

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
