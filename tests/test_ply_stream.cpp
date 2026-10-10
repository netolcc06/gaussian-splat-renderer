// Tests for the in-memory and streaming entry points of the PLY loader (web port, phase W1):
//
//   test_ply_stream --synthetic-dir data/synthetic --scratch-dir <dir> [--scenes-dir data/scenes]
//
// For every synthetic file and utsuho_plush (if present), the scene from
//   A  loadGaussianPly(path)                      (memory-mapped, multi-threaded: the reference)
//   B  loadGaussianPly(span, name)                (bytes already in memory)
//   C  PlyStreamDecoder, chunk sizes 1 B .. whole file, total size known or not, several batch
//      capacities, batches assembled into one scene
// must be bit-identical: count, degree, comments, warnings, every array (activated and raw),
// the stats and the NaN / zero-quaternion counters. Malformed files (truncated, trailing
// bytes, wrong format, ...) must give the same PlyError text (or the same warnings) on every
// path.
#include "scene/gaussian_scene.h"
#include "scene/ply_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using gsr::GaussianScene;

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

std::vector<uint8_t> readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeFile(const fs::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

std::vector<uint8_t> bytesOf(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

// Bitwise equality of two arrays of trivially copyable values (NaN-safe, -0 != +0).
template <typename T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
template <typename T>
bool sameBits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool sameStats(const gsr::SceneStats& a, const gsr::SceneStats& b) {
    return sameBits(a.bboxMin, b.bboxMin) && sameBits(a.bboxMax, b.bboxMax) && sameBits(a.robustMin, b.robustMin) &&
           sameBits(a.robustMax, b.robustMax) && sameBits(a.alphaMin, b.alphaMin) && sameBits(a.alphaMax, b.alphaMax) &&
           sameBits(a.alphaMean, b.alphaMean) && a.alphaBelowVisible == b.alphaBelowVisible &&
           sameBits(a.sigmaMin, b.sigmaMin) && sameBits(a.sigmaMax, b.sigmaMax) &&
           sameBits(a.sigmaMaxMedian, b.sigmaMaxMedian) && sameBits(a.quatNormMin, b.quatNormMin) &&
           sameBits(a.quatNormMax, b.quatNormMax) && a.zeroQuaternions == b.zeroQuaternions &&
           a.nonFiniteValues == b.nonFiniteValues;
}

// Everything except sourcePath and timings.
void compareScenes(const GaussianScene& ref, const GaussianScene& got, const std::string& what) {
    CHECK(ref.count == got.count && ref.shDegree == got.shDegree, "%s: count %u/%u degree %d/%d", what.c_str(),
          ref.count, got.count, ref.shDegree, got.shDegree);
    CHECK(ref.comments == got.comments, "%s: comments differ", what.c_str());
    CHECK(ref.warnings == got.warnings, "%s: warnings differ (%zu vs %zu)", what.c_str(), ref.warnings.size(),
          got.warnings.size());
    CHECK(sameBits(ref.posAlpha, got.posAlpha), "%s: posAlpha differs", what.c_str());
    CHECK(sameBits(ref.scales, got.scales), "%s: scales differ", what.c_str());
    CHECK(sameBits(ref.rotations, got.rotations), "%s: rotations differ", what.c_str());
    CHECK(sameBits(ref.sh, got.sh), "%s: sh differs", what.c_str());
    CHECK(sameBits(ref.rawOpacity, got.rawOpacity), "%s: rawOpacity differs", what.c_str());
    CHECK(sameBits(ref.rawLogScale, got.rawLogScale), "%s: rawLogScale differs", what.c_str());
    CHECK(sameBits(ref.rawRotation, got.rawRotation), "%s: rawRotation differs", what.c_str());
    CHECK(sameStats(ref.stats, got.stats), "%s: stats differ", what.c_str());
}

struct StreamResult {
    GaussianScene scene;
    bool threw = false;
    bool plyError = false;
    std::string error;
    uint32_t batches = 0;
    bool headerBeforeBatches = true;
    bool contiguous = true;  // batch `first` values follow each other
};

// Streams `bytes` in chunks of `chunk` bytes and assembles the batches into one scene.
StreamResult streamLoad(const std::vector<uint8_t>& bytes, const std::string& name, size_t chunk, bool knownSize,
                        uint32_t batchCapacity) {
    StreamResult r;
    bool headerSeen = false;
    uint32_t next = 0;
    GaussianScene& s = r.scene;
    auto onHeader = [&](const GaussianScene& info) {
        headerSeen = true;
        s.count = info.count;
        s.shDegree = info.shDegree;
        const size_t n = info.count;
        s.posAlpha.resize(n);
        s.scales.resize(n);
        s.rotations.resize(n);
        s.sh.resize(n * s.shCoeffsPerChannel() * 3);
        s.rawOpacity.resize(n);
        s.rawLogScale.resize(n);
        s.rawRotation.resize(n);
    };
    auto onBatch = [&](uint32_t first, const GaussianScene& b) {
        ++r.batches;
        r.headerBeforeBatches = r.headerBeforeBatches && headerSeen;
        r.contiguous = r.contiguous && first == next && b.count > 0 && b.count <= batchCapacity;
        next = first + b.count;
        if (next > s.count) {
            r.contiguous = false;
            return;
        }
        const uint32_t K = s.shCoeffsPerChannel();
        std::copy(b.posAlpha.begin(), b.posAlpha.end(), s.posAlpha.begin() + first);
        std::copy(b.scales.begin(), b.scales.end(), s.scales.begin() + first);
        std::copy(b.rotations.begin(), b.rotations.end(), s.rotations.begin() + first);
        std::copy(b.sh.begin(), b.sh.end(), s.sh.begin() + size_t(first) * K * 3);
        std::copy(b.rawOpacity.begin(), b.rawOpacity.end(), s.rawOpacity.begin() + first);
        std::copy(b.rawLogScale.begin(), b.rawLogScale.end(), s.rawLogScale.begin() + first);
        std::copy(b.rawRotation.begin(), b.rawRotation.end(), s.rawRotation.begin() + first);
    };
    try {
        gsr::PlyStreamDecoder decoder(name, knownSize ? bytes.size() : 0, onHeader, onBatch, batchCapacity);
        for (size_t at = 0; at < bytes.size(); at += chunk)
            decoder.push(std::span<const uint8_t>(bytes.data() + at, std::min(chunk, bytes.size() - at)));
        decoder.finish();
        r.contiguous = r.contiguous && next == s.count;
        const GaussianScene& info = decoder.info();
        s.comments = info.comments;
        s.warnings = info.warnings;
        // As loadGaussianPly: the stats from all arrays, then the decoder's counters.
        gsr::computeSceneStats(s);
        s.stats.nonFiniteValues = info.stats.nonFiniteValues;
        s.stats.zeroQuaternions = info.stats.zeroQuaternions;
        CHECK(decoder.recordsDecoded() == s.count && decoder.bytesReceived() == bytes.size(),
              "decoder counters %llu records, %llu bytes", static_cast<unsigned long long>(decoder.recordsDecoded()),
              static_cast<unsigned long long>(decoder.bytesReceived()));
    } catch (const gsr::PlyError& e) {
        r.threw = r.plyError = true;
        r.error = e.what();
    } catch (const std::exception& e) {
        r.threw = true;
        r.error = e.what();
    }
    return r;
}

struct StreamConfig {
    size_t chunk;
    bool knownSize;
    uint32_t batchCapacity;
};

std::vector<StreamConfig> configsFor(size_t fileSize) {
    std::vector<StreamConfig> configs;
    // 1-byte chunks only where they stay cheap (every push is a call).
    if (fileSize <= (32u << 20)) {
        configs.push_back({1, false, 16384});
        configs.push_back({1, true, 7});
    }
    configs.push_back({7, true, 16384});
    configs.push_back({7, false, 1});
    configs.push_back({4093, false, 1000});
    configs.push_back({1u << 20, true, 16384});
    configs.push_back({1u << 20, false, 3});
    configs.push_back({fileSize + 1, true, 16384});  // whole file in one push
    return configs;
}

std::string configText(const StreamConfig& c) {
    return "stream chunk " + std::to_string(c.chunk) + (c.knownSize ? " size known" : " size unknown") + " batch " +
           std::to_string(c.batchCapacity);
}

void checkValidFile(const fs::path& path) {
    const std::string name = path.filename().string();
    std::printf("[valid] %s\n", name.c_str());
    GaussianScene ref;
    try {
        ref = gsr::loadGaussianPly(path);
    } catch (const std::exception& e) {
        CHECK(false, "%s: reference load failed: %s", name.c_str(), e.what());
        return;
    }
    const std::vector<uint8_t> bytes = readFile(path);
    const GaussianScene fromMemory = gsr::loadGaussianPly(std::span<const uint8_t>(bytes), name);
    compareScenes(ref, fromMemory, name + " span");
    CHECK(fromMemory.sourcePath == name, "span sourcePath '%s'", fromMemory.sourcePath.c_str());

    for (const StreamConfig& c : configsFor(bytes.size())) {
        const StreamResult r = streamLoad(bytes, name, c.chunk, c.knownSize, c.batchCapacity);
        const std::string what = name + " " + configText(c);
        CHECK(!r.threw, "%s: threw %s", what.c_str(), r.error.c_str());
        if (r.threw) continue;
        CHECK(r.headerBeforeBatches && r.contiguous, "%s: batch order", what.c_str());
        const uint32_t expectedBatches = (ref.count + c.batchCapacity - 1) / c.batchCapacity;
        CHECK(r.batches == expectedBatches, "%s: %u batches, expected %u", what.c_str(), r.batches, expectedBatches);
        compareScenes(ref, r.scene, what);
    }
}

// Every path must fail with the same message (or succeed with the same warnings).
void checkSameOutcome(const std::string& what, const fs::path& scratch, const std::string& name,
                      const std::vector<uint8_t>& bytes) {
    std::printf("[malformed] %s\n", what.c_str());
    const fs::path path = scratch / name;
    writeFile(path, bytes);
    std::string refError;
    bool refThrew = false;
    GaussianScene ref;
    try {
        ref = gsr::loadGaussianPly(path);
    } catch (const gsr::PlyError& e) {
        refThrew = true;
        refError = e.what();
    }
    std::printf("    -> %s\n", refThrew ? refError.c_str() : (ref.warnings.empty() ? "loaded" : ref.warnings[0].c_str()));

    try {
        const GaussianScene m = gsr::loadGaussianPly(std::span<const uint8_t>(bytes), name);
        CHECK(!refThrew, "%s span: loaded, expected '%s'", what.c_str(), refError.c_str());
        if (!refThrew) compareScenes(ref, m, what + " span");
    } catch (const gsr::PlyError& e) {
        CHECK(refThrew && refError == e.what(), "%s span: '%s' vs '%s'", what.c_str(), e.what(), refError.c_str());
    }

    for (const StreamConfig& c : configsFor(bytes.size())) {
        const StreamResult r = streamLoad(bytes, name, c.chunk, c.knownSize, c.batchCapacity);
        const std::string label = what + " " + configText(c);
        if (refThrew) {
            CHECK(r.threw && r.plyError && r.error == refError, "%s: got '%s', expected '%s'", label.c_str(),
                  r.threw ? r.error.c_str() : "(no error)", refError.c_str());
        } else {
            CHECK(!r.threw, "%s: threw %s", label.c_str(), r.error.c_str());
            if (!r.threw) compareScenes(ref, r.scene, label);
        }
    }
}

void checkMalformed(const fs::path& scratch, const fs::path& syntheticDir) {
    const std::vector<uint8_t> grid = readFile(syntheticDir / "grid_512_sh3.ply");
    if (grid.empty()) {
        CHECK(false, "missing grid_512_sh3.ply");
        return;
    }
    const std::string text(grid.begin(), grid.end());
    const size_t header = text.find("end_header\n") + 11;

    auto prefix = [&](size_t n) { return std::vector<uint8_t>(grid.begin(), grid.begin() + std::ptrdiff_t(n)); };
    checkSameOutcome("truncated payload", scratch, "s_truncated_payload.ply", prefix(header + (grid.size() - header) / 2));
    checkSameOutcome("one byte short", scratch, "s_one_byte_short.ply", prefix(grid.size() - 1));
    checkSameOutcome("header only", scratch, "s_header_only.ply", prefix(header));
    checkSameOutcome("truncated inside the header", scratch, "s_truncated_header.ply", prefix(200));
    checkSameOutcome("empty file", scratch, "s_empty.ply", {});
    std::vector<uint8_t> garbage = grid;
    for (int i = 0; i < 1000; ++i) garbage.push_back(uint8_t(i * 37));
    checkSameOutcome("trailing garbage", scratch, "s_trailing.ply", garbage);
    std::vector<uint8_t> noise(5000);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = uint8_t((i * 2654435761u) >> 13);
    checkSameOutcome("random bytes", scratch, "s_noise.ply", noise);
    std::string crlf = text.substr(0, header);
    for (size_t p = crlf.find('\n'); p != std::string::npos; p = crlf.find('\n', p + 2)) crlf.insert(p, "\r");
    std::vector<uint8_t> crlfFile = bytesOf(crlf);
    crlfFile.insert(crlfFile.end(), grid.begin() + std::ptrdiff_t(header), grid.end());
    checkSameOutcome("CRLF header", scratch, "s_crlf.ply", crlfFile);

    const std::string props =
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    auto file = [&](const std::string& head, size_t payloadBytes) {
        std::vector<uint8_t> b = bytesOf(head);
        for (size_t i = 0; i < payloadBytes; ++i) b.push_back(uint8_t(i % 7 == 0 ? 0x3f : i));
        return b;
    };
    checkSameOutcome("ascii", scratch, "s_ascii.ply",
                     file("ply\nformat ascii 1.0\nelement vertex 1\n" + props + "end_header\n", 0));
    checkSameOutcome("big endian", scratch, "s_be.ply",
                     file("ply\nformat binary_big_endian 1.0\nelement vertex 1\n" + props + "end_header\n", 56));
    checkSameOutcome("missing property", scratch, "s_missing.ply",
                     file("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n", 4));
    // Elements around vertex: one before (skipped), one after (trailing); and a truncated one before.
    const std::string around = "ply\nformat binary_little_endian 1.0\ncomment c1\nelement extra 3\nproperty int a\n"
                               "element vertex 2\n" + props + "element after 2\nproperty short s\nend_header\n";
    checkSameOutcome("elements before and after vertex", scratch, "s_around.ply", file(around, 12 + 2 * 56 + 4));
    checkSameOutcome("elements after vertex truncated", scratch, "s_after_trunc.ply", file(around, 12 + 2 * 56 + 1));
    checkSameOutcome("truncated inside an element before vertex", scratch, "s_before_trunc.ply", file(around, 5));
    const std::string huge = "ply\nformat binary_little_endian 1.0\nelement extra 4611686018427387904\nproperty int a\n"
                             "element vertex 2\n" + props + "end_header\n";
    checkSameOutcome("absurd element count before vertex", scratch, "s_huge.ply", file(huge, 200));
    // Mixed scalar types (slow decode path) and a NaN.
    std::string mixed = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty double x\nproperty float y\n"
                        "property uchar z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                        "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                        "property int rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                        "end_header\n";
    std::vector<uint8_t> mixedFile = file(mixed, 3 * (8 + 4 + 1 + 10 * 4 + 4));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(mixedFile.data() + mixed.size() + 8, &nan, 4);  // y of record 0
    checkSameOutcome("mixed types and NaN", scratch, "s_mixed.ply", mixedFile);
}

}  // namespace

int main(int argc, char** argv) {
    fs::path syntheticDir, scratch, scenesDir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--synthetic-dir") syntheticDir = argv[i + 1];
        else if (arg == "--scratch-dir") scratch = argv[i + 1];
        else if (arg == "--scenes-dir") scenesDir = argv[i + 1];
    }
    if (syntheticDir.empty() || scratch.empty()) {
        std::printf("usage: test_ply_stream --synthetic-dir D --scratch-dir S [--scenes-dir D]\n");
        return 2;
    }
    fs::create_directories(scratch);

    int files = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(syntheticDir)) {
        if (entry.path().extension() != ".ply") continue;
        checkValidFile(entry.path());
        ++files;
    }
    CHECK(files >= 9, "only %d synthetic files in %s", files, syntheticDir.string().c_str());
    if (!scenesDir.empty()) {
        const fs::path utsuho = scenesDir / "utsuho_plush" / "point_cloud.ply";
        if (fs::exists(utsuho)) checkValidFile(utsuho);
        else std::printf("[valid] utsuho_plush: not present, skipped\n");
    }
    checkMalformed(scratch, syntheticDir);

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
