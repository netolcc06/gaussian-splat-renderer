#include "scene/scene_compression.h"

#include "scene/half.h"
#include "scene/parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace gsr {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

uint32_t vec4sFor(uint32_t bytes) { return (bytes + 15u) / 16u; }

// SH band of coefficient k >= 1: 1 (k 1-3), 2 (k 4-8), 3 (k 9-15).
uint32_t bandOf(uint32_t k) { return k < 4 ? 1u : k < 9 ? 2u : 3u; }

void putHalf(std::vector<uint32_t>& words, size_t wordBase, uint32_t halfIndex, float value) {
    uint32_t& w = words[wordBase + halfIndex / 2];
    const uint32_t h = floatToHalf(value);
    w = (halfIndex & 1u) ? ((w & 0x0000FFFFu) | (h << 16)) : ((w & 0xFFFF0000u) | h);
}

float getHalf(const uint32_t* words, uint32_t halfIndex) {
    const uint32_t w = words[halfIndex / 2];
    return (halfIndex & 1u) ? unpackHalfHi(w) : unpackHalfLo(w);
}

void encodeGeometry(GeometryEncoding g, const glm::vec4& pa, const glm::vec3& sigma, const glm::vec4& q,
                    uint32_t* w) {
    if (g == GeometryEncoding::RecordF32) {
        const float v[12] = {pa.x, pa.y, pa.z, pa.w, sigma.x, sigma.y, sigma.z, 0.0f, q.x, q.y, q.z, q.w};
        for (int i = 0; i < 12; ++i) w[i] = floatToBits(v[i]);
    } else {  // RecordF16
        w[0] = floatToBits(pa.x);
        w[1] = floatToBits(pa.y);
        w[2] = floatToBits(pa.z);
        w[3] = packHalf2(pa.w, sigma.x);
        w[4] = packHalf2(sigma.y, sigma.z);
        w[5] = packHalf2(q.x, q.y);  // (w, x) of the quaternion (GaussianScene: real part first)
        w[6] = packHalf2(q.z, q.w);  // (y, z)
        w[7] = 0;
    }
}

}  // namespace

float shBandMaxContribution(const glm::vec3* sh, int band) {
    if (band <= 0) return 0.0f;
    const int k0 = band * band, k1 = (band + 1) * (band + 1);
    glm::vec3 sq(0.0f);
    for (int k = k0; k < k1; ++k) sq += sh[k] * sh[k];
    const float norm = std::sqrt(std::max(sq.x, std::max(sq.y, sq.z)));
    constexpr float kInv4Pi = 0.07957747154594767f;
    return std::sqrt(float(2 * band + 1) * kInv4Pi) * norm;
}

int selectShDegree(const glm::vec3* sh, int storedDegree, float opacity, float threshold255) {
    const float threshold = threshold255 / 255.0f;
    for (int band = storedDegree; band >= 1; --band)
        if (opacity * shBandMaxContribution(sh, band) > threshold) return band;  // threshold 0: keep non-zero bands
    return 0;
}

const char* sceneFormatName(SceneFormat f) {
    switch (f) {
        case SceneFormat::F32: return "f32";
        case SceneFormat::F16: return "f16";
        case SceneFormat::Q8: return "q8";
        case SceneFormat::Codebook: return "cb";
    }
    return "?";
}

const char* sceneLayoutName(SceneLayout l) {
    switch (l) {
        case SceneLayout::Soa: return "soa";
        case SceneLayout::Split: return "split";
        case SceneLayout::Aos: return "aos";
    }
    return "?";
}

std::optional<SceneFormat> parseSceneFormat(std::string_view t) {
    if (t == "f32") return SceneFormat::F32;
    if (t == "f16") return SceneFormat::F16;
    if (t == "q8") return SceneFormat::Q8;
    if (t == "cb" || t == "codebook") return SceneFormat::Codebook;
    return std::nullopt;
}

std::optional<SceneLayout> parseSceneLayout(std::string_view t) {
    if (t == "soa") return SceneLayout::Soa;
    if (t == "split") return SceneLayout::Split;
    if (t == "aos") return SceneLayout::Aos;
    return std::nullopt;
}

uint32_t geometryRecordBytes(GeometryEncoding g) {
    switch (g) {
        case GeometryEncoding::SoaF32: return 48;
        case GeometryEncoding::RecordF32: return 48;
        case GeometryEncoding::RecordF16: return 32;
    }
    return 0;
}

uint32_t colorRecordBytes(ColorEncoding c, uint32_t k) {
    switch (c) {
        case ColorEncoding::SoaF32: return k * 3 * 4;
        case ColorEncoding::RecordF32: return vec4sFor(k * 3 * 4) * 16;
        case ColorEncoding::RecordF16: return vec4sFor(k * 3 * 2) * 16;
        case ColorEncoding::Q8: return vec4sFor(12 + (k - 1) * 3) * 16;
        case ColorEncoding::Codebook: return 8;
    }
    return 0;
}

size_t EncodedScene::totalBytes() const {
    size_t b = 0;
    for (const auto& v : buffers) b += v.size() * 4;
    return b;
}

uint16_t quantizeBandQ8(const float* v, uint32_t n, int8_t* q) {
    float m = 0.0f;
    for (uint32_t i = 0; i < n; ++i)
        if (std::isfinite(v[i])) m = std::max(m, std::abs(v[i]));
    if (m == 0.0f) {
        std::fill(q, q + n, int8_t(0));
        return 0;
    }
    // Smallest half scale s with 127 s >= m: then |v / s| <= 127 and rounding never clips.
    uint16_t h = floatToHalf(m / 127.0f);
    if (h == 0) h = 1;
    while (halfToFloat(h) * 127.0f < m) ++h;  // positive halves are ordered like their bits
    const float s = halfToFloat(h);
    for (uint32_t i = 0; i < n; ++i) {
        const float x = std::isfinite(v[i]) ? v[i] / s : 0.0f;
        q[i] = int8_t(std::clamp(std::lround(x), -127L, 127L));
    }
    return h;
}

EncodedScene encodeScene(const GaussianScene& scene, const CompressionOptions& opt) {
    const auto start = Clock::now();
    EncodedScene e;
    e.format = opt.format;
    e.layout = opt.layout;
    if (opt.format != SceneFormat::F32 && opt.layout == SceneLayout::Soa)
        throw std::runtime_error("scene layout 'soa' exists only for the f32 format (use split or aos)");
    e.count = scene.count;
    e.shDegree = std::clamp(std::min(scene.shDegree, opt.maxShDegree), 0, 3);
    e.shCoeffs = uint32_t((e.shDegree + 1) * (e.shDegree + 1));
    const uint32_t n = scene.count, K = e.shCoeffs, srcK = scene.shCoeffsPerChannel();

    auto position = [&](uint32_t i) {
        glm::vec4 pa = scene.posAlpha[i];
        if (opt.fp16Positions) {
            pa.x = roundToHalf(pa.x);
            pa.y = roundToHalf(pa.y);
            pa.z = roundToHalf(pa.z);
        }
        return pa;
    };
    // Phase 6: per-gaussian SH degree; dropped coefficients are encoded as 0.
    const bool variable = opt.shThreshold >= 0.0f && e.shDegree > 0;
    std::vector<uint8_t> degrees;
    if (variable) {
        degrees.resize(n);
        parallelChunks(n, opt.threads, [&](uint32_t begin, uint32_t end) {
            std::vector<glm::vec3> c(K);
            for (uint32_t i = begin; i < end; ++i) {
                for (uint32_t k = 0; k < K; ++k)
                    c[k] = {scene.sh[(size_t(i) * srcK + k) * 3], scene.sh[(size_t(i) * srcK + k) * 3 + 1],
                            scene.sh[(size_t(i) * srcK + k) * 3 + 2]};
                degrees[i] = uint8_t(selectShDegree(c.data(), e.shDegree, scene.posAlpha[i].w, opt.shThreshold));
            }
        });
        for (uint8_t d : degrees) e.degreeHistogram[d]++;
    }
    auto keptCoeffs = [&](uint32_t i) { return variable ? uint32_t((degrees[i] + 1) * (degrees[i] + 1)) : K; };
    auto shValue = [&](uint32_t i, uint32_t k, uint32_t c) {
        return k < keptCoeffs(i) ? scene.sh[(size_t(i) * srcK + k) * 3 + c] : 0.0f;
    };

    // --- f32 SoA: the Phase 1 arrays (truncated to the stored degree) -----------------
    if (opt.layout == SceneLayout::Soa) {
        e.geometry = GeometryEncoding::SoaF32;
        e.color = ColorEncoding::SoaF32;
        e.buffers.assign(4, {});
        e.bufferNames = {"gaussians: posAlpha", "gaussians: scales", "gaussians: rotations", "gaussians: sh"};
        for (int b = 0; b < 3; ++b) e.buffers[b].resize(size_t(n) * 4);
        // Variable degree: sh[] compacted, gaussian i's (d+1)^2 RGB triplets at shOffset[i].
        std::vector<uint32_t> shOffset(n);
        size_t shFloats = size_t(n) * K * 3;
        if (variable) {
            e.variableSh = VariableSh::Compact;
            size_t o = 0;
            for (uint32_t i = 0; i < n; ++i) {
                shOffset[i] = uint32_t(o);
                o += size_t(keptCoeffs(i)) * 3;
            }
            if (o >= (1ull << 30)) throw std::runtime_error("variable SH: compacted array exceeds 2^30 floats");
            shFloats = o;
        }
        e.buffers[3].resize(shFloats);
        parallelChunks(n, opt.threads, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            const glm::vec4 pa = position(i);
            for (int c = 0; c < 4; ++c) {
                e.buffers[0][size_t(i) * 4 + c] = floatToBits(pa[c]);
                e.buffers[1][size_t(i) * 4 + c] = floatToBits(scene.scales[i][c]);
                e.buffers[2][size_t(i) * 4 + c] = floatToBits(scene.rotations[i][c]);
            }
            if (variable) {
                e.buffers[1][size_t(i) * 4 + 3] = (uint32_t(degrees[i]) << 30) | shOffset[i];
                for (uint32_t k = 0; k < keptCoeffs(i); ++k)
                    for (uint32_t c = 0; c < 3; ++c) e.buffers[3][shOffset[i] + k * 3 + c] = floatToBits(shValue(i, k, c));
                continue;
            }
            for (uint32_t k = 0; k < K; ++k)
                for (uint32_t c = 0; c < 3; ++c) e.buffers[3][(size_t(i) * K + k) * 3 + c] = floatToBits(shValue(i, k, c));
        }
        });
        e.encodeMs = msSince(start);
        return e;
    }

    // --- Record formats -----------------------------------------------------------------
    switch (opt.format) {
        case SceneFormat::F32: e.geometry = GeometryEncoding::RecordF32; e.color = ColorEncoding::RecordF32; break;
        case SceneFormat::F16: e.geometry = GeometryEncoding::RecordF16; e.color = ColorEncoding::RecordF16; break;
        case SceneFormat::Q8: e.geometry = GeometryEncoding::RecordF16; e.color = ColorEncoding::Q8; break;
        case SceneFormat::Codebook: e.geometry = GeometryEncoding::RecordF16; e.color = ColorEncoding::Codebook; break;
    }
    const uint32_t geomWords = vec4sFor(geometryRecordBytes(e.geometry)) * 4;
    const bool cbSplit = e.color == ColorEncoding::Codebook && opt.layout == SceneLayout::Split;
    // Colour words per record: CB split packs two 8-byte records per uvec4.
    const uint32_t colorWords = cbSplit ? 2u : vec4sFor(colorRecordBytes(e.color, K)) * 4;

    std::vector<uint32_t> geom(size_t(n) * geomWords, 0u);
    std::vector<uint32_t> color(size_t(n) * colorWords, 0u);

    // Codebook: k-means over the SH-rest vectors (K-1)*3.
    std::vector<uint32_t> cbIndex;
    std::vector<float> centroids;
    const uint32_t restDims = (K - 1) * 3;
    uint32_t cbEntries = 0;
    if (e.color == ColorEncoding::Codebook && restDims > 0 && n > 0) {
        std::vector<float> points(size_t(n) * restDims);
        parallelChunks(n, opt.threads, [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i)
                for (uint32_t k = 1; k < K; ++k)
                    for (uint32_t c = 0; c < 3; ++c) {
                        const float v = shValue(i, k, c);
                        points[size_t(i) * restDims + (k - 1) * 3 + c] = std::isfinite(v) ? v : 0.0f;
                    }
        });
        KMeansOptions ko;
        ko.k = std::clamp(opt.codebookSize, 1u, 65536u);
        ko.iterations = opt.codebookIterations;
        ko.samples = opt.codebookSamples;
        ko.threads = opt.threads;
        KMeansResult km = kmeans(points.data(), n, restDims, ko);
        cbIndex = std::move(km.assignment);
        centroids = std::move(km.centroids);
        e.codebook = km.stats;
        cbEntries = km.stats.entries;
    }

    parallelChunks(n, opt.threads, [&](uint32_t begin, uint32_t end) {
    for (uint32_t i = begin; i < end; ++i) {
        encodeGeometry(e.geometry, position(i), glm::vec3(scene.scales[i]), scene.rotations[i],
                       &geom[size_t(i) * geomWords]);
        const size_t cb = size_t(i) * colorWords;
        switch (e.color) {
            case ColorEncoding::RecordF32:
                for (uint32_t k = 0; k < K; ++k)
                    for (uint32_t c = 0; c < 3; ++c) color[cb + k * 3 + c] = floatToBits(shValue(i, k, c));
                break;
            case ColorEncoding::RecordF16:
                for (uint32_t k = 0; k < K; ++k)
                    for (uint32_t c = 0; c < 3; ++c) putHalf(color, cb, k * 3 + c, shValue(i, k, c));
                break;
            case ColorEncoding::Q8: {
                color[cb + 0] = packHalf2(shValue(i, 0, 0), shValue(i, 0, 1));
                uint16_t scales[4] = {0, 0, 0, 0};
                int8_t q[45] = {};
                for (uint32_t band = 1; band * band < keptCoeffs(i); ++band) {
                    const uint32_t k0 = band * band, k1 = (band + 1) * (band + 1);  // [k0, k1)
                    float values[21];
                    uint32_t m = 0;
                    for (uint32_t k = k0; k < k1; ++k)
                        for (uint32_t c = 0; c < 3; ++c) values[m++] = shValue(i, k, c);
                    scales[band] = quantizeBandQ8(values, m, &q[(k0 - 1) * 3]);
                }
                color[cb + 1] = floatToHalf(shValue(i, 0, 2)) | (uint32_t(scales[1]) << 16);
                color[cb + 2] = uint32_t(scales[2]) | (uint32_t(scales[3]) << 16);
                for (uint32_t j = 0; j < restDims; ++j)
                    color[cb + 3 + j / 4] |= uint32_t(uint8_t(q[j])) << (8 * (j % 4));
                break;
            }
            case ColorEncoding::Codebook: {
                const uint32_t index = cbIndex.empty() ? 0u : cbIndex[i];
                color[cb + 0] = packHalf2(shValue(i, 0, 0), shValue(i, 0, 1));
                color[cb + 1] = floatToHalf(shValue(i, 0, 2)) | (index << 16);
                break;
            }
            case ColorEncoding::SoaF32: break;
        }
    }
    });

    // Phase 6 variable degree: degree (+ compacted colour offset) in geometry word 7.
    if (variable) {
        const bool compact = opt.layout == SceneLayout::Split && e.color != ColorEncoding::Codebook;
        e.variableSh = compact ? VariableSh::Compact : VariableSh::DegreeOnly;
        if (compact) {
            std::vector<uint32_t> packed;
            packed.reserve(color.size());
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t words = vec4sFor(colorRecordBytes(e.color, keptCoeffs(i))) * 4;
                if (packed.size() / 4 >= (1ull << 30)) throw std::runtime_error("variable SH: colour buffer too large");
                geom[size_t(i) * geomWords + 7] = (uint32_t(degrees[i]) << 30) | uint32_t(packed.size() / 4);
                packed.insert(packed.end(), color.begin() + std::ptrdiff_t(size_t(i) * colorWords),
                              color.begin() + std::ptrdiff_t(size_t(i) * colorWords + words));
            }
            color = std::move(packed);
        } else {
            for (uint32_t i = 0; i < n; ++i) geom[size_t(i) * geomWords + 7] = uint32_t(degrees[i]) << 30;
        }
    }

    std::vector<uint32_t> codebookWords;
    if (e.color == ColorEncoding::Codebook) {
        const uint32_t entryWords = std::max(4u, vec4sFor(restDims * 4) * 4);
        e.codebookStride = entryWords / 4;
        codebookWords.assign(size_t(std::max(cbEntries, 1u)) * entryWords, 0u);
        for (uint32_t c = 0; c < cbEntries; ++c)
            for (uint32_t d = 0; d < restDims; ++d)
                codebookWords[size_t(c) * entryWords + d] = floatToBits(centroids[size_t(c) * restDims + d]);
    }

    const std::string fmt = sceneFormatName(opt.format);
    if (opt.layout == SceneLayout::Split) {
        e.geometryStride = geomWords / 4;
        e.colorStride = cbSplit ? 0u : colorWords / 4;
        e.colorOffset = 0;
        e.buffers.push_back(std::move(geom));
        e.buffers.push_back(std::move(color));
        e.bufferNames = {"gaussians: geometry records (" + fmt + ")", "gaussians: colour records (" + fmt + ")"};
    } else {  // Aos: geometry record then colour record, one buffer
        const uint32_t recordWords = geomWords + colorWords;
        std::vector<uint32_t> records(size_t(n) * recordWords);
        parallelChunks(n, opt.threads, [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                std::copy_n(&geom[size_t(i) * geomWords], geomWords, &records[size_t(i) * recordWords]);
                std::copy_n(&color[size_t(i) * colorWords], colorWords, &records[size_t(i) * recordWords + geomWords]);
            }
        });
        e.geometryStride = recordWords / 4;
        e.colorStride = recordWords / 4;
        e.colorOffset = geomWords / 4;
        e.buffers.push_back(std::move(records));
        e.buffers.emplace_back();  // the GPU binds the record buffer for both
        e.bufferNames = {"gaussians: records (" + fmt + ", aos)", ""};
    }
    if (e.color == ColorEncoding::Codebook) {
        e.buffers.push_back(std::move(codebookWords));
        e.bufferNames.push_back("gaussians: SH codebook");
    }
    e.encodeMs = msSince(start);
    return e;
}

void decodeGaussian(const EncodedScene& e, uint32_t i, DecodedGaussian& out) {
    uint32_t K = e.shCoeffs;
    out.sh.assign(K, glm::vec3(0.0f));
    if (e.geometry == GeometryEncoding::SoaF32) {
        const uint32_t* pa = &e.buffers[0][size_t(i) * 4];
        const uint32_t* sc = &e.buffers[1][size_t(i) * 4];
        const uint32_t* ro = &e.buffers[2][size_t(i) * 4];
        out.posAlpha = {bitsToFloat(pa[0]), bitsToFloat(pa[1]), bitsToFloat(pa[2]), bitsToFloat(pa[3])};
        out.sigma = {bitsToFloat(sc[0]), bitsToFloat(sc[1]), bitsToFloat(sc[2])};
        out.rotation = {bitsToFloat(ro[0]), bitsToFloat(ro[1]), bitsToFloat(ro[2]), bitsToFloat(ro[3])};
        size_t base = size_t(i) * K * 3;
        if (e.variableSh != VariableSh::Off) {
            const uint32_t d = sc[3] >> 30;
            K = std::min(K, (d + 1) * (d + 1));
            base = sc[3] & 0x3FFFFFFFu;
        }
        const uint32_t* sh = e.buffers[3].data() + base;
        for (uint32_t k = 0; k < K; ++k)
            out.sh[k] = {bitsToFloat(sh[k * 3]), bitsToFloat(sh[k * 3 + 1]), bitsToFloat(sh[k * 3 + 2])};
        return;
    }
    const uint32_t* g = &e.buffers[0][size_t(i) * e.geometryStride * 4];
    if (e.geometry == GeometryEncoding::RecordF32) {
        out.posAlpha = {bitsToFloat(g[0]), bitsToFloat(g[1]), bitsToFloat(g[2]), bitsToFloat(g[3])};
        out.sigma = {bitsToFloat(g[4]), bitsToFloat(g[5]), bitsToFloat(g[6])};
        out.rotation = {bitsToFloat(g[8]), bitsToFloat(g[9]), bitsToFloat(g[10]), bitsToFloat(g[11])};
    } else {
        out.posAlpha = {bitsToFloat(g[0]), bitsToFloat(g[1]), bitsToFloat(g[2]), unpackHalfLo(g[3])};
        out.sigma = {unpackHalfHi(g[3]), unpackHalfLo(g[4]), unpackHalfHi(g[4])};
        const glm::vec4 q(unpackHalfLo(g[5]), unpackHalfHi(g[5]), unpackHalfLo(g[6]), unpackHalfHi(g[6]));
        // Same expression as the shader's normalize(): q * inversesqrt(dot(q, q)).
        const float len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        out.rotation = len2 > 0.0f ? q * (1.0f / std::sqrt(len2)) : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    }

    const std::vector<uint32_t>& colorBuffer = e.layout == SceneLayout::Aos ? e.buffers[0] : e.buffers[1];
    // The colour record: fixed stride, or (Phase 6 compacted variable degree) at the offset in word 7.
    size_t colorWord = e.colorStride == 0 ? size_t(i) * 2 : (size_t(i) * e.colorStride + e.colorOffset) * 4;
    if (e.variableSh != VariableSh::Off) {
        const uint32_t meta = g[7];
        const uint32_t d = meta >> 30;
        K = std::min(K, (d + 1) * (d + 1));
        if (e.variableSh == VariableSh::Compact) colorWord = size_t(meta & 0x3FFFFFFFu) * 4;
    }
    const uint32_t* c = colorBuffer.data() + colorWord;
    switch (e.color) {
        case ColorEncoding::RecordF32:
            for (uint32_t k = 0; k < K; ++k)
                out.sh[k] = {bitsToFloat(c[k * 3]), bitsToFloat(c[k * 3 + 1]), bitsToFloat(c[k * 3 + 2])};
            break;
        case ColorEncoding::RecordF16:
            for (uint32_t k = 0; k < K; ++k)
                out.sh[k] = {getHalf(c, k * 3), getHalf(c, k * 3 + 1), getHalf(c, k * 3 + 2)};
            break;
        case ColorEncoding::Q8: {
            out.sh[0] = {unpackHalfLo(c[0]), unpackHalfHi(c[0]), unpackHalfLo(c[1])};
            const float scale[4] = {0.0f, unpackHalfHi(c[1]), unpackHalfLo(c[2]), unpackHalfHi(c[2])};
            for (uint32_t k = 1; k < K; ++k)
                for (uint32_t ch = 0; ch < 3; ++ch) {
                    const uint32_t j = (k - 1) * 3 + ch;
                    const int8_t q = int8_t(uint8_t(c[3 + j / 4] >> (8 * (j % 4))));
                    out.sh[k][int(ch)] = float(q) * scale[bandOf(k)];
                }
            break;
        }
        case ColorEncoding::Codebook: {
            out.sh[0] = {unpackHalfLo(c[0]), unpackHalfHi(c[0]), unpackHalfLo(c[1])};
            const uint32_t index = c[1] >> 16;
            if (K > 1) {
                const uint32_t* entry = &e.buffers[2][size_t(index) * e.codebookStride * 4];
                for (uint32_t k = 1; k < K; ++k)
                    for (uint32_t ch = 0; ch < 3; ++ch)
                        out.sh[k][int(ch)] = bitsToFloat(entry[(k - 1) * 3 + ch]);
            }
            break;
        }
        case ColorEncoding::SoaF32: break;
    }
}

GaussianScene decodeScene(const EncodedScene& e, const GaussianScene& source) {
    GaussianScene s;
    s.sourcePath = source.sourcePath;
    s.count = e.count;
    s.shDegree = e.shDegree;
    s.stats = source.stats;
    s.timings = source.timings;
    s.posAlpha.resize(e.count);
    s.scales.resize(e.count);
    s.rotations.resize(e.count);
    s.sh.resize(size_t(e.count) * e.shCoeffs * 3);
    parallelChunks(e.count, 0, [&](uint32_t begin, uint32_t end) {
        DecodedGaussian g;
        for (uint32_t i = begin; i < end; ++i) {
            decodeGaussian(e, i, g);
            s.posAlpha[i] = g.posAlpha;
            s.scales[i] = glm::vec4(g.sigma, 0.0f);
            s.rotations[i] = g.rotation;
            for (uint32_t k = 0; k < e.shCoeffs; ++k)
                for (int c = 0; c < 3; ++c) s.sh[(size_t(i) * e.shCoeffs + k) * 3 + c] = g.sh[k][c];
        }
    });
    return s;
}

}  // namespace gsr
