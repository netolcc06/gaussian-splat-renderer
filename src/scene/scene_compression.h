// Scene compression (Phase 5): GPU-ready encodings of a GaussianScene that trade precision
// for memory and bandwidth. No Vulkan here: the encoder runs at load time on the CPU, the
// GPU only decodes (shaders/scene_decode.glsl, which mirrors decodeGaussian() below).
//
// A gaussian = GEOMETRY (position, opacity, scale, rotation) + COLOUR (SH coefficients).
// Formats (`--scene-format`):
//
//   format  geometry                                   colour                                    B/gaussian (SH3)
//   f32     posAlpha, sigma, quaternion: fp32 (48 B)   SH: 48 fp32 (192 B)                       240
//   f16     position fp32; alpha, sigma, quaternion    SH: 48 fp16 (96 B)                        128
//           fp16 (32 B record, 4 B spare)
//   q8      as f16                                     DC fp16; rest int8 with one fp16 scale    96
//                                                      per gaussian and band (64 B)
//   cb      as f16                                     DC fp16 + 16-bit index into a k-means     40 (+ codebook)
//                                                      codebook of SH-rest vectors (8 B)
//
// Positions stay fp32 in every format (docs/notes/phase5.md: fp16 positions move splats by
// up to |x| * 2^-11, e.g. 6 mm at 12 m, visible as pixel-sized jitter on large scenes).
//
// Layouts (`--scene-layout`), for the preprocess's access pattern:
//   soa    f32 only: the Phase 1-4 arrays (posAlpha, scales, rotations, sh), SH fetched one
//          float at a time. The reference path.
//   split  two arrays: one geometry record and one colour record per gaussian (16-byte
//          aligned, fetched as uvec4). Gaussians culled early never touch their colour.
//   aos    one record per gaussian = geometry record followed by the colour record.
//
// Record layouts (32-bit words; "h(a,b)" = packHalf2x16(a, b), a in the low 16 bits):
//   G32  (12 words): pos.xyz, alpha | sigma.xyz, 0 | q.wxyz           (q stored w first, as GaussianScene)
//   G16  (8 words):  pos.x, pos.y, pos.z, h(alpha, sigma.x), h(sigma.y, sigma.z),
//                    h(q.w, q.x), h(q.y, q.z), 0                     (q renormalized after decoding)
//   S32  (K*3 floats, padded to 4):   coefficient k, channel c at float k*3 + c
//   S16  (K*3 halves, padded to 8):   coefficient k, channel c at half k*3 + c
//   Q8   word 0 h(dc.r, dc.g), word 1 h(dc.b, s1), word 2 h(s2, s3), then (K-1)*3 int8 at
//        byte 12 + (k-1)*3 + c; value = int8 * s_band(k) (band 1: k 1-3, band 2: 4-8, band 3: 9-15)
//   CB   word 0 h(dc.r, dc.g), word 1 h(dc.b, 0) | index << 16; rest(k, c) = codebook[index][(k-1)*3 + c]
//        (codebook entries: (K-1)*3 floats padded to 4, fp32). Split layout: 8 B per gaussian
//        (two per uvec4); aos layout: the record is padded to 48 B.
// K = (stored degree + 1)^2. Records are padded to 16 bytes.
//
// Phase 6, variable SH degree (`--sh-threshold x`): each gaussian keeps the SH bands whose
// largest possible contribution to a pixel, opacity * max_dir |sum_m c_lm Y_lm(dir)| (bound:
// opacity * sqrt((2l+1)/(4 pi)) * |c_l|, per channel), exceeds x/255 (x = 0: non-zero bands); higher bands are
// dropped (stored as zero / not stored). The degree d (2 bits) and, for the compacted
// layouts, the offset of the gaussian's colour data (30 bits) live in a spare 32-bit word:
//   soa:   scales[i].w = d << 30 | offset in floats of the compacted sh[] array
//   split: geometry record word 7 (G32: after sigma.z; G16: the spare word)
//          = d << 30 | offset in uvec4s of the compacted colour buffer (f32/f16/q8; the
//          colour record of degree d is the degree-3 record truncated to (d+1)^2 coefficients)
//   aos, cb: word 7 = d << 30 (degree only: fixed-size records, the fetch is skipped)
#pragma once

#include "scene/gaussian_scene.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gsr {

enum class SceneFormat : uint32_t { F32 = 0, F16 = 1, Q8 = 2, Codebook = 3 };
enum class SceneLayout : uint32_t { Soa = 0, Split = 1, Aos = 2 };

const char* sceneFormatName(SceneFormat f);
const char* sceneLayoutName(SceneLayout l);
std::optional<SceneFormat> parseSceneFormat(std::string_view text);
std::optional<SceneLayout> parseSceneLayout(std::string_view text);

// Encodings of the two halves of a gaussian; the GPU decoder is specialized on these
// (specialization constants, see shaders/scene_decode.glsl). Values are part of the
// shader interface.
enum class GeometryEncoding : uint32_t { SoaF32 = 0, RecordF32 = 1, RecordF16 = 2 };
enum class ColorEncoding : uint32_t { SoaF32 = 0, RecordF32 = 1, RecordF16 = 2, Q8 = 3, Codebook = 4 };

struct CompressionOptions {
    SceneFormat format = SceneFormat::F32;
    SceneLayout layout = SceneLayout::Soa;  // f32 default = the Phase 1-4 arrays; others default to split
    int maxShDegree = 3;                    // drop stored SH bands above this (memory; `--scene-sh-degree`)
    uint32_t codebookSize = 4096;           // cb: number of codebook entries (<= 65536)
    uint32_t codebookIterations = 10;       // cb: Lloyd iterations on the training sample
    uint32_t codebookSamples = 0;           // cb: training sample size (0 = min(N, 32 * size))
    uint32_t threads = 0;                   // 0 = all hardware threads
    bool fp16Positions = false;             // debug: round positions through fp16 (shows why we don't)
    float shThreshold = -1.0f;              // Phase 6: per-gaussian SH degree threshold in 1/255 (< 0: off)
};

// Phase 6: what the variable SH degree did (EncodedScene::variableSh).
enum class VariableSh : uint32_t { Off = 0, DegreeOnly = 1, Compact = 2 };

struct CodebookStats {
    uint32_t entries = 0;          // codebook size actually used (<= requested, <= N)
    uint32_t dimensions = 0;       // (K-1)*3
    uint32_t samples = 0;          // training sample size
    uint32_t iterations = 0;       // Lloyd iterations run
    uint32_t emptyReseeded = 0;    // empty clusters re-seeded over all iterations
    double trainMs = 0.0, assignMs = 0.0, totalMs = 0.0;
    double sampleMse = 0.0;        // final mean squared error per coefficient on the training sample
    double fullMse = 0.0;          // same over all gaussians
    bool simd = false;             // AVX2/FMA distance kernel used
};

// A scene encoded for the GPU. Buffers are raw 32-bit words; strides/offsets in 16-byte units.
struct EncodedScene {
    SceneFormat format = SceneFormat::F32;
    SceneLayout layout = SceneLayout::Soa;
    GeometryEncoding geometry = GeometryEncoding::SoaF32;
    ColorEncoding color = ColorEncoding::SoaF32;
    uint32_t count = 0;
    int shDegree = 0;          // stored degree (after maxShDegree)
    uint32_t shCoeffs = 1;     // stored K
    // soa: the four Phase 1 arrays (posAlpha, scales, rotations, sh), as words.
    // split: buffers[0] = geometry records, buffers[1] = colour records (+ buffers[2] = codebook).
    // aos: buffers[0] = records (+ buffers[2] = codebook; buffers[1] empty, the GPU binds [0] twice).
    std::vector<std::vector<uint32_t>> buffers;
    std::vector<std::string> bufferNames;
    uint32_t geometryStride = 0;  // uvec4 per geometry record (0 for soa)
    uint32_t colorStride = 0;     // uvec4 per colour record (0 = CB split: two 8-byte records per uvec4)
    uint32_t colorOffset = 0;     // uvec4 offset of the colour record in its buffer's record (aos)
    uint32_t codebookStride = 0;  // uvec4 per codebook entry
    VariableSh variableSh = VariableSh::Off;  // Phase 6 per-gaussian SH degree
    std::array<uint32_t, 4> degreeHistogram{};  // gaussians per kept degree (variableSh != Off)
    double encodeMs = 0.0;        // whole encode (incl. codebook)
    CodebookStats codebook;

    size_t totalBytes() const;
    double bytesPerGaussian() const { return count ? double(totalBytes()) / double(count) : 0.0; }
};

EncodedScene encodeScene(const GaussianScene& scene, const CompressionOptions& options);

// CPU decode of gaussian i, exactly what the GPU decoder computes (same float operations,
// except that the GPU's normalize() may differ in the last bit). `sh` receives shCoeffs
// RGB triplets.
struct DecodedGaussian {
    glm::vec4 posAlpha{0.0f};
    glm::vec3 sigma{0.0f};
    glm::vec4 rotation{1.0f, 0.0f, 0.0f, 0.0f};  // (w, x, y, z), unit
    std::vector<glm::vec3> sh;
};
void decodeGaussian(const EncodedScene& enc, uint32_t i, DecodedGaussian& out);

// A GaussianScene holding the decoded values (activated arrays only; raw* arrays are left
// empty). Lets the CPU reference renderer render exactly what the GPU sees.
GaussianScene decodeScene(const EncodedScene& enc, const GaussianScene& source);

// Per-format record sizes in bytes for a given stored degree (documentation, tests, UI).
uint32_t geometryRecordBytes(GeometryEncoding g);
uint32_t colorRecordBytes(ColorEncoding c, uint32_t shCoeffs);

// --- Building blocks, exposed for tests -------------------------------------------------

// Phase 6: the SH degree a gaussian keeps for a threshold in 1/255 (see the header comment).
// sh: K RGB triplets (K = (storedDegree+1)^2).
int selectShDegree(const glm::vec3* sh, int storedDegree, float opacity, float threshold255);
// The bound used above: max over directions of |band l's contribution| per channel (Cauchy-
// Schwarz with the addition theorem sum_m Y_lm^2 = (2l+1)/(4 pi)), max over channels.
float shBandMaxContribution(const glm::vec3* sh, int band);

// Q8: quantize one band's coefficients (values for all its (k, c)) with one fp16 scale.
// Returns the scale as half bits; q receives int8 values. |v - q * scale| <= scale / 2.
uint16_t quantizeBandQ8(const float* values, uint32_t n, int8_t* q);

// k-means codebook over `points` (n x dims, row-major). Deterministic for given inputs and
// options, independent of the thread count. `assignment` receives the nearest entry of
// every point (exact nearest under the trained codebook).
struct KMeansOptions {
    uint32_t k = 4096;
    uint32_t iterations = 10;
    uint32_t samples = 0;  // training sample (0 = min(n, 32k)); all n points are assigned at the end
    uint32_t threads = 0;
    uint64_t seed = 0x5EED5EEDull;
    bool allowSimd = true;
};
struct KMeansResult {
    std::vector<float> centroids;     // k x dims
    std::vector<uint32_t> assignment;  // n
    CodebookStats stats;
};
KMeansResult kmeans(const float* points, uint32_t n, uint32_t dims, const KMeansOptions& options);

}  // namespace gsr
