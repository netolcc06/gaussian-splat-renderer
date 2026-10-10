#include "scene/ply_loader.h"

#include "scene/mapped_file.h"
#include "scene/parallel.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace gsr {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Real headers are ~1.5 KB; anything beyond this is not a 3DGS file.
constexpr size_t kMaxHeaderBytes = 1 << 20;

std::vector<std::string_view> splitWords(std::string_view line) {
    std::vector<std::string_view> words;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        const size_t start = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
        if (i > start) words.push_back(line.substr(start, i - start));
    }
    return words;
}

bool parseScalarType(std::string_view name, PlyScalar& out) {
    // Both the classic PLY names and the sized aliases (plyfile writes the classic ones).
    struct Entry {
        std::string_view name;
        PlyScalar type;
    };
    static constexpr Entry kTypes[] = {
        {"char", PlyScalar::Int8},     {"int8", PlyScalar::Int8},       {"uchar", PlyScalar::UInt8},
        {"uint8", PlyScalar::UInt8},   {"short", PlyScalar::Int16},     {"int16", PlyScalar::Int16},
        {"ushort", PlyScalar::UInt16}, {"uint16", PlyScalar::UInt16},   {"int", PlyScalar::Int32},
        {"int32", PlyScalar::Int32},   {"uint", PlyScalar::UInt32},     {"uint32", PlyScalar::UInt32},
        {"float", PlyScalar::Float32}, {"float32", PlyScalar::Float32}, {"double", PlyScalar::Float64},
        {"float64", PlyScalar::Float64},
    };
    for (const Entry& e : kTypes) {
        if (e.name == name) {
            out = e.type;
            return true;
        }
    }
    return false;
}

template <typename T>
T loadUnaligned(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));  // records are tightly packed: no alignment guarantee
    return v;
}

// Converts one little-endian scalar of any PLY type to float (the slow, general path).
float readScalar(const uint8_t* p, PlyScalar type) {
    switch (type) {
        case PlyScalar::Int8: return float(loadUnaligned<int8_t>(p));
        case PlyScalar::UInt8: return float(loadUnaligned<uint8_t>(p));
        case PlyScalar::Int16: return float(loadUnaligned<int16_t>(p));
        case PlyScalar::UInt16: return float(loadUnaligned<uint16_t>(p));
        case PlyScalar::Int32: return float(loadUnaligned<int32_t>(p));
        case PlyScalar::UInt32: return float(loadUnaligned<uint32_t>(p));
        case PlyScalar::Float32: return loadUnaligned<float>(p);
        case PlyScalar::Float64: return float(loadUnaligned<double>(p));
    }
    return 0.0f;
}

// NaN or +-Inf: exponent bits all ones. A bit test is cheaper than std::isfinite in
// the hot loop (MSVC does not always inline it).
bool isNonFinite(float v) { return (std::bit_cast<uint32_t>(v) & 0x7f800000u) == 0x7f800000u; }

// Where each attribute the renderer needs lives inside a vertex record.
struct Field {
    uint32_t offset = 0;
    PlyScalar type = PlyScalar::Float32;
};

struct VertexLayout {
    Field pos[3], dc[3], opacity, scale[3], rot[4];
    std::vector<Field> rest;  // f_rest_0 .. f_rest_{n-1}, file (channel-major) order
    uint32_t stride = 0;
    bool allFloat = true;  // fast path: every needed field is float32
};

// Decodes gaussians [begin, end): reads the record fields, transposes SH, applies the
// activations. Templated on the "all float32" fast path so the common case compiles
// to plain unaligned float loads.
template <bool kAllFloat>
void decodeRange(const VertexLayout& layout, const uint8_t* base, size_t begin, size_t end, int shDegree,
                 GaussianScene& scene, uint64_t& nonFinite, uint32_t& zeroQuats) {
    auto read = [&](const uint8_t* rec, const Field& f) -> float {
        float v;
        if constexpr (kAllFloat) {
            v = loadUnaligned<float>(rec + f.offset);
        } else {
            v = readScalar(rec + f.offset, f.type);
        }
        nonFinite += isNonFinite(v) ? 1 : 0;
        return v;
    };

    const uint32_t K = uint32_t((shDegree + 1) * (shDegree + 1));  // coefficients per channel
    for (size_t i = begin; i < end; ++i) {
        const uint8_t* rec = base + i * layout.stride;

        const glm::vec3 pos(read(rec, layout.pos[0]), read(rec, layout.pos[1]), read(rec, layout.pos[2]));
        const float opacity = read(rec, layout.opacity);
        const glm::vec3 logScale(read(rec, layout.scale[0]), read(rec, layout.scale[1]), read(rec, layout.scale[2]));
        const glm::vec4 q(read(rec, layout.rot[0]), read(rec, layout.rot[1]), read(rec, layout.rot[2]),
                          read(rec, layout.rot[3]));

        scene.rawOpacity[i] = opacity;
        scene.rawLogScale[i] = logScale;
        scene.rawRotation[i] = q;

        // Activations (gaussian_model.py setup_functions).
        const float alpha = 1.0f / (1.0f + std::exp(-opacity));
        scene.posAlpha[i] = glm::vec4(pos, alpha);
        scene.scales[i] = glm::vec4(glm::exp(logScale), 0.0f);
        const float n2 = glm::dot(q, q);
        if (n2 > 1e-24f && !isNonFinite(n2)) {
            scene.rotations[i] = q / std::sqrt(n2);
        } else {
            scene.rotations[i] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);  // identity (w first)
            ++zeroQuats;
        }

        // SH: file has f_dc_c (k = 0) and channel-major f_rest[c*(K-1) + (k-1)];
        // we write per-coefficient RGB triplets sh[(i*K + k)*3 + c].
        float* out = scene.sh.data() + size_t(i) * K * 3;
        for (int c = 0; c < 3; ++c) out[c] = read(rec, layout.dc[c]);
        for (uint32_t k = 1; k < K; ++k)
            for (uint32_t c = 0; c < 3; ++c) out[k * 3 + c] = read(rec, layout.rest[c * (K - 1) + (k - 1)]);
    }
}

// Splits [0, count) into contiguous chunks, one per hardware thread.
template <typename Fn>
void parallelFor(size_t count, Fn&& fn) {
    const size_t minPerThread = 16384;  // below this, thread start-up costs more than it saves
    const size_t hw = hardwareThreads();  // 1 in the web build (no threads there)
    const size_t threads = std::clamp<size_t>(count / minPerThread, 1, hw);
    if (threads == 1) {
        fn(size_t(0), size_t(0), count);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        const size_t begin = count * t / threads;
        const size_t end = count * (t + 1) / threads;
        pool.emplace_back([&fn, t, begin, end] { fn(t, begin, end); });
    }
    for (std::thread& th : pool) th.join();
}

std::string bytesText(uint64_t bytes) { return std::to_string(bytes) + " bytes"; }

}  // namespace

const PlyProperty* PlyElement::find(std::string_view propertyName) const {
    for (const PlyProperty& p : properties)
        if (p.name == propertyName) return &p;
    return nullptr;
}

uint32_t plyScalarSize(PlyScalar type) {
    switch (type) {
        case PlyScalar::Int8:
        case PlyScalar::UInt8: return 1;
        case PlyScalar::Int16:
        case PlyScalar::UInt16: return 2;
        case PlyScalar::Int32:
        case PlyScalar::UInt32:
        case PlyScalar::Float32: return 4;
        case PlyScalar::Float64: return 8;
    }
    return 0;
}

int shDegreeFromRestCount(size_t restCount) {
    // restCount = 3 * ((deg + 1)^2 - 1)
    switch (restCount) {
        case 0: return 0;
        case 9: return 1;
        case 24: return 2;
        case 45: return 3;
        default: return -1;
    }
}

PlyHeader parsePlyHeader(std::string_view bytes) { return *parsePlyHeaderPrefix(bytes, true); }

std::optional<PlyHeader> parsePlyHeaderPrefix(std::string_view bytes, bool final) {
    PlyHeader header;
    size_t pos = 0;
    size_t lineNumber = 0;
    bool sawFormat = false;
    const size_t limit = std::min(bytes.size(), kMaxHeaderBytes);

    while (true) {
        const size_t eol = bytes.substr(0, limit).find('\n', pos);
        if (eol == std::string_view::npos && !final && bytes.size() < kMaxHeaderBytes)
            return std::nullopt;  // streaming: the rest of the header has not arrived yet
        if (eol == std::string_view::npos)
            throw PlyError(lineNumber == 0 ? "not a PLY file (empty or no header)"
                                           : "PLY header has no 'end_header' line (file truncated or not a PLY)");
        std::string_view line = bytes.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);  // tolerate CRLF headers
        pos = eol + 1;
        ++lineNumber;
        const std::string where = "PLY header line " + std::to_string(lineNumber) + ": ";

        if (lineNumber == 1) {
            if (line != "ply") throw PlyError("not a PLY file (first line is not 'ply')");
            continue;
        }
        const std::vector<std::string_view> w = splitWords(line);
        if (w.empty()) continue;

        if (w[0] == "end_header") {
            if (!sawFormat) throw PlyError("PLY header has no 'format' line");
            header.headerBytes = pos;
            break;
        } else if (w[0] == "comment" || w[0] == "obj_info") {
            header.comments.emplace_back(line.size() > w[0].size() + 1 ? line.substr(w[0].size() + 1) : "");
        } else if (w[0] == "format") {
            if (w.size() < 2) throw PlyError(where + "malformed 'format' line");
            if (w[1] == "binary_little_endian") header.format = PlyFormat::BinaryLittleEndian;
            else if (w[1] == "binary_big_endian") header.format = PlyFormat::BinaryBigEndian;
            else if (w[1] == "ascii") header.format = PlyFormat::Ascii;
            else throw PlyError(where + "unknown format '" + std::string(w[1]) + "'");
            sawFormat = true;
        } else if (w[0] == "element") {
            if (w.size() != 3) throw PlyError(where + "malformed 'element' line");
            PlyElement element;
            element.name = std::string(w[1]);
            const auto [ptr, ec] = std::from_chars(w[2].data(), w[2].data() + w[2].size(), element.count);
            if (ec != std::errc() || ptr != w[2].data() + w[2].size())
                throw PlyError(where + "bad element count '" + std::string(w[2]) + "'");
            header.elements.push_back(std::move(element));
        } else if (w[0] == "property") {
            if (header.elements.empty()) throw PlyError(where + "'property' before any 'element'");
            PlyElement& element = header.elements.back();
            PlyProperty prop;
            if (w.size() == 5 && w[1] == "list") {
                // property list <count type> <item type> <name>: variable-size records.
                PlyScalar countType, itemType;
                if (!parseScalarType(w[2], countType) || !parseScalarType(w[3], itemType))
                    throw PlyError(where + "unknown list types");
                prop.isList = true;
                prop.name = std::string(w[4]);
                element.hasList = true;
            } else if (w.size() == 3) {
                if (!parseScalarType(w[1], prop.type))
                    throw PlyError(where + "unknown property type '" + std::string(w[1]) + "'");
                prop.name = std::string(w[2]);
                prop.offset = element.stride;  // every scalar counts, including unknown names
                element.stride += plyScalarSize(prop.type);
            } else {
                throw PlyError(where + "malformed 'property' line");
            }
            element.properties.push_back(std::move(prop));
        } else {
            throw PlyError(where + "unknown keyword '" + std::string(w[0]) + "'");
        }
    }
    return header;
}

namespace {

// What the header says about the vertex payload, after the 3DGS-specific checks.
struct VertexPlan {
    VertexLayout layout;
    int shDegree = 0;
    uint64_t count = 0;
    uint64_t vertexOffset = 0;   // file offset of the first vertex record
    uint64_t trailingBytes = 0;  // size of list-free elements after vertex (for the size check)
    bool trailingUnknown = false;
    std::vector<std::string> warnings;  // skipped elements
};

// Validates a parsed header for 3DGS use and locates the vertex records. `fileSize` bounds the
// elements stored before `vertex` (UINT64_MAX while a stream's size is unknown).
VertexPlan planVertexDecode(const PlyHeader& header, const std::string& file, uint64_t fileSize) {
    if (header.format == PlyFormat::Ascii)
        throw PlyError(file + ": ASCII PLY is not supported (3DGS files are binary_little_endian; "
                              "re-export as binary)");
    if (header.format == PlyFormat::BinaryBigEndian)
        throw PlyError(file + ": binary_big_endian PLY is not supported (expected binary_little_endian)");

    // PlayCanvas/SuperSplat "compressed PLY" is a PLY too, but with quantized fields.
    for (const PlyElement& e : header.elements) {
        bool packed = e.name == "chunk";
        for (const PlyProperty& p : e.properties) packed = packed || p.name.rfind("packed_", 0) == 0;
        if (packed)
            throw PlyError(file + ": this is a PlayCanvas compressed PLY (element 'chunk' / packed_* properties), "
                                  "not a standard 3DGS PLY. Decompress it first (e.g. @playcanvas/splat-transform).");
    }

    // Locate the vertex element; skip any list-free elements stored before it.
    VertexPlan plan;
    const PlyElement* vertex = nullptr;
    plan.vertexOffset = header.headerBytes;
    for (const PlyElement& e : header.elements) {
        if (e.name == "vertex") {
            vertex = &e;
            continue;
        }
        if (!vertex) {
            if (e.hasList)
                throw PlyError(file + ": element '" + e.name + "' with list properties precedes 'vertex'");
            // Overflow-safe: an absurd count must not wrap the offset around.
            if (e.stride != 0 && e.count > (fileSize - std::min<uint64_t>(plan.vertexOffset, fileSize)) / e.stride)
                throw PlyError(file + ": file is truncated inside element '" + e.name + "'");
            plan.vertexOffset += e.count * e.stride;
        } else if (e.hasList) {
            plan.trailingUnknown = true;
        } else {
            plan.trailingBytes += e.count * e.stride;
        }
        plan.warnings.push_back("skipped element '" + e.name + "' (" + std::to_string(e.count) + " records)");
    }
    if (!vertex) throw PlyError(file + ": no 'vertex' element");
    if (vertex->hasList) throw PlyError(file + ": 'vertex' element has list properties (not a 3DGS file)");
    if (vertex->count == 0) throw PlyError(file + ": 'vertex' element is empty (0 gaussians)");
    if (vertex->count > UINT32_MAX) throw PlyError(file + ": too many gaussians");
    plan.count = vertex->count;

    VertexLayout& layout = plan.layout;
    layout.stride = vertex->stride;
    auto require = [&](const char* name) -> Field {
        const PlyProperty* p = vertex->find(name);
        if (!p) throw PlyError(file + ": missing required property '" + std::string(name) + "'");
        layout.allFloat = layout.allFloat && p->type == PlyScalar::Float32;
        return {p->offset, p->type};
    };
    layout.pos[0] = require("x");
    layout.pos[1] = require("y");
    layout.pos[2] = require("z");
    layout.dc[0] = require("f_dc_0");
    layout.dc[1] = require("f_dc_1");
    layout.dc[2] = require("f_dc_2");
    layout.opacity = require("opacity");
    layout.scale[0] = require("scale_0");
    layout.scale[1] = require("scale_1");
    layout.scale[2] = require("scale_2");
    layout.rot[0] = require("rot_0");
    layout.rot[1] = require("rot_1");
    layout.rot[2] = require("rot_2");
    layout.rot[3] = require("rot_3");

    // f_rest_<i>: count decides the SH degree; indices must be exactly 0..n-1.
    std::vector<const PlyProperty*> rest;
    for (const PlyProperty& p : vertex->properties) {
        if (p.name.rfind("f_rest_", 0) != 0) continue;
        const std::string_view suffix = std::string_view(p.name).substr(7);
        size_t index = 0;
        const auto [ptr, ec] = std::from_chars(suffix.data(), suffix.data() + suffix.size(), index);
        if (ec != std::errc() || ptr != suffix.data() + suffix.size())
            throw PlyError(file + ": bad SH property name '" + p.name + "'");
        if (index >= rest.size()) rest.resize(index + 1, nullptr);
        if (rest[index]) throw PlyError(file + ": duplicate property '" + p.name + "'");
        rest[index] = &p;
    }
    const size_t restCount = size_t(std::count_if(rest.begin(), rest.end(), [](auto* p) { return p != nullptr; }));
    if (restCount != rest.size())
        throw PlyError(file + ": f_rest_* indices are not contiguous (highest index " +
                       std::to_string(rest.size() - 1) + ", but only " + std::to_string(restCount) + " present)");
    plan.shDegree = shDegreeFromRestCount(restCount);
    if (plan.shDegree < 0)
        throw PlyError(file + ": unexpected number of f_rest_* properties: " + std::to_string(restCount) +
                       " (expected 0, 9, 24 or 45 for SH degree 0..3)");
    for (const PlyProperty* p : rest) {
        layout.rest.push_back({p->offset, p->type});
        layout.allFloat = layout.allFloat && p->type == PlyScalar::Float32;
    }
    return plan;
}

// Size check: the payload must hold every record the header promises. Appends the warnings
// about bytes after the vertex records.
void checkPayloadSize(const VertexPlan& plan, uint64_t fileSize, const std::string& file,
                      std::vector<std::string>& warnings) {
    const uint32_t stride = plan.layout.stride;
    const uint64_t available = fileSize > plan.vertexOffset ? fileSize - plan.vertexOffset : 0;
    const uint64_t vertexBytes = plan.count * stride;
    if (available < vertexBytes) {
        throw PlyError(file + ": file is truncated: header declares " + std::to_string(plan.count) +
                       " gaussians x " + std::to_string(stride) + " bytes = " + bytesText(vertexBytes) +
                       " of vertex data, but only " + bytesText(available) + " follow the header (" +
                       std::to_string(available / stride) + " complete records)");
    }
    if (!plan.trailingUnknown && available > vertexBytes + plan.trailingBytes)
        warnings.push_back(std::to_string(available - vertexBytes - plan.trailingBytes) +
                           " unexpected trailing bytes after the last element");
    if (!plan.trailingUnknown && available < vertexBytes + plan.trailingBytes)
        warnings.push_back("elements after 'vertex' are truncated (ignored)");
}

// Warnings about the decoded values (scene.stats counters already set).
void addValueWarnings(GaussianScene& scene, bool allFloat) {
    if (scene.stats.nonFiniteValues)
        scene.warnings.push_back(std::to_string(scene.stats.nonFiniteValues) + " NaN/Inf values in the file");
    if (scene.stats.zeroQuaternions)
        scene.warnings.push_back(std::to_string(scene.stats.zeroQuaternions) +
                                 " zero-length quaternions replaced by identity");
    if (!allFloat) scene.warnings.push_back("non-float32 properties converted to float");
}

// Sizes every per-gaussian array of `scene` for n gaussians of the scene's SH degree.
void resizeScene(GaussianScene& scene, size_t n) {
    scene.count = uint32_t(n);
    scene.posAlpha.resize(n);
    scene.scales.resize(n);
    scene.rotations.resize(n);
    scene.sh.resize(n * scene.shCoeffsPerChannel() * 3);
    scene.rawOpacity.resize(n);
    scene.rawLogScale.resize(n);
    scene.rawRotation.resize(n);
}

}  // namespace

GaussianScene loadGaussianPly(const std::filesystem::path& path) {
    const auto tStart = Clock::now();
    const std::string file = std::string(reinterpret_cast<const char*>(path.filename().u8string().c_str()));
    MappedFile mapped(path);
    const double mapMs = msSince(tStart);

    GaussianScene scene = loadGaussianPly(mapped.bytes(), file);
    scene.sourcePath = std::string(reinterpret_cast<const char*>(path.u8string().c_str()));  // UTF-8
    scene.timings.mapMs = mapMs;
    scene.timings.totalMs = msSince(tStart);
    return scene;
}

GaussianScene loadGaussianPly(std::span<const uint8_t> bytes, const std::string& file) {
    const auto tStart = Clock::now();
    GaussianScene scene;
    scene.sourcePath = file;

    // --- Header + validation -----------------------------------------------------
    const auto tHeader = Clock::now();
    PlyHeader header;
    try {
        header = parsePlyHeader(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    } catch (const PlyError& e) {
        throw PlyError(file + ": " + e.what());
    }
    scene.comments = header.comments;
    const VertexPlan plan = planVertexDecode(header, file, bytes.size());
    scene.warnings = plan.warnings;
    checkPayloadSize(plan, bytes.size(), file, scene.warnings);
    scene.shDegree = plan.shDegree;
    scene.timings.headerMs = msSince(tHeader);

    // --- Decode (multi-threaded, one contiguous chunk per thread) -------------------
    const auto tDecode = Clock::now();
    const size_t n = size_t(plan.count);
    resizeScene(scene, n);

    const VertexLayout& layout = plan.layout;
    const uint8_t* base = bytes.data() + plan.vertexOffset;
    const size_t hw = hardwareThreads();
    std::vector<uint64_t> nonFinite(hw, 0);
    std::vector<uint32_t> zeroQuats(hw, 0);
    parallelFor(n, [&](size_t t, size_t begin, size_t end) {
        if (layout.allFloat)
            decodeRange<true>(layout, base, begin, end, scene.shDegree, scene, nonFinite[t], zeroQuats[t]);
        else
            decodeRange<false>(layout, base, begin, end, scene.shDegree, scene, nonFinite[t], zeroQuats[t]);
    });
    scene.timings.decodeMs = msSince(tDecode);

    // --- Stats -------------------------------------------------------------------------
    const auto tStats = Clock::now();
    computeSceneStats(scene);
    for (size_t t = 0; t < hw; ++t) {
        scene.stats.nonFiniteValues += nonFinite[t];
        scene.stats.zeroQuaternions += zeroQuats[t];
    }
    addValueWarnings(scene, layout.allFloat);
    scene.timings.statsMs = msSince(tStats);
    scene.timings.totalMs = msSince(tStart);
    return scene;
}

// --- Streaming decoder ----------------------------------------------------------------

struct PlyStreamDecoder::Impl {
    std::string file;
    uint64_t totalBytes = 0;  // 0 = unknown
    uint32_t batchCapacity = 0;
    HeaderCallback onHeader;
    BatchCallback onBatch;

    uint64_t received = 0;   // bytes pushed so far (= file offset of the next byte)
    std::string headerText;  // the first bytes, kept until the header parses
    bool headerDone = false;
    bool finished = false;
    bool failed = false;
    PlyHeader header;
    VertexPlan plan;
    uint64_t vertexEnd = 0;  // file offset after the last vertex record (saturated)

    std::vector<uint8_t> pending;  // whole records copied from the chunks, waiting for the decode
    size_t pendingBytes = 0;
    uint64_t decodedRecords = 0;
    GaussianScene batch;  // decoded records of one batch (reused)
    GaussianScene info;   // count, degree, comments, warnings, counters, timings
    uint64_t nonFinite = 0;
    uint32_t zeroQuats = 0;
    Clock::time_point start = Clock::now();
    double decodeMs = 0.0;

    void startPayload();
    void consumePayload(const uint8_t* data, size_t size, uint64_t fileOffset);
    void flush();
};

PlyStreamDecoder::PlyStreamDecoder(std::string name, uint64_t totalBytes, HeaderCallback onHeader,
                                   BatchCallback onBatch, uint32_t batchCapacity)
    : impl_(std::make_unique<Impl>()) {
    impl_->file = std::move(name);
    impl_->totalBytes = totalBytes;
    impl_->batchCapacity = std::max(1u, batchCapacity);
    impl_->onHeader = std::move(onHeader);
    impl_->onBatch = std::move(onBatch);
    impl_->info.sourcePath = impl_->file;
}

PlyStreamDecoder::~PlyStreamDecoder() = default;

bool PlyStreamDecoder::headerParsed() const { return impl_->headerDone; }
bool PlyStreamDecoder::finished() const { return impl_->finished; }
uint64_t PlyStreamDecoder::bytesReceived() const { return impl_->received; }
uint64_t PlyStreamDecoder::recordsDecoded() const { return impl_->decodedRecords; }
const GaussianScene& PlyStreamDecoder::info() const { return impl_->info; }

void PlyStreamDecoder::Impl::startPayload() {
    headerDone = true;
    info.comments = header.comments;
    // With the total size known, the checks of loadGaussianPly run now (a truncated file fails
    // before any record is decoded); otherwise they run in finish().
    plan = planVertexDecode(header, file, totalBytes ? totalBytes : UINT64_MAX);
    if (totalBytes) {
        std::vector<std::string> warningsLater;  // reported by finish()
        checkPayloadSize(plan, totalBytes, file, warningsLater);
    }
    const uint64_t vertexBytes = plan.count * plan.layout.stride;
    vertexEnd = plan.vertexOffset > UINT64_MAX - vertexBytes ? UINT64_MAX : plan.vertexOffset + vertexBytes;
    info.count = uint32_t(plan.count);
    info.shDegree = plan.shDegree;
    info.timings.headerMs = msSince(start);

    const size_t capacity = size_t(std::min<uint64_t>(plan.count, batchCapacity));
    pending.resize(capacity * plan.layout.stride);
    batch.shDegree = plan.shDegree;
    batch.comments = header.comments;
    if (onHeader) onHeader(info);
}

void PlyStreamDecoder::Impl::consumePayload(const uint8_t* data, size_t size, uint64_t fileOffset) {
    // Only the bytes of the vertex records matter; elements before and after are skipped.
    const uint64_t begin = std::max(fileOffset, plan.vertexOffset);
    const uint64_t end = std::min(fileOffset + size, vertexEnd);
    for (uint64_t at = begin; at < end;) {
        const size_t take = size_t(std::min<uint64_t>(end - at, pending.size() - pendingBytes));
        std::memcpy(pending.data() + pendingBytes, data + (at - fileOffset), take);
        pendingBytes += take;
        at += take;
        if (pendingBytes == pending.size()) flush();
    }
    if (fileOffset + size >= vertexEnd && pendingBytes > 0) flush();  // the last, partial batch
}

void PlyStreamDecoder::Impl::flush() {
    const size_t m = pendingBytes / plan.layout.stride;
    const auto t0 = Clock::now();
    resizeScene(batch, m);
    if (plan.layout.allFloat)
        decodeRange<true>(plan.layout, pending.data(), 0, m, plan.shDegree, batch, nonFinite, zeroQuats);
    else
        decodeRange<false>(plan.layout, pending.data(), 0, m, plan.shDegree, batch, nonFinite, zeroQuats);
    decodeMs += msSince(t0);
    if (onBatch) onBatch(uint32_t(decodedRecords), batch);
    decodedRecords += m;
    pendingBytes = 0;
}

void PlyStreamDecoder::push(std::span<const uint8_t> bytes) {
    Impl& s = *impl_;
    if (s.failed || s.finished) throw std::logic_error("PlyStreamDecoder::push after an error or finish()");
    try {
        size_t used = 0;
        if (!s.headerDone) {
            // Collect the header (at most kMaxHeaderBytes) and try to parse it whenever a line ends.
            used = std::min(bytes.size(), kMaxHeaderBytes - std::min(kMaxHeaderBytes, s.headerText.size()));
            const size_t before = s.headerText.size();
            s.headerText.append(reinterpret_cast<const char*>(bytes.data()), used);
            s.received += used;
            const bool full = s.headerText.size() >= kMaxHeaderBytes;
            if (full || s.headerText.find('\n', before) != std::string::npos) {
                std::optional<PlyHeader> header;
                try {
                    header = parsePlyHeaderPrefix(s.headerText, full);
                } catch (const PlyError& e) {
                    throw PlyError(s.file + ": " + e.what());
                }
                if (header) {
                    s.header = std::move(*header);
                    s.startPayload();
                    // Bytes collected past "end_header\n" are payload.
                    const size_t h = s.header.headerBytes;
                    s.consumePayload(reinterpret_cast<const uint8_t*>(s.headerText.data()) + h,
                                     s.headerText.size() - h, h);
                    s.headerText = std::string();
                }
            }
        }
        if (s.headerDone && used < bytes.size()) {
            s.consumePayload(bytes.data() + used, bytes.size() - used, s.received);
            s.received += bytes.size() - used;
        }
    } catch (...) {
        s.failed = true;
        throw;
    }
}

void PlyStreamDecoder::finish() {
    Impl& s = *impl_;
    if (s.failed || s.finished) throw std::logic_error("PlyStreamDecoder::finish after an error or finish()");
    try {
        if (!s.headerDone) {
            try {
                parsePlyHeaderPrefix(s.headerText, true);  // throws: no complete header arrived
            } catch (const PlyError& e) {
                throw PlyError(s.file + ": " + e.what());
            }
            throw PlyError(s.file + ": incomplete PLY header");  // not reached: a complete header parses in push()
        }
        if (s.totalBytes && s.received != s.totalBytes)
            throw std::runtime_error(s.file + ": stream ended after " + std::to_string(s.received) + " of " +
                                     std::to_string(s.totalBytes) + " bytes");
        // The checks of loadGaussianPly with the real size: the same errors and warnings.
        s.plan = planVertexDecode(s.header, s.file, s.received);
        s.info.warnings = s.plan.warnings;
        checkPayloadSize(s.plan, s.received, s.file, s.info.warnings);
        s.info.stats.nonFiniteValues = s.nonFinite;
        s.info.stats.zeroQuaternions = s.zeroQuats;
        addValueWarnings(s.info, s.plan.layout.allFloat);
        s.info.timings.decodeMs = s.decodeMs;
        s.info.timings.totalMs = msSince(s.start);
        s.finished = true;
    } catch (...) {
        s.failed = true;
        throw;
    }
}

}  // namespace gsr
