// Loader for the standard 3DGS .ply format (INRIA graphdeco `save_ply`).
// Format spec and gotchas: docs/notes/ply-format.md.
//
// Accepted: binary_little_endian, one `vertex` element whose properties are found by
// name (any order, any scalar type; unknown properties such as `nxx` or `filter_3D`
// are skipped but still counted in the record stride), SH degree 0..3 from the
// number of f_rest_* properties (0/9/24/45), optional other elements without list
// properties.
// Rejected with a PlyError naming the problem: ASCII and big-endian files, PlayCanvas
// compressed PLY (`chunk` element / `packed_*` properties), missing required
// properties, unexpected f_rest counts or gaps, list properties in `vertex`, and
// files shorter than the header says (truncated).
#pragma once

#include "scene/gaussian_scene.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gsr {

class PlyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class PlyFormat { BinaryLittleEndian, BinaryBigEndian, Ascii };

enum class PlyScalar { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

struct PlyProperty {
    std::string name;
    PlyScalar type = PlyScalar::Float32;
    bool isList = false;
    uint32_t offset = 0;  // byte offset inside the element's record (scalar properties)
};

struct PlyElement {
    std::string name;
    uint64_t count = 0;
    std::vector<PlyProperty> properties;
    uint32_t stride = 0;  // bytes per record (sum of all scalar property sizes)
    bool hasList = false;

    const PlyProperty* find(std::string_view propertyName) const;
};

struct PlyHeader {
    PlyFormat format = PlyFormat::BinaryLittleEndian;
    std::vector<PlyElement> elements;
    std::vector<std::string> comments;
    size_t headerBytes = 0;  // payload starts at this offset (right after "end_header\n")
};

uint32_t plyScalarSize(PlyScalar type);

// Parses the ASCII header (line by line, up to and including "end_header").
// Only checks syntax; the 3DGS-specific checks happen in loadGaussianPly.
PlyHeader parsePlyHeader(std::string_view bytes);
// The same on the first bytes of a stream: nullopt while the header is still incomplete
// (unless `final`: then it throws like parsePlyHeader).
std::optional<PlyHeader> parsePlyHeaderPrefix(std::string_view bytes, bool final);

// SH degree for a number of f_rest_* properties (0/9/24/45 -> 0..3), or -1.
int shDegreeFromRestCount(size_t restCount);

// Loads and activates a scene. Throws PlyError (bad file) or std::runtime_error (I/O).
// The file is memory-mapped and handed to the overload below.
GaussianScene loadGaussianPly(const std::filesystem::path& path);
// The same from bytes already in memory; `name` (a file name) prefixes the error messages and
// becomes sourcePath.
GaussianScene loadGaussianPly(std::span<const uint8_t> bytes, const std::string& name);

// Streaming decoder (the web build: the file arrives in chunks and is never whole in memory).
// push() takes the bytes in order, in chunks of any size; the header is parsed as soon as it
// is complete, then every complete vertex record is decoded with the same code as
// loadGaussianPly and handed out in batches of up to `batchCapacity` gaussians:
//   onHeader(info)          once, after the header checks: info.count, shDegree, comments
//   onBatch(first, batch)   gaussians [first, first + batch.count) in GaussianScene arrays
//                           (activated and raw), valid only during the call
// finish() runs the size checks of loadGaussianPly on the bytes received and fills
// info().warnings and the NaN / zero-quaternion counters of info().stats (the other stats need
// all positions: computeSceneStats on an assembled scene, or computePositionStats). The
// concatenated batches equal loadGaussianPly's arrays bit for bit, and a bad file throws the
// same PlyError text. `totalBytes` (0 = unknown, e.g. a gzip stream) lets a truncated file
// fail at the header instead of at finish().
class PlyStreamDecoder {
public:
    using HeaderCallback = std::function<void(const GaussianScene& info)>;
    using BatchCallback = std::function<void(uint32_t first, const GaussianScene& batch)>;

    PlyStreamDecoder(std::string name, uint64_t totalBytes, HeaderCallback onHeader, BatchCallback onBatch,
                     uint32_t batchCapacity = 16384);
    ~PlyStreamDecoder();
    PlyStreamDecoder(const PlyStreamDecoder&) = delete;
    PlyStreamDecoder& operator=(const PlyStreamDecoder&) = delete;

    void push(std::span<const uint8_t> bytes);  // throws PlyError; the decoder is then unusable
    void finish();                              // throws PlyError (or runtime_error: short stream)

    bool headerParsed() const;
    bool finished() const;
    uint64_t bytesReceived() const;
    uint64_t recordsDecoded() const;
    const GaussianScene& info() const;  // no per-gaussian arrays

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gsr
