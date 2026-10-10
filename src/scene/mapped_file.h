// MappedFile: read-only view of a whole file's bytes.
//
// On Windows the file is memory-mapped (CreateFileMapping/MapViewOfFile): no copy into
// a user buffer, and the OS pages the data in as the parser touches it. Elsewhere it
// falls back to one bulk read into a heap buffer. Either way the loader sees one
// contiguous `std::span<const uint8_t>` and never issues per-record reads.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace gsr {

class MappedFile {
public:
    // Throws std::runtime_error if the file cannot be opened or mapped.
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    std::span<const uint8_t> bytes() const { return {m_data, m_size}; }
    size_t size() const { return m_size; }

private:
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;
#ifdef _WIN32
    void* m_file = nullptr;     // HANDLE
    void* m_mapping = nullptr;  // HANDLE
#else
    std::vector<uint8_t> m_buffer;
#endif
};

}  // namespace gsr
