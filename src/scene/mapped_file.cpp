#include "scene/mapped_file.h"

#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fstream>
#endif

namespace gsr {

#ifdef _WIN32

MappedFile::MappedFile(const std::filesystem::path& path) {
    const std::string name = std::string(reinterpret_cast<const char*>(path.u8string().c_str()));  // UTF-8 (string() throws for non-ANSI names)
    m_file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (m_file == INVALID_HANDLE_VALUE) {
        m_file = nullptr;
        throw std::runtime_error("Cannot open file: " + name);
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(m_file, &size)) {
        CloseHandle(m_file);
        throw std::runtime_error("Cannot query file size: " + name);
    }
    m_size = size_t(size.QuadPart);
    if (m_size == 0) return;  // CreateFileMapping fails on empty files; an empty span is fine

    m_mapping = CreateFileMappingW(m_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m_mapping) {
        CloseHandle(m_file);
        throw std::runtime_error("Cannot map file: " + name);
    }
    m_data = static_cast<const uint8_t*>(MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0));
    if (!m_data) {
        CloseHandle(m_mapping);
        CloseHandle(m_file);
        throw std::runtime_error("Cannot map view of file: " + name);
    }
}

MappedFile::~MappedFile() {
    if (m_data) UnmapViewOfFile(m_data);
    if (m_mapping) CloseHandle(m_mapping);
    if (m_file) CloseHandle(m_file);
}

#else

MappedFile::MappedFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path.string());
    m_size = size_t(file.tellg());
    m_buffer.resize(m_size);
    file.seekg(0);
    if (m_size && !file.read(reinterpret_cast<char*>(m_buffer.data()), std::streamsize(m_size)))
        throw std::runtime_error("Cannot read file: " + path.string());
    m_data = m_buffer.data();
}

MappedFile::~MappedFile() = default;

#endif

}  // namespace gsr
