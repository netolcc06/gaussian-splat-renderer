#include "app/image_io.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <stb_image.h>
#include <stb_image_write.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gsr {

namespace {

void createParentDirectories(const std::string& path) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

bool hasExtension(const std::string& path, const char* ext) {
    std::string e = std::filesystem::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return e == ext;
}

}  // namespace

void writePngRgb(const std::string& path, const uint8_t* rgba, uint32_t width, uint32_t height) {
    createParentDirectories(path);
    std::vector<uint8_t> rgb(size_t(width) * height * 3);
    for (size_t i = 0; i < size_t(width) * height; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    if (!stbi_write_png(path.c_str(), int(width), int(height), 3, rgb.data(), int(width * 3)))
        throw std::runtime_error("Failed to write PNG: " + path);
}

void writePngRgb(const std::string& path, const ImageRgbF& image) {
    createParentDirectories(path);
    std::vector<uint8_t> rgb(image.rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i) {
        const float v = std::clamp(image.rgb[i], 0.0f, 1.0f);  // NaN -> 0 via the comparison below
        rgb[i] = uint8_t(std::lround(std::isnan(v) ? 0.0f : v * 255.0f));
    }
    if (!stbi_write_png(path.c_str(), int(image.width), int(image.height), 3, rgb.data(), int(image.width * 3)))
        throw std::runtime_error("Failed to write PNG: " + path);
}

void writePfm(const std::string& path, const ImageRgbF& image) {
    createParentDirectories(path);
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write " + path);
    // Header: "PF" (3 channels), size, scale. A negative scale means little endian.
    file << "PF\n" << image.width << " " << image.height << "\n-1.0\n";
    const size_t rowFloats = size_t(image.width) * 3;
    for (uint32_t y = image.height; y-- > 0;)  // PFM stores the bottom row first
        file.write(reinterpret_cast<const char*>(image.rgb.data() + y * rowFloats),
                   std::streamsize(rowFloats * sizeof(float)));
    if (!file) throw std::runtime_error("Failed to write " + path);
}

ImageRgbF readImageRgbF(const std::string& path) {
    ImageRgbF image;
    if (hasExtension(path, ".pfm")) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open " + path);
        std::string magic;
        double scale = 0.0;
        file >> magic >> image.width >> image.height >> scale;
        file.get();  // the single whitespace character after the scale
        if (magic != "PF" || !file || image.width == 0 || image.height == 0)
            throw std::runtime_error("Not a colour PFM file: " + path);
        if (scale > 0.0) throw std::runtime_error("Big-endian PFM is not supported: " + path);
        const size_t rowFloats = size_t(image.width) * 3;
        image.rgb.resize(rowFloats * image.height);
        for (uint32_t y = image.height; y-- > 0;)
            file.read(reinterpret_cast<char*>(image.rgb.data() + y * rowFloats),
                      std::streamsize(rowFloats * sizeof(float)));
        if (!file) throw std::runtime_error("PFM file is truncated: " + path);
        return image;
    }

    int w = 0, h = 0, channels = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) throw std::runtime_error("Cannot read image " + path + ": " + stbi_failure_reason());
    image.width = uint32_t(w);
    image.height = uint32_t(h);
    image.rgb.resize(size_t(w) * h * 3);
    for (size_t i = 0; i < image.rgb.size(); ++i) image.rgb[i] = float(data[i]) / 255.0f;
    stbi_image_free(data);
    return image;
}

}  // namespace gsr
