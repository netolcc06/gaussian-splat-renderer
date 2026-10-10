// Reading and writing images: PNG (stb_image / stb_image_write) and PFM (portable float
// map, for exact float comparisons with the CPU reference renderer).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gsr {

// Writes tightly packed RGBA8 pixels (row 0 = top) as an RGB PNG (alpha dropped).
// Creates missing parent directories. Throws on failure.
void writePngRgb(const std::string& path, const uint8_t* rgba, uint32_t width, uint32_t height);

// A float RGB image, row 0 = top, 3 floats per pixel, values in display units (1.0 = white).
struct ImageRgbF {
    uint32_t width = 0, height = 0;
    std::vector<float> rgb;
};

// PNG from floats: each channel is clamped to [0, 1] and rounded to the nearest 8-bit
// level (round(v * 255)), the same mapping the GPU uses when it writes an UNORM target.
void writePngRgb(const std::string& path, const ImageRgbF& image);

// PFM ("PF" colour variant), little endian, bottom row first as the format requires.
void writePfm(const std::string& path, const ImageRgbF& image);

// Reads a .pfm (exact floats) or any 8-bit image stb_image understands (.png, ...: v / 255).
// Throws on failure.
ImageRgbF readImageRgbF(const std::string& path);

}  // namespace gsr
