// Small RAII-less resource helpers for the WebGPU backend (the web counterpart of
// src/core/resources.*, without an allocator: WebGPU allocates memory itself).
#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>

namespace gsr {

// A 2D texture with one default view.
struct WgpuTexture2D {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    uint32_t width = 0, height = 0;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    uint64_t generation = 0;  // unique per create(): bind groups made for an older texture are stale

    void create(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat fmt, WGPUTextureUsage usage,
                const char* label);
    void release();
    explicit operator bool() const { return texture != nullptr; }
};

// Rows of a texture-to-buffer copy must start at multiples of 256 bytes.
inline uint32_t alignedBytesPerRow(uint32_t bytesPerRow) { return (bytesPerRow + 255u) & ~255u; }

}  // namespace gsr
