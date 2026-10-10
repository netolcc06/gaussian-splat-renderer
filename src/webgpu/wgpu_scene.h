// WgpuScene: the gaussians in WebGPU storage buffers, in one of the scene formats the web offers
// (f32 soa, f16 split, q8 split; scene/scene_compression.h). The web counterpart of
// src/render/gpu_scene.*, with two differences:
//   - The scene is uploaded in batches while the file streams in (PlyStreamDecoder): create()
//     sizes the buffers from the header, then upload(first, batch) encodes each batch with the
//     native encoder (encodeScene, per gaussian, so a batch encodes to exactly its slice of the
//     whole-scene encoding) and writes it at its offset with queue.writeBuffer. No CPU copy of
//     the scene is kept.
//   - The format falls back (f32 -> f16 -> q8) when a buffer would exceed the device's
//     maxStorageBufferBindingSize / maxBufferSize (DESIGN.md 2.2); fallbackNote says so.
//
// Every shader that reads gaussians includes shaders/wgsl/scene_decode.wgsl and binds the
// scene's bind group at group 0 (bindings 0-3, the same table as native):
//
//   binding  f32 soa                     split (f16, q8; f32 split for tests)
//   0        posAlpha  vec4[N]           geometry records
//   1        scales    vec4[N]           colour records
//   2        rotations vec4[N]           unused (buffer 0 bound)
//   3        sh        float[N*K*3]      unused (buffer 0 bound)
//
// The decoder's format is fixed per pipeline by override constants (decodeConstants()).
#pragma once

#include "scene/scene_compression.h"

#include <webgpu/webgpu.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gsr {

struct GaussianScene;

struct WgpuSceneOptions {
    SceneFormat format = SceneFormat::F32;  // requested; may fall back (see fallbackNote)
    int maxShDegree = 3;                    // drop stored SH bands above this
    uint64_t debugMaxBufferBytes = 0;       // tests: a lower buffer limit than the device's (0 = off)
};

class WgpuScene {
public:
    WgpuScene() = default;
    WgpuScene(const WgpuScene&) = delete;
    WgpuScene& operator=(const WgpuScene&) = delete;
    ~WgpuScene() { release(); }

    // The bind group layout of group 0 (4 read-only storage buffers, all stages). Shared by every
    // pipeline that decodes gaussians; owned by the caller.
    static WGPUBindGroupLayout createBindGroupLayout(WGPUDevice device);

    // Chooses the format (with fallback) and creates empty buffers for `count` gaussians of SH
    // degree `shDegree` (from the PLY header). Returns false with `error` set if no format fits
    // the device limits.
    bool create(WGPUDevice device, WGPUBindGroupLayout layout, const WGPULimits& limits, uint32_t count,
                int shDegree, const WgpuSceneOptions& options, std::string* error);
    // Encodes gaussians [first, first + batch.count) and queues their upload.
    void upload(WGPUQueue queue, uint32_t first, const GaussianScene& batch);
    // Tests: a whole scene encoded on the CPU (any encoding incl. variable SH, except cb).
    bool createFromEncoded(WGPUDevice device, WGPUQueue queue, WGPUBindGroupLayout layout,
                           const EncodedScene& encoded, std::string* error);
    void release();

    bool valid() const { return bindGroup_ != nullptr; }
    WGPUBindGroup bindGroup() const { return bindGroup_; }
    // Override constants for scene_decode.wgsl (keys are the WGSL names).
    std::vector<WGPUConstantEntry> decodeConstants() const;
    // A string identifying the decoder variant (pipelines are cached per key).
    std::string decoderKey() const;

    uint32_t count = 0;
    int shDegree = 0;            // STORED degree
    uint32_t shCoeffs = 1;       // stored K
    SceneFormat requestedFormat = SceneFormat::F32;
    SceneFormat format = SceneFormat::F32;
    SceneLayout sceneLayout = SceneLayout::Soa;
    GeometryEncoding geometry = GeometryEncoding::SoaF32;
    ColorEncoding color = ColorEncoding::SoaF32;
    uint32_t geometryStride = 0, colorStride = 0, colorOffset = 0;
    VariableSh variableSh = VariableSh::Off;
    std::string fallbackNote;    // non-empty if the format differs from the requested one
    uint64_t totalBytes = 0;
    std::array<uint64_t, 4> bufferBytes{};
    double encodeMs = 0.0;       // CPU encoding of the batches (0 for the zero-copy f32 soa path)
    double uploadMs = 0.0;       // queue.writeBuffer calls (the browser copies the data)
    uint32_t batches = 0;

private:
    bool createBuffers(WGPUDevice device, WGPUBindGroupLayout layout, const std::array<uint64_t, 4>& sizes,
                       std::string* error);

    CompressionOptions encodeOptions_;
    std::array<uint32_t, 4> wordsPerGaussian_{};  // per buffer (0 = no buffer)
    bool zeroCopy_ = false;
    std::array<WGPUBuffer, 4> buffers_{};
    WGPUBindGroup bindGroup_ = nullptr;
};

}  // namespace gsr
