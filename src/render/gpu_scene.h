// GpuScene: the gaussians in device-local storage buffers, in one of the Phase 5 scene
// formats (scene/scene_compression.h; `--scene-format`, `--scene-layout`).
//
// Owned by the Application, not by a renderer, so switching renderers never re-uploads
// the scene. Every shader that reads gaussians includes shaders/scene_decode.glsl, binds
// the scene's FOUR buffers at its bindings 0-3 (bind()) and is specialized on the format
// (addSpecialization(), constant_id 0-6), so one SPIR-V file decodes every format:
//
//   binding  f32 soa (default, Phase 1-4)       split                      aos
//   0        posAlpha  vec4[N]                  geometry records           records (geometry + colour)
//   1        scales    vec4[N]                  colour records             records (same buffer)
//   2        rotations vec4[N]                  codebook (cb) / unused     codebook (cb) / unused
//   3        sh        float[N*K*3]             unused                     unused
//
// "unused" bindings hold a valid buffer (the scene's first one) that is never read.
#pragma once

#include "core/resources.h"
#include "scene/scene_compression.h"

#include <array>

namespace gsr {

class VulkanContext;
class DescriptorWriter;
class SpecializationConstants;
struct GaussianScene;

// constant_id values of shaders/scene_decode.glsl.
enum SceneSpecId : uint32_t {
    kSpecGeometryEncoding = 0,
    kSpecColorEncoding = 1,
    kSpecGeometryStride = 2,
    kSpecColorStride = 3,
    kSpecColorOffset = 4,
    kSpecStoredCoeffs = 5,
    kSpecCodebookStride = 6,
    kSpecVariableSh = 7,  // Phase 6
};

struct GpuScene {
    uint32_t count = 0;
    int shDegree = 0;                 // STORED degree (after --scene-sh-degree)
    uint32_t shCoeffsPerChannel = 1;  // stored K = (shDegree + 1)^2
    SceneFormat format = SceneFormat::F32;
    SceneLayout layout = SceneLayout::Soa;
    GeometryEncoding geometry = GeometryEncoding::SoaF32;
    ColorEncoding color = ColorEncoding::SoaF32;
    uint32_t geometryStride = 0, colorStride = 0, colorOffset = 0, codebookStride = 0;
    VariableSh variableSh = VariableSh::Off;    // Phase 6 per-gaussian SH degree
    std::array<uint32_t, 4> degreeHistogram{};  // gaussians per kept degree

    std::array<Buffer, 4> buffers;  // see the binding table above (empty entries: not allocated)

    VkDeviceSize totalBytes = 0;
    double encodeMs = 0.0;  // CPU encode (incl. codebook build), 0 for the zero-copy f32 soa path
    double uploadMs = 0.0;  // staging copy + GPU copy
    CodebookStats codebook;

    bool valid() const { return count > 0; }
    double bytesPerGaussian() const { return count ? double(totalBytes) / double(count) : 0.0; }
    // Encodes (unless f32 soa at full degree: uploaded straight from `scene`), then one
    // staging buffer for all arrays and one blocking submit (init time only).
    // `preEncoded`: encodeScene(scene, options) already done elsewhere (the scene-loading
    // worker thread); it is consumed. On failure (e.g. out of GPU memory) everything
    // allocated so far is freed, the GpuScene is left empty and the exception propagates.
    void upload(VulkanContext& ctx, const GaussianScene& scene, const CompressionOptions& options = {},
                EncodedScene* preEncoded = nullptr);
    // True if upload() needs no encoding: the GaussianScene arrays are the GPU layout.
    static bool zeroCopy(const GaussianScene& scene, const CompressionOptions& options);
    void destroy(VulkanContext& ctx);

    // Writes the scene buffers at bindings first .. first+3 (STORAGE_BUFFER).
    void bind(DescriptorWriter& writer, uint32_t firstBinding = 0) const;
    // Sets constant_id 0-6 of shaders/scene_decode.glsl.
    void addSpecialization(SpecializationConstants& spec) const;
};

}  // namespace gsr
