// In-page GPU self-test of the WGSL scene decoder (gsr_selftest("decode"), web phase W1): random
// gaussians are encoded on the CPU in every web scene format (incl. reduced SH degree, the split
// f32 records and the Phase 6 variable SH degree), uploaded, decoded on the GPU by
// shaders/wgsl/decode_test.wgsl and compared with decodeGaussian(), the CPU decoder. One case
// also streams its scene in uneven batches through WgpuScene::create/upload (the load path).
// The web counterpart of the decoder part of tests/test_scene_formats.cpp.
//
// Asynchronous: start() records and submits everything; the results arrive with the buffer
// maps; done() turns true when every case has been compared.
#pragma once

#include <webgpu/webgpu.h>

#include <memory>
#include <string>
#include <vector>

namespace gsr {

class WgslShaderCache;

class WgpuDecodeSelftest {
public:
    WgpuDecodeSelftest();
    ~WgpuDecodeSelftest();
    WgpuDecodeSelftest(const WgpuDecodeSelftest&) = delete;
    WgpuDecodeSelftest& operator=(const WgpuDecodeSelftest&) = delete;

    bool started() const;
    void start(WGPUDevice device, WGPUQueue queue, WgslShaderCache& shaders, WGPUBindGroupLayout sceneLayout);
    bool done() const;
    std::string resultJson() const;  // {"pass": bool, "cases": [...]}; {"pending": true} before done()

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gsr
