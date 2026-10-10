#include "webgpu/wgpu_resources.h"

#include "webgpu/wgpu_context.h"

namespace gsr {

void WgpuTexture2D::create(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat fmt,
                           WGPUTextureUsage usage, const char* label) {
    release();
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.label = toStringView(label);
    desc.usage = usage;
    desc.dimension = WGPUTextureDimension_2D;
    desc.size = {w, h, 1};
    desc.format = fmt;
    texture = wgpuDeviceCreateTexture(device, &desc);
    view = wgpuTextureCreateView(texture, nullptr);
    width = w;
    height = h;
    format = fmt;
    static uint64_t nextGeneration = 1;
    generation = nextGeneration++;
}

void WgpuTexture2D::release() {
    if (view) wgpuTextureViewRelease(view);
    if (texture) {
        wgpuTextureDestroy(texture);
        wgpuTextureRelease(texture);
    }
    view = nullptr;
    texture = nullptr;
    width = height = 0;
}

}  // namespace gsr
