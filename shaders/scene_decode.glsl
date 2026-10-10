// Scene decoder (Phase 5): reads one gaussian from the scene buffers in any scene format.
// C++ side: src/render/gpu_scene.h (bindings, specialization constants) and
// src/scene/scene_compression.h (record layouts, the encoder, and decodeGaussian(), the CPU
// twin of the functions below; keep the two in sync).
//
// The including shader must declare nothing at set 0, bindings 0-3. The format is fixed at
// pipeline creation through specialization constants, so every branch on k* below is
// resolved by the driver's compiler and the unused decode paths disappear.
//
//   loadPosAlpha(i)          position (always fp32) + opacity
//   loadShape(i, s, q)       sigma (scale) and unit quaternion (w, x, y, z) in .xyzw
//   loadSH(i, used, c)       the first `used` SH RGB triplets (c[k] = 0 beyond)
//   loadDC(i)                SH coefficient 0 (the point-cloud view's colour)
//   gaussianShDegree(i)      Phase 6: the gaussian's own SH degree (3 unless --sh-threshold)

#ifndef SCENE_DECODE_GLSL
#define SCENE_DECODE_GLSL

#extension GL_EXT_control_flow_attributes : require

// GeometryEncoding / ColorEncoding in scene_compression.h.
const uint kGeomSoaF32 = 0u;
const uint kGeomF32 = 1u;
const uint kGeomF16 = 2u;
const uint kColorSoaF32 = 0u;
const uint kColorF32 = 1u;
const uint kColorF16 = 2u;
const uint kColorQ8 = 3u;
const uint kColorCodebook = 4u;

layout(constant_id = 0) const uint kGeometryEncoding = 0u;
layout(constant_id = 1) const uint kColorEncoding = 0u;
layout(constant_id = 2) const uint kGeometryStride = 0u;  // uvec4 per geometry record
layout(constant_id = 3) const uint kColorStride = 0u;     // uvec4 per colour record (0: CB, 2 records per uvec4)
layout(constant_id = 4) const uint kColorOffset = 0u;     // uvec4 offset of the colour record (aos)
layout(constant_id = 5) const uint kStoredCoeffs = 16u;   // K stored per gaussian (record formats)
layout(constant_id = 6) const uint kCodebookStride = 12u; // uvec4 per codebook entry
// Phase 6 per-gaussian SH degree (VariableSh in scene_compression.h): 0 off, 1 degree only
// (aos, cb: fixed-size records), 2 degree + compacted colour data (soa, split).
layout(constant_id = 7) const uint kVariableSh = 0u;

// Binding 0: soa posAlpha vec4[] | geometry records | aos records.
layout(std430, set = 0, binding = 0) readonly buffer SceneBuffer0 { uvec4 sceneB0[]; };
// Binding 1: soa scales vec4[]  | colour records    | aos records (same buffer as 0).
layout(std430, set = 0, binding = 1) readonly buffer SceneBuffer1 { uvec4 sceneB1[]; };
// Binding 2: soa rotations vec4[] | codebook (cb).
layout(std430, set = 0, binding = 2) readonly buffer SceneBuffer2 { uvec4 sceneB2[]; };
// Binding 3: soa sh float[] (fetched one float at a time, exactly as in Phases 1-4).
layout(std430, set = 0, binding = 3) readonly buffer SceneBuffer3 { float sceneSh[]; };

vec4 loadPosAlpha(uint i) {
    if (kGeometryEncoding == kGeomSoaF32) return uintBitsToFloat(sceneB0[i]);
    uvec4 w = sceneB0[i * kGeometryStride];
    if (kGeometryEncoding == kGeomF32) return uintBitsToFloat(w);
    return vec4(uintBitsToFloat(w.xyz), unpackHalf2x16(w.w).x);  // kGeomF16
}

void loadShape(uint i, out vec3 sigma, out vec4 q) {
    if (kGeometryEncoding == kGeomSoaF32) {
        sigma = uintBitsToFloat(sceneB1[i].xyz);
        q = uintBitsToFloat(sceneB2[i]);
    } else if (kGeometryEncoding == kGeomF32) {
        uint base = i * kGeometryStride;
        sigma = uintBitsToFloat(sceneB0[base + 1u].xyz);
        q = uintBitsToFloat(sceneB0[base + 2u]);
    } else {  // kGeomF16: word 3 = h(alpha, sx), 4 = h(sy, sz), 5 = h(qw, qx), 6 = h(qy, qz)
        uint base = i * kGeometryStride;
        uint w3 = sceneB0[base].w;
        uvec4 b = sceneB0[base + 1u];
        sigma = vec3(unpackHalf2x16(w3).y, unpackHalf2x16(b.x));
        vec4 r = vec4(unpackHalf2x16(b.y), unpackHalf2x16(b.z));
        float len2 = dot(r, r);
        q = len2 > 0.0 ? r * inversesqrt(len2) : vec4(1.0, 0.0, 0.0, 0.0);
    }
}

// Phase 6: the spare word holding (degree << 30 | compacted offset): soa scales.w, record word 7.
uint shMeta(uint i) {
    if (kGeometryEncoding == kGeomSoaF32) return sceneB1[i].w;
    return sceneB0[i * kGeometryStride + 1u].w;
}
uint gaussianShDegree(uint i) { return kVariableSh == 0u ? 3u : shMeta(i) >> 30; }

// Colour record base (uvec4 index into sceneB1) of gaussian i (not for CB split).
uint colorBase(uint i) {
    if (kVariableSh == 2u) return shMeta(i) & 0x3FFFFFFFu;
    return i * kColorStride + kColorOffset;
}
// soa: first float of gaussian i's SH in sceneSh.
uint soaShBase(uint i) {
    if (kVariableSh == 2u) return shMeta(i) & 0x3FFFFFFFu;
    return i * kStoredCoeffs * 3u;
}

// CB record: (h(dc.r, dc.g), h(dc.b, 0) | index << 16).
uvec2 codebookRecord(uint i) {
    if (kColorStride == 0u) {
        uvec4 v = sceneB1[i >> 1];
        return (i & 1u) != 0u ? v.zw : v.xy;
    }
    return sceneB1[colorBase(i)].xy;
}

// Word j of a record of 32-bit words held in uvec4 registers.
#define REC_WORD(arr, j) arr[(j) >> 2][(j) & 3u]

void loadSH(uint i, uint used, out vec3 c[16]) {
    [[unroll]] for (uint k = 0u; k < 16u; ++k) c[k] = vec3(0.0);
    if (kVariableSh != 0u) {
        uint d = gaussianShDegree(i);
        used = min(used, (d + 1u) * (d + 1u));
    }

    if (kColorEncoding == kColorSoaF32) {
        // Phase 1-4 path, unchanged: K comes from the stored layout (kStoredCoeffs).
        uint base = soaShBase(i);
        [[unroll]] for (uint k = 0u; k < 16u; ++k)
            if (k < used) c[k] = vec3(sceneSh[base + k * 3u], sceneSh[base + k * 3u + 1u], sceneSh[base + k * 3u + 2u]);
        return;
    }

    if (kColorEncoding == kColorF32) {
        // K*3 floats; only the uvec4s that hold the used coefficients are fetched.
        uint base = colorBase(i);
        uvec4 w[12];
        [[unroll]] for (uint v = 0u; v < 12u; ++v)
            w[v] = (v * 4u < kStoredCoeffs * 3u && v * 4u < used * 3u) ? sceneB1[base + v] : uvec4(0u);
        [[unroll]] for (uint k = 0u; k < 16u; ++k)
            if (k < used)
                c[k] = uintBitsToFloat(uvec3(REC_WORD(w, k * 3u), REC_WORD(w, k * 3u + 1u), REC_WORD(w, k * 3u + 2u)));
        return;
    }

    if (kColorEncoding == kColorF16) {
        // K*3 halves, two per word.
        uint base = colorBase(i);
        uvec4 w[6];
        [[unroll]] for (uint v = 0u; v < 6u; ++v)
            w[v] = (v * 8u < kStoredCoeffs * 3u && v * 8u < used * 3u) ? sceneB1[base + v] : uvec4(0u);
        [[unroll]] for (uint k = 0u; k < 16u; ++k) {
            if (k >= used) continue;
            vec3 v;
            [[unroll]] for (uint ch = 0u; ch < 3u; ++ch) {
                uint h = k * 3u + ch;
                vec2 pair = unpackHalf2x16(REC_WORD(w, h >> 1));
                v[ch] = (h & 1u) != 0u ? pair.y : pair.x;
            }
            c[k] = v;
        }
        return;
    }

    if (kColorEncoding == kColorQ8) {
        // w0 = h(dc.r, dc.g), w1 = h(dc.b, s1), w2 = h(s2, s3), int8 rest from byte 12.
        uint base = colorBase(i);
        uvec4 w[4];
        [[unroll]] for (uint v = 0u; v < 4u; ++v)
            w[v] = (v * 16u < 12u + (kStoredCoeffs - 1u) * 3u && v * 16u < 12u + (used - 1u) * 3u) ? sceneB1[base + v]
                                                                                                    : uvec4(0u);
        vec2 dcRG = unpackHalf2x16(w[0].x), dcB1 = unpackHalf2x16(w[0].y), s23 = unpackHalf2x16(w[0].z);
        c[0] = vec3(dcRG, dcB1.x);
        float scale[4] = float[](0.0, dcB1.y, s23.x, s23.y);
        [[unroll]] for (uint k = 1u; k < 16u; ++k) {
            if (k >= used) continue;
            uint band = k < 4u ? 1u : (k < 9u ? 2u : 3u);
            vec3 v;
            [[unroll]] for (uint ch = 0u; ch < 3u; ++ch) {
                uint j = (k - 1u) * 3u + ch;
                int q = bitfieldExtract(int(REC_WORD(w, 3u + (j >> 2))), int(8u * (j & 3u)), 8);  // sign-extended
                v[ch] = float(q) * scale[band];
            }
            c[k] = v;
        }
        return;
    }

    // kColorCodebook: DC from the record, rest from codebook entry `index` (fp32).
    uvec2 r = codebookRecord(i);
    c[0] = vec3(unpackHalf2x16(r.x), unpackHalf2x16(r.y).x);
    if (used > 1u && kStoredCoeffs > 1u) {
        uint base = (r.y >> 16) * kCodebookStride;
        uvec4 w[12];
        [[unroll]] for (uint v = 0u; v < 12u; ++v)
            w[v] = (v * 4u < (kStoredCoeffs - 1u) * 3u && v * 4u < (used - 1u) * 3u) ? sceneB2[base + v] : uvec4(0u);
        [[unroll]] for (uint k = 1u; k < 16u; ++k)
            if (k < used)
                c[k] = uintBitsToFloat(uvec3(REC_WORD(w, (k - 1u) * 3u), REC_WORD(w, (k - 1u) * 3u + 1u),
                                             REC_WORD(w, (k - 1u) * 3u + 2u)));
    }
}

vec3 loadDC(uint i) {
    if (kColorEncoding == kColorSoaF32) {
        uint base = soaShBase(i);
        return vec3(sceneSh[base], sceneSh[base + 1u], sceneSh[base + 2u]);
    }
    if (kColorEncoding == kColorF32) return uintBitsToFloat(sceneB1[colorBase(i)].xyz);
    if (kColorEncoding == kColorF16) {
        uvec4 w = sceneB1[colorBase(i)];
        return vec3(unpackHalf2x16(w.x), unpackHalf2x16(w.y).x);
    }
    if (kColorEncoding == kColorQ8) {
        uvec4 w = sceneB1[colorBase(i)];
        return vec3(unpackHalf2x16(w.x), unpackHalf2x16(w.y).x);
    }
    uvec2 r = codebookRecord(i);
    return vec3(unpackHalf2x16(r.x), unpackHalf2x16(r.y).x);
}

#undef REC_WORD

#endif  // SCENE_DECODE_GLSL
