// Port of shaders/scene_decode.glsl (web phase W1): reads one gaussian from the scene buffers in
// the scene formats the web offers. C++ side: src/webgpu/wgpu_scene.h (bindings, override
// constants) and src/scene/scene_compression.h (record layouts, the encoder and
// decodeGaussian(), the CPU twin of the functions below).
//
// Ported: f32 soa (the Phase 1-4 arrays), f32 / f16 / q8 records (split layout; aos works too,
// the record offsets are overrides), and the Phase 6 per-gaussian SH degree word. Not ported:
// the cb codebook (k-means at load time is too slow in single-threaded wasm, DESIGN.md 2.2).
// Differences from the GLSL: specialization constants are `override`s (the browser's compiler
// folds the branches when the pipeline is created), there are no out parameters (loadShape
// returns a struct, loadSH an array), and the [[unroll]] hints are gone.
//
// The including shader declares nothing else at group 0, bindings 0-3.
//
//   loadPosAlpha(i)          position (always fp32) + opacity
//   loadShape(i)             sigma (scale) and unit quaternion (w, x, y, z) in .xyzw
//   loadSH(i, used)          the first `used` SH RGB triplets (0 beyond)
//   loadDC(i)                SH coefficient 0 (the point view's colour)
//   gaussianShDegree(i)      Phase 6: the gaussian's own SH degree (3 unless variable)

// GeometryEncoding / ColorEncoding in scene_compression.h.
const kGeomSoaF32: u32 = 0u;
const kGeomF32: u32 = 1u;
const kGeomF16: u32 = 2u;
const kColorSoaF32: u32 = 0u;
const kColorF32: u32 = 1u;
const kColorF16: u32 = 2u;
const kColorQ8: u32 = 3u;

override kGeometryEncoding: u32 = 0u;
override kColorEncoding: u32 = 0u;
override kGeometryStride: u32 = 0u;  // vec4<u32> per geometry record
override kColorStride: u32 = 0u;     // vec4<u32> per colour record
override kColorOffset: u32 = 0u;     // vec4<u32> offset of the colour record (aos)
override kStoredCoeffs: u32 = 16u;   // K stored per gaussian
// Phase 6 per-gaussian SH degree (VariableSh): 0 off, 1 degree only, 2 degree + compacted colour.
override kVariableSh: u32 = 0u;

// Binding 0: soa posAlpha vec4[] | geometry records | aos records.
@group(0) @binding(0) var<storage, read> sceneB0: array<vec4<u32>>;
// Binding 1: soa scales vec4[]  | colour records.
@group(0) @binding(1) var<storage, read> sceneB1: array<vec4<u32>>;
// Binding 2: soa rotations vec4[] (unused otherwise: the codebook is not ported).
@group(0) @binding(2) var<storage, read> sceneB2: array<vec4<u32>>;
// Binding 3: soa sh float[] (fetched one float at a time, as in Phases 1-4).
@group(0) @binding(3) var<storage, read> sceneSh: array<f32>;

fn loadPosAlpha(i: u32) -> vec4<f32> {
    if (kGeometryEncoding == kGeomSoaF32) {
        return bitcast<vec4<f32>>(sceneB0[i]);
    }
    let w = sceneB0[i * kGeometryStride];
    if (kGeometryEncoding == kGeomF32) {
        return bitcast<vec4<f32>>(w);
    }
    return vec4<f32>(bitcast<vec3<f32>>(w.xyz), unpack2x16float(w.w).x);  // kGeomF16
}

struct GaussianShape {
    sigma: vec3<f32>,
    q: vec4<f32>,  // (w, x, y, z), unit
}

fn loadShape(i: u32) -> GaussianShape {
    var s: GaussianShape;
    if (kGeometryEncoding == kGeomSoaF32) {
        s.sigma = bitcast<vec3<f32>>(sceneB1[i].xyz);
        s.q = bitcast<vec4<f32>>(sceneB2[i]);
    } else if (kGeometryEncoding == kGeomF32) {
        let base = i * kGeometryStride;
        s.sigma = bitcast<vec3<f32>>(sceneB0[base + 1u].xyz);
        s.q = bitcast<vec4<f32>>(sceneB0[base + 2u]);
    } else {  // kGeomF16: word 3 = h(alpha, sx), 4 = h(sy, sz), 5 = h(qw, qx), 6 = h(qy, qz)
        let base = i * kGeometryStride;
        let w3 = sceneB0[base].w;
        let b = sceneB0[base + 1u];
        s.sigma = vec3<f32>(unpack2x16float(w3).y, unpack2x16float(b.x));
        let r = vec4<f32>(unpack2x16float(b.y), unpack2x16float(b.z));
        let len2 = dot(r, r);
        s.q = select(vec4<f32>(1.0, 0.0, 0.0, 0.0), r * inverseSqrt(len2), len2 > 0.0);
    }
    return s;
}

// Phase 6: the spare word holding (degree << 30 | compacted offset): soa scales.w, record word 7.
fn shMeta(i: u32) -> u32 {
    if (kGeometryEncoding == kGeomSoaF32) {
        return sceneB1[i].w;
    }
    return sceneB0[i * kGeometryStride + 1u].w;
}

fn gaussianShDegree(i: u32) -> u32 {
    if (kVariableSh == 0u) {
        return 3u;
    }
    return shMeta(i) >> 30u;
}

// Colour record base (vec4<u32> index into sceneB1) of gaussian i.
fn colorBase(i: u32) -> u32 {
    if (kVariableSh == 2u) {
        return shMeta(i) & 0x3FFFFFFFu;
    }
    return i * kColorStride + kColorOffset;
}

// soa: first float of gaussian i's SH in sceneSh.
fn soaShBase(i: u32) -> u32 {
    if (kVariableSh == 2u) {
        return shMeta(i) & 0x3FFFFFFFu;
    }
    return i * kStoredCoeffs * 3u;
}

// The first `usedIn` SH coefficients (RGB triplets) of gaussian i; the rest are 0.
fn loadSH(i: u32, usedIn: u32) -> array<vec3<f32>, 16> {
    var c: array<vec3<f32>, 16>;  // zero-initialized
    var used = usedIn;
    if (kVariableSh != 0u) {
        let d = gaussianShDegree(i);
        used = min(used, (d + 1u) * (d + 1u));
    }

    if (kColorEncoding == kColorSoaF32) {
        let base = soaShBase(i);
        for (var k = 0u; k < 16u; k++) {
            if (k < used) {
                c[k] = vec3<f32>(sceneSh[base + k * 3u], sceneSh[base + k * 3u + 1u], sceneSh[base + k * 3u + 2u]);
            }
        }
        return c;
    }

    if (kColorEncoding == kColorF32) {
        // K*3 floats; only the vec4s that hold the used coefficients are fetched.
        let base = colorBase(i);
        var w: array<vec4<u32>, 12>;
        for (var v = 0u; v < 12u; v++) {
            if (v * 4u < kStoredCoeffs * 3u && v * 4u < used * 3u) {
                w[v] = sceneB1[base + v];
            }
        }
        for (var k = 0u; k < 16u; k++) {
            if (k < used) {
                let j = k * 3u;
                c[k] = bitcast<vec3<f32>>(vec3<u32>(w[j >> 2u][j & 3u], w[(j + 1u) >> 2u][(j + 1u) & 3u],
                                                    w[(j + 2u) >> 2u][(j + 2u) & 3u]));
            }
        }
        return c;
    }

    if (kColorEncoding == kColorF16) {
        // K*3 halves, two per word.
        let base = colorBase(i);
        var w: array<vec4<u32>, 6>;
        for (var v = 0u; v < 6u; v++) {
            if (v * 8u < kStoredCoeffs * 3u && v * 8u < used * 3u) {
                w[v] = sceneB1[base + v];
            }
        }
        for (var k = 0u; k < 16u; k++) {
            if (k < used) {
                var value: vec3<f32>;
                for (var ch = 0u; ch < 3u; ch++) {
                    let h = k * 3u + ch;
                    let word = h >> 1u;
                    let pair = unpack2x16float(w[word >> 2u][word & 3u]);
                    value[ch] = select(pair.x, pair.y, (h & 1u) != 0u);
                }
                c[k] = value;
            }
        }
        return c;
    }

    // kColorQ8: w0 = h(dc.r, dc.g), w1 = h(dc.b, s1), w2 = h(s2, s3), int8 rest from byte 12.
    let base = colorBase(i);
    var w: array<vec4<u32>, 4>;
    for (var v = 0u; v < 4u; v++) {
        if (v * 16u < 12u + (kStoredCoeffs - 1u) * 3u && v * 16u < 12u + (used - 1u) * 3u) {
            w[v] = sceneB1[base + v];
        }
    }
    let dcRG = unpack2x16float(w[0].x);
    let dcB1 = unpack2x16float(w[0].y);
    let s23 = unpack2x16float(w[0].z);
    c[0] = vec3<f32>(dcRG, dcB1.x);
    var scale = array<f32, 4>(0.0, dcB1.y, s23.x, s23.y);  // var: indexed at run time
    for (var k = 1u; k < 16u; k++) {
        if (k < used) {
            let band = select(select(3u, 2u, k < 9u), 1u, k < 4u);
            var value: vec3<f32>;
            for (var ch = 0u; ch < 3u; ch++) {
                let j = (k - 1u) * 3u + ch;
                let word = 3u + (j >> 2u);
                let q = extractBits(bitcast<i32>(w[word >> 2u][word & 3u]), 8u * (j & 3u), 8u);  // sign-extended
                value[ch] = f32(q) * scale[band];
            }
            c[k] = value;
        }
    }
    return c;
}

fn loadDC(i: u32) -> vec3<f32> {
    if (kColorEncoding == kColorSoaF32) {
        let base = soaShBase(i);
        return vec3<f32>(sceneSh[base], sceneSh[base + 1u], sceneSh[base + 2u]);
    }
    if (kColorEncoding == kColorF32) {
        return bitcast<vec3<f32>>(sceneB1[colorBase(i)].xyz);
    }
    // f16 and q8 both start with h(dc.r, dc.g), h(dc.b, .).
    let w = sceneB1[colorBase(i)];
    return vec3<f32>(unpack2x16float(w.x), unpack2x16float(w.y).x);
}
