// IEEE 754 binary16 ("half") conversion on the CPU (Phase 5 scene compression).
//
// binary16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
//   normal numbers  2^-14 (6.10e-5) .. 65504, relative spacing 2^-10, so round-to-nearest
//                   has a relative error <= 2^-11 = 4.88e-4
//   subnormals      2^-24 (5.96e-8) .. 2^-14, fixed absolute spacing 2^-24 (the relative
//                   error grows as the value shrinks)
//   above 65504     rounds to infinity (we never encode such values: the encoders check)
//
// floatToHalf rounds to nearest, ties to even (the IEEE default, and what F16C's
// _mm_cvtps_ph does with rounding mode 0), handles subnormals, infinities and NaN.
// halfToFloat is exact (every half is representable as a float). The GPU side only ever
// DECODES (unpackHalf2x16, exact), so the CPU encoder alone defines the rounding.
#pragma once

#include <cstdint>
#include <cstring>

namespace gsr {

inline uint32_t floatToBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float bitsToFloat(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t floatToHalf(float value) {
    const uint32_t f = floatToBits(value);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const uint32_t absBits = f & 0x7FFFFFFFu;
    if (absBits >= 0x7F800000u) {  // Inf or NaN
        return uint16_t(sign | 0x7C00u | (absBits > 0x7F800000u ? 0x200u : 0u));
    }
    if (absBits >= 0x477FF000u) return uint16_t(sign | 0x7C00u);  // >= 65520 rounds to Inf
    if (absBits < 0x38800000u) {
        // Result is subnormal (or zero): value = m * 2^-24 with m < 1024. Shift the full
        // 24-bit significand right and round to nearest even.
        if (absBits < 0x33000000u) return uint16_t(sign);  // < 2^-25: rounds to 0
        const uint32_t exponent = absBits >> 23;              // 102 .. 112
        const uint32_t mantissa = (absBits & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - exponent;               // 14 .. 24
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
        return uint16_t(sign | half);
    }
    // Normal: rebias the exponent (127 -> 15) and round the 13 dropped mantissa bits.
    uint32_t half = ((absBits - 0x38000000u) >> 13);
    const uint32_t rest = absBits & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;  // may carry into the exponent: correct
    return uint16_t(sign | half);
}

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        // Zero or subnormal: m * 2^-24 (exact in float).
        const float v = float(mantissa) * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    if (exponent == 31) return bitsToFloat(sign | 0x7F800000u | (mantissa << 13));
    return bitsToFloat(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

// The value a float takes after a round trip through half.
inline float roundToHalf(float v) { return halfToFloat(floatToHalf(v)); }

// GLSL packHalf2x16 / unpackHalf2x16 layout: first component in the low 16 bits.
inline uint32_t packHalf2(float lo, float hi) { return uint32_t(floatToHalf(lo)) | (uint32_t(floatToHalf(hi)) << 16); }
inline float unpackHalfLo(uint32_t w) { return halfToFloat(uint16_t(w & 0xFFFFu)); }
inline float unpackHalfHi(uint32_t w) { return halfToFloat(uint16_t(w >> 16)); }

}  // namespace gsr
