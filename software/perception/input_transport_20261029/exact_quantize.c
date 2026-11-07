/* Model-specific SDK-equivalent quantization; see verification artifacts.
 * Gated by fixed model/runtime identities in exact_input.py, not a generic
 * implementation of affine quantization from the public tensor scale.
 */
#include <arm_neon.h>
#include <fenv.h>
#include <stdint.h>
#include <stddef.h>

int exact_quantize(const float *input, int8_t *output, size_t count) {
    if (!input || !output || count != 196608 || fegetround() != FE_TONEAREST) return -1;
    const float32x4_t multiplier = vdupq_n_f32(235.34375f);
    const float32x4_t offset = vdupq_n_f32(-112.0f);
    for (size_t index = 0; index < count; index += 16) {
        int32x4_t a = vcvtnq_s32_f32(vfmaq_f32(offset, vld1q_f32(input + index), multiplier));
        int32x4_t b = vcvtnq_s32_f32(vfmaq_f32(offset, vld1q_f32(input + index + 4), multiplier));
        int32x4_t c = vcvtnq_s32_f32(vfmaq_f32(offset, vld1q_f32(input + index + 8), multiplier));
        int32x4_t d = vcvtnq_s32_f32(vfmaq_f32(offset, vld1q_f32(input + index + 12), multiplier));
        int16x8_t ab = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        int16x8_t cd = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
        vst1q_s8(output + index, vcombine_s8(vqmovn_s16(ab), vqmovn_s16(cd)));
    }
    return 0;
}
