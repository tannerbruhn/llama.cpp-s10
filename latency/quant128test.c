// Standalone: cc -O3 quant128test.c -lm -o quant128test
// On the phone add -march=armv8.2-a+dotprod+fp16. Do not use fast-math.
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

// Copies of the production helpers; keep in sync with s10-gpu-assist.c.
static void s10ga_quant128_scalar(const float * x, int64_t nb, int8_t * qx, float * dx, int32_t * as) {
    for (int64_t b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 128; j++) {
            amax = fmaxf(amax, fabsf(x[b * 128 + j]));
        }
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        dx[b] = d;
        int32_t sum = 0;
        for (int j = 0; j < 128; j++) {
            const int8_t q = (int8_t) lrintf(x[b * 128 + j] * id);
            qx[b * 128 + j] = q;
            sum += q;
        }
        as[b] = sum;
    }
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// One column, with the same scale arithmetic and nearest-even rounding as the scalar loop.
static void s10ga_quant128_neon(const float * x, int64_t nb, int8_t * qx, float * dx, int32_t * as) {
    for (int64_t b = 0; b < nb; b++, x += 128, qx += 128) {
        float32x4_t m = vdupq_n_f32(0.0f);
        for (int j = 0; j < 128; j += 4) m = vmaxnmq_f32(m, vabsq_f32(vld1q_f32(x + j)));
        const float amax = vmaxnmvq_f32(m);
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        // Very small denormal blocks can overflow the reciprocal; preserve scalar conversion semantics.
        if (!isfinite(id)) {
            s10ga_quant128_scalar(x, 1, qx, dx + b, as + b);
            continue;
        }
        dx[b] = d;
        int32_t sum = 0;
        for (int j = 0; j < 128; j += 16) {
            const int32x4_t a0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j),      id));
            const int32x4_t a1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 4),  id));
            const int32x4_t a2 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 8),  id));
            const int32x4_t a3 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + j + 12), id));
            const int8x16_t q = vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(a0), vqmovn_s32(a1))),
                                           vqmovn_s16(vcombine_s16(vqmovn_s32(a2), vqmovn_s32(a3))));
            vst1q_s8(qx + j, q);
            sum += vaddlvq_s8(q);
        }
        as[b] = sum;
    }
}
#endif
