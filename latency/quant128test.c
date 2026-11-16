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

// ---- test driver: random + edge blocks, bitwise equality, then ns per value ----
static uint64_t rng = 88172645463325252ull;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (float) ((rng >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int compare(const float * x, int64_t nb, const char * what) {
#if defined(__ARM_NEON) && defined(__aarch64__)
    int8_t qa[17408], qb[17408]; float da[136], db[136]; int32_t aa[136], ab[136];
    s10ga_quant128_scalar(x, nb, qa, da, aa);
    s10ga_quant128_neon(x, nb, qb, db, ab);
    if (memcmp(qa, qb, nb * 128) || memcmp(da, db, nb * sizeof(float)) || memcmp(aa, ab, nb * sizeof(int32_t))) {
        printf("FAIL %s\n", what); return 1;
    }
    return 0;
#else
    (void) x; (void) nb; (void) what; return 0;
#endif
}

int main(void) {
    static float x[17408];
    int fail = 0;
    const int64_t nb = 136;
    for (int it = 0; it < 2000; it++) {                       // random scales, wide dynamic range
        const float s = powf(10.0f, (float) (it % 13) - 6.0f);
        for (int j = 0; j < 17408; j++) x[j] = frand() * s;
        fail |= compare(x, nb, "random");
    }
    memset(x, 0, sizeof(x)); fail |= compare(x, nb, "all zero");
    for (int j = 0; j < 17408; j++) x[j] = frand(); x[5] = 1e30f; x[300] = -3e38f; fail |= compare(x, nb, "huge value");
    for (int b = 0; b < nb; b++) for (int j = 0; j < 128; j++) x[b * 128 + j] = (j == 0 ? 127.0f : (float) (j % 64) - 31.5f);
    fail |= compare(x, nb, "exact .5 ties");
    for (int j = 0; j < 17408; j++) x[j] = frand() * 1e-40f; fail |= compare(x, nb, "denormals");
    for (int j = 0; j < 17408; j++) x[j] = (j % 128 == 7) ? 1e-44f : 0.0f; fail |= compare(x, nb, "single denormal");
    printf("%s\n", fail ? "QUANT128 FAILED" : "QUANT128 OK: NEON bit-identical to scalar on all cases");
#if defined(__ARM_NEON) && defined(__aarch64__)
    static int8_t q[17408]; static float d[136]; static int32_t a[136];
    for (int j = 0; j < 17408; j++) x[j] = frand();
    for (int k = 0; k < 2; k++) {
        const int64_t n = k ? 136 : 40;
        double ts = 1e9, tn = 1e9;
        for (int r = 0; r < 200; r++) {
            double t0 = now(); s10ga_quant128_scalar(x, n, q, d, a); double t1 = now(); s10ga_quant128_neon(x, n, q, d, a); double t2 = now();
            if (t1 - t0 < ts) ts = t1 - t0; if (t2 - t1 < tn) tn = t2 - t1;
        }
        printf("K=%lld: scalar %.3f ns/value, neon %.3f ns/value (x%.1f)\n", (long long) (n * 128), ts * 1e9 / (n * 128), tn * 1e9 / (n * 128), ts / tn);
    }
#endif
    return fail;
}
