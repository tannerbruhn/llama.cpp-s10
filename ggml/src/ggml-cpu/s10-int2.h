#pragma once

// S10 spec C: CPU rows of PTQ1_0 mat-muls from in-RAM 2-bit ("s2") codes when GGML_S10_INT2 is set.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_s10i2 {
    const void * key;          // src0->data
    const uint8_t * codes;     // rows [row0, row0 + rows)
    const uint16_t * scales;   // fp16 block scales (ggml_half)
    int64_t row0, rows;
    int nb;                    // blocks per row
    size_t pf;                 // prefetch distance on the code stream in bytes (0 = off)
};

// Thread 0 only, before the barrier. Returns the entry if the CPU rows [gpu_rows, M) of src0 can be taken from
// s2 codes for this call, else NULL. Converts the tensor on first use.
const struct ggml_s10i2 * ggml_s10i2_get(const struct ggml_tensor * src0, const struct ggml_tensor * src1, int64_t gpu_rows);

// Any thread, after the barrier: rows [r0, r1) x columns [c0, c1) of dst from the s2 codes and the Q8_0 columns in y.
void ggml_s10i2_chunk(const struct ggml_s10i2 * e, int n, const void * y, size_t y_stride,
                      struct ggml_tensor * dst, int64_t r0, int64_t r1, int64_t c0, int64_t c1);

#ifdef __cplusplus
}
#endif
