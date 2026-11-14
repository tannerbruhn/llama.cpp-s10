#pragma once

// S10 GPU assist: during a PTQ1_0 MUL_MAT on the CPU backend, the first rows of the weight matrix are
// computed by the Mali GPU (OpenCL, Rusticl) while the CPU threads compute the rest.
// Enabled at run time by GGML_S10_GPU_FRAC=<fraction of rows for the GPU>, e.g. 0.37.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Thread 0 only, before the barrier. Starts the GPU work and returns the number of leading rows it
// took (a multiple of 64), or 0 when the assist does not apply.
int64_t ggml_s10ga_begin(const struct ggml_tensor * src0, const struct ggml_tensor * src1);

// Thread 0 only, after its CPU chunks. Waits for the GPU and writes its rows into dst.
void ggml_s10ga_end(struct ggml_tensor * dst, int64_t rows);

#ifdef __cplusplus
}
#endif
