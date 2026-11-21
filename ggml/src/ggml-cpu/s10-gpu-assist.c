// S10 GPU assist, see s10-gpu-assist.h.
//
// Weights: on first use, the GPU's rows of a PTQ1_0 tensor are re-encoded once into a 2-bit layout the
// kernel decodes cheaply (32 B of codes per 128 weights, all codes first, then the fp16 block scales)
// and kept in a host-visible OpenCL buffer. Their pages in the model mmap are never read again.
// Activations: quantized to int8 with one fp32 scale per 128 values for every call.

#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <sched.h>
#include <CL/cl.h>

#include "s10-gpu-assist.h"
#include "s10-gpu-assist-kernel.h"
#include "s10-gpu-assist-mm4.h"
#include "s10-gpu-assist-mm2.h"
#include "s10-gpu-assist-mm8.h"
#include "ggml-impl.h"

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "s10-direct.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

// Keep the cached row count aligned to both kernels. Zero slots selects the tuned geometry.
static int g_slots = 0;
#define S10GA_TILE_ROWS   8
#define S10GA_LANES      16
#define S10GA_MAX_ROWS    32768  // rows per dispatch, keeps each GPU job short (Panfrost job timeout)
#define S10GA_TABLE       4096

struct s10ga_entry {
    const void * data;   // src0->data, the key
    cl_mem       w;      // re-encoded rows [0, rows)
    int64_t      rows;   // 0 = assist not used for this tensor
    char         name[GGML_MAX_NAME];
    struct s10ga_entry * partner;   // ffn_gate -> its ffn_up (same input): both GPU shares go in one submit
    struct s10d_buf dw;             // GGML_S10_GPU_DIRECT: the same rows in a Panfrost buffer (w stays NULL)
};
// GGML_S10_GPU_DIRECT=1: submit the kernels through s10-direct.c (Panfrost ioctls, pre-compiled shaders) instead of
// Rusticl. Same kernels and geometry, bit-identical results, ~2x faster per call. Only the 8-row mv kernel and the
// mm2/mm4 kernels are embedded: a tensor that would need the 4-row or tail kernel stays on the CPU.
static int g_direct, g_d_mm2 = 1, g_d_mm4 = 1;
static struct s10d_buf g_dqx, g_ddx, g_das, g_dy, g_dy2;

// Fused gate+up (GGML_S10_GPU_FUSE=0 disables): the gate call also queues the up tensor's GPU rows, which
// the following up call (same src1) then only collects.
static int     g_fuse = 1;
static struct s10ga_entry * g_pend;
static const void * g_pend_x;
static int64_t g_pend_ncols;
static cl_event g_pend_kev, g_pend_rev;
static cl_mem  g_by2;
static size_t  g_cap_y2, g_hcap_y2;
static float * g_hy2;
static int     g_cur_is_pend;
static long    g_n_used, g_n_dropped, g_n_queued;

static int        g_state = -1;   // -1 not initialized, 0 off, 1 on
static float      g_frac;
static int        g_check;
static int        g_late, g_qneon, g_qcheck, g_deferred;
static struct s10ga_entry * g_def_e;
static const struct ggml_tensor * g_def_src0, * g_def_src1;
// per-shape shares (default: GGML_S10_GPU_FRAC): GGML_S10_GPU_FRAC_UPGATE, _DOWN, _OTHER
static float      g_frac_upgate = -1, g_frac_down = -1, g_frac_other = -1;
static float s10ga_frac_for(const char * name) {
    if (strstr(name, "ffn_down")) return g_frac_down >= 0 ? g_frac_down : g_frac;
    if (strstr(name, "ffn_gate") || strstr(name, "ffn_up")) return g_frac_upgate >= 0 ? g_frac_upgate : g_frac;
    return g_frac_other >= 0 ? g_frac_other : g_frac;
}
static double     g_min_bytes;   // GGML_S10_GPU_MINMB: only tensors at least this big (MiB) get GPU rows
static cl_context g_ctx;
static cl_command_queue g_q;
static cl_kernel  g_kern_mm2;    // 2-token batches (speculative checks): 4 rows x 2 columns; GGML_S10_GPU_MM2=0 disables
static cl_kernel  g_kern, g_kern8, g_kern_tail, g_kern_mm4; // mm4: prompt kernel, 4 rows x g_mmc columns per item
static int        g_mmc = 4;    // columns per prompt work-item (GGML_S10_GPU_MMC=8 selects the 8-column kernel)
static struct s10ga_entry g_tab[S10GA_TABLE];
static size_t     g_gpu_bytes;
static int        g_ntensors;

static int    g_prof, g_xfer, g_hprof, g_fake;   // GGML_S10_GPU_FAKE: skip the GPU entirely (control experiment; rows left unset)   // GGML_S10_GPU_HPROF: host-clock timing only (no GPU profiling jobs)
static volatile double g_t_done;
static double g_h_lat, g_h_cpu, g_h_wait, g_t_flush; static long g_h_n;
static void CL_CALLBACK s10ga_done_cb(cl_event ev, cl_int st, void * ud) { (void) ev; (void) st; (void) ud; struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g_t_done = t.tv_sec + t.tv_nsec * 1e-9; }   // GGML_S10_GPU_PROF: timing; GGML_S10_GPU_XFER=1: write/read instead of map
static double g_t[6], g_tcpu, g_tgpu, g_tbegin_end, g_tq2sub, g_tsub2start;
static cl_event g_last_ev;
static cl_event g_read_ev;   // non-blocking result read queued in begin(), polled in end()
static long   g_calls;
// Cumulative host timings: latency/waits cover submits, plan covers submits + fused collects.
static double g_h_plan, g_h_cpu_wait, g_h_gpu_idle;
static long g_h_collect;
static int g_h_t0cpu = -1;
static int8_t * g_hqx; static float * g_hdx, * g_hy; static size_t g_hcap_qx, g_hcap_dx, g_hcap_y;
static double s10ga_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void * s10ga_host(void * p, size_t * cap, size_t need) {
    if (*cap >= need) return p;
    free(p); *cap = need; return malloc(need);
}
static cl_mem g_bqx, g_bdx, g_by, g_bas;   // g_bas: per-block sums of the int8 activations
static size_t g_cap_as; static int32_t * g_has; static size_t g_hcap_as;
// Default "spin": thread 0 busy-polls the kernel event before waiting for the queued result read. Rusticl
// holds an event's mutex across the read's bo_wait, so polling only the read event sleeps on a futex; then
// schedutil sees idle time and lowers the big-core clocks. (A persistently mapped output buffer returned stale
// data on Rusticl, so the read stays.) GGML_S10_GPU_WAIT=read: wait on the read event only.
static int    g_spin = 1;
static size_t g_cap_qx, g_cap_dx, g_cap_y;

static int s10ga_init_cl(void);

static int s10ga_init(void) {
    const char * s = getenv("GGML_S10_GPU_FRAC");
    g_frac  = s ? (float) atof(s) : 0.0f;
    g_check = getenv("GGML_S10_GPU_CHECK") != NULL;
    if (getenv("GGML_S10_GPU_FRAC_UPGATE")) g_frac_upgate = (float) atof(getenv("GGML_S10_GPU_FRAC_UPGATE"));
    if (getenv("GGML_S10_GPU_FRAC_DOWN"))   g_frac_down   = (float) atof(getenv("GGML_S10_GPU_FRAC_DOWN"));
    if (getenv("GGML_S10_GPU_FRAC_OTHER"))  g_frac_other  = (float) atof(getenv("GGML_S10_GPU_FRAC_OTHER"));
    g_prof  = getenv("GGML_S10_GPU_PROF") != NULL;
    g_hprof = getenv("GGML_S10_GPU_HPROF") != NULL;
    g_fake  = getenv("GGML_S10_GPU_FAKE") != NULL;
    if (getenv("GGML_S10_GPU_WAIT") && !strcmp(getenv("GGML_S10_GPU_WAIT"), "read")) g_spin = 0;
    if (getenv("GGML_S10_GPU_SLOTS")) {
        g_slots = atoi(getenv("GGML_S10_GPU_SLOTS"));
        if (g_slots < 1 || g_slots > 4) g_slots = 0;
    }
    g_min_bytes = getenv("GGML_S10_GPU_MINMB") ? atof(getenv("GGML_S10_GPU_MINMB")) * 1048576.0 : 0.0;
    g_fuse  = !(getenv("GGML_S10_GPU_FUSE") && atoi(getenv("GGML_S10_GPU_FUSE")) == 0);
    g_xfer  = !(getenv("GGML_S10_GPU_XFER") && atoi(getenv("GGML_S10_GPU_XFER")) == 0);
    g_late = getenv("GGML_S10_GPU_LATE") && atoi(getenv("GGML_S10_GPU_LATE")) != 0;
    g_late = g_late && g_xfer;   // map failures must still be able to fall back before the barrier
#if defined(__ARM_NEON) && defined(__aarch64__)
    g_qneon = getenv("GGML_S10_GPU_QNEON") && atoi(getenv("GGML_S10_GPU_QNEON")) != 0;
    g_qcheck = getenv("GGML_S10_GPU_QCHECK") && atoi(getenv("GGML_S10_GPU_QCHECK")) != 0;
#endif
    if (!(g_frac > 0.0f && g_frac < 1.0f)) {
        return 0;
    }
    g_direct = getenv("GGML_S10_GPU_DIRECT") && atoi(getenv("GGML_S10_GPU_DIRECT")) != 0;
    if (g_direct) {
        if (s10d_init()) {
            g_xfer = 1; g_spin = 1; g_prof = 0;
            g_d_mm4 = !(getenv("GGML_S10_GPU_MM4") && atoi(getenv("GGML_S10_GPU_MM4")) == 0);
            g_d_mm2 = !(getenv("GGML_S10_GPU_MM2") && atoi(getenv("GGML_S10_GPU_MM2")) == 0);
            fprintf(stderr, "s10-gpu-assist: direct submission (no Rusticl), GPU takes %.0f%% of each PTQ1_0 mat-mul's rows%s\n",
                    g_frac * 100.0f, g_check ? " (checking results)" : "");
            return 1;
        }
        fprintf(stderr, "s10-gpu-assist: direct submission unavailable, using Rusticl\n");
        g_direct = 0;
    }

    // The OpenCL driver's worker threads inherit this thread's CPU mask. Create them on the cores in
    // GGML_S10_GPU_DRIVER_CPUS (default 0-3, the little cores) so they do not queue behind the compute threads.
    cpu_set_t saved, drv;
    const int have_mask = sched_getaffinity(0, sizeof(saved), &saved) == 0;
    {
        const char * m = getenv("GGML_S10_GPU_DRIVER_CPUS");
        CPU_ZERO(&drv);
        int lo = 0, hi = 3;
        if (m) { sscanf(m, "%d-%d", &lo, &hi); }
        for (int c = lo; c <= hi; c++) CPU_SET(c, &drv);
        if (have_mask) sched_setaffinity(0, sizeof(drv), &drv);
    }
    const int ok = s10ga_init_cl();
    if (have_mask) sched_setaffinity(0, sizeof(saved), &saved);
    return ok;
}

static int s10ga_init_cl(void) {
    cl_int err;
    cl_platform_id plat;
    cl_device_id dev;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS ||
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) {
        fprintf(stderr, "s10-gpu-assist: no OpenCL GPU (set RUSTICL_ENABLE=panfrost?), assist off\n");
        return 0;
    }
    g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { return 0; }
    const cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g_q = clCreateCommandQueueWithProperties(g_ctx, dev, getenv("GGML_S10_GPU_PROF") ? qprops : NULL, &err);
    if (err != CL_SUCCESS) { return 0; }

    const char * srcs[4] = { s10ga_kernel_src, s10ga_kernel_mm4_src, s10ga_kernel_mm8_src, s10ga_kernel_mm2_src };
    cl_program prog = clCreateProgramWithSource(g_ctx, 4, srcs, NULL, &err);
    if (err != CL_SUCCESS) { return 0; }
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192] = {0};
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "s10-gpu-assist: kernel build failed (%d), assist off:\n%s\n", err, log);
        return 0;
    }
    g_kern = clCreateKernel(prog, "s10ga_ptq1_mv", &err);
    if (err == CL_SUCCESS) {
        g_kern8 = clCreateKernel(prog, "s10ga_ptq1_mv8", &err);
    }
    if (err == CL_SUCCESS) {
        g_kern_tail = clCreateKernel(prog, "s10ga_ptq1_mv_tail", &err);
    }
    if (err == CL_SUCCESS && !(getenv("GGML_S10_GPU_MM4") && atoi(getenv("GGML_S10_GPU_MM4")) == 0)) {
        cl_int e4;
        if (getenv("GGML_S10_GPU_MMC") && atoi(getenv("GGML_S10_GPU_MMC")) == 8) { g_mmc = 8; }
        g_kern_mm4 = clCreateKernel(prog, g_mmc == 8 ? "s10ga_ptq1_mm8" : "s10ga_ptq1_mm4", &e4);   // optional; prompts fall back to mv
        if (e4 != CL_SUCCESS) { g_kern_mm4 = NULL; }
    }
    if (err == CL_SUCCESS && !(getenv("GGML_S10_GPU_MM2") && atoi(getenv("GGML_S10_GPU_MM2")) == 0)) {
        cl_int e2;
        g_kern_mm2 = clCreateKernel(prog, "s10ga_ptq1_mm2", &e2);   // optional
        if (e2 != CL_SUCCESS) { g_kern_mm2 = NULL; }
    }
    clReleaseProgram(prog);
    if (err != CL_SUCCESS) {
        if (g_kern) { clReleaseKernel(g_kern); g_kern = NULL; }
        if (g_kern8) { clReleaseKernel(g_kern8); g_kern8 = NULL; }
        return 0;
    }

    char name[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
    fprintf(stderr, "s10-gpu-assist: %s takes %.0f%% of each PTQ1_0 mat-mul's rows%s\n",
                  name, g_frac * 100.0f, g_check ? " (checking results)" : "");
    return 1;
}

static int s10ga_trit(unsigned q, unsigned p) {
    return (int) (((q * p) & 255) * 3 >> 8) - 1;
}

// PTQ1_0 block (qs[24] base-3 x5, qh[2] base-3 x4, fp16 d) -> 2-bit codes (trit + 1); weight k sits in
// byte k%32 at bit 2*(k/32). Element order as in dequantize_row_ptq1_0.
static void s10ga_encode(const uint8_t * src, int64_t rows, int64_t nb, uint8_t * dst) {
    static const unsigned pow3[5] = {1, 3, 9, 27, 81};
    uint8_t * codes  = dst;
    uint8_t * scales = dst + rows * nb * 32;
    for (int64_t i = 0; i < rows * nb; i++) {
        const uint8_t * b = src + i * 28;
        uint8_t * c = codes + i * 32;
        memset(c, 0, 32);
        for (int k = 0; k < 128; k++) {
            int idx, p;
            if (k < 80)       { idx = k % 16;              p = k / 16; }
            else if (k < 120) { idx = 16 + (k - 80) % 8;  p = (k - 80) / 8; }
            else              { idx = 24 + (k - 120) % 2; p = (k - 120) / 2; }
            const int code = s10ga_trit(b[idx], pow3[p]) + 1;
            c[k % 32] |= (uint8_t) (code << ((k / 32) * 2));
        }
        scales[i * 2 + 0] = b[26];
        scales[i * 2 + 1] = b[27];
    }
}

static struct s10ga_entry * s10ga_lookup(const struct ggml_tensor * src0) {
    size_t h = ((uintptr_t) src0->data >> 6) % S10GA_TABLE;
    for (int n = 0; n < S10GA_TABLE; n++, h = (h + 1) % S10GA_TABLE) {
        if (g_tab[h].data == src0->data) {
            return &g_tab[h];
        }
        if (g_tab[h].data == NULL) {
            struct s10ga_entry * e = &g_tab[h];
            e->data = src0->data;
            e->rows = 0;
            e->w    = NULL;
            e->partner = NULL;
            snprintf(e->name, sizeof(e->name), "%s", src0->name);
            const int64_t K = src0->ne[0], M = src0->ne[1], nb = K / 128;
            const int64_t rows = ((int64_t) (M * s10ga_frac_for(src0->name))) / S10GA_TILE_ROWS * S10GA_TILE_ROWS;
            if (rows < S10GA_TILE_ROWS || (double) ggml_nbytes(src0) < g_min_bytes) {
                return e;
            }
            if (g_fake && getenv("GGML_S10_GPU_FAKE_NOALLOC")) {
                e->rows = rows;
                return e;
            }
            const size_t bytes = (size_t) rows * nb * 34;
            if (g_direct) {
                // the direct path embeds only the 8-row mv kernel (s10ga_enqueue_mv picks 4 rows for a K tail or for
                // long K with many rows) and mm2/mm4 (no K tail)
                if (nb % 4 != 0 || (nb >= 128 && rows >= 4096)) {
                    fprintf(stderr, "s10-gpu-assist: %s needs the 4-row kernel (not in the direct path), CPU only\n", src0->name);
                    return e;
                }
                uint8_t * tmp = malloc(bytes);
                if (!tmp || !s10d_buf_alloc(&e->dw, bytes)) {
                    free(tmp);
                    fprintf(stderr, "s10-gpu-assist: %s: GPU buffer of %zu bytes failed, CPU only\n", src0->name, bytes);
                    return e;
                }
                s10ga_encode((const uint8_t *) src0->data, rows, nb, tmp);   // encode in RAM, one streaming copy into WC memory
                memcpy(e->dw.cpu, tmp, bytes);
                free(tmp);
            } else {
            cl_int err;
            cl_mem w = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "s10-gpu-assist: %s: GPU buffer of %zu bytes failed (%d), CPU only\n", src0->name, bytes, err);
                return e;
            }
            void * p = clEnqueueMapBuffer(g_q, w, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes, 0, NULL, NULL, &err);
            if (err != CL_SUCCESS) {
                clReleaseMemObject(w);
                return e;
            }
            s10ga_encode((const uint8_t *) src0->data, rows, nb, (uint8_t *) p);
            clEnqueueUnmapMemObject(g_q, w, p, 0, NULL, NULL);
            clFinish(g_q);
            e->w    = w;
            }
            e->rows = rows;
            g_gpu_bytes += bytes;
            // pair ffn_up and ffn_gate of the same layer: the one met first in the graph (llama.cpp computes
            // up before gate) gets the other as partner, so its call also queues the partner's GPU rows
            {
                const char * u = strstr(e->name, "ffn_up.weight");
                const char * g = strstr(e->name, "ffn_gate.weight");
                if (u || g) {
                    char other[GGML_MAX_NAME];
                    if (u) snprintf(other, sizeof(other), "%.*sffn_gate.weight", (int) (u - e->name), e->name);
                    else   snprintf(other, sizeof(other), "%.*sffn_up.weight", (int) (g - e->name), e->name);
                    for (int i = 0; i < S10GA_TABLE; i++) {
                        if (g_tab[i].data && g_tab[i].rows > 0 && !strcmp(g_tab[i].name, other)) {
                            g_tab[i].partner = e;
                            break;
                        }
                    }
                }
            }
            // The CPU never reads these rows again: drop their pages so the weights are not held twice in RAM
            // (with ~1.4 GB of GPU copies the phone runs out of free memory and reclaim slows the CPU threads).
            // For an mmapped model these are clean file pages, re-read from flash if ever touched.
            {
                const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
                const uintptr_t a  = ((uintptr_t) src0->data + pg - 1) & ~(pg - 1);
                const uintptr_t b  = ((uintptr_t) src0->data + (uintptr_t) (rows * src0->nb[1])) & ~(pg - 1);
                if (b > a && !getenv("GGML_S10_GPU_KEEP_PAGES")) {
#ifdef MADV_PAGEOUT
                    madvise((void *) a, b - a, MADV_PAGEOUT);
#else
                    madvise((void *) a, b - a, MADV_DONTNEED);
#endif
                }
            }
            if ((++g_ntensors % 64) == 0) {
                fprintf(stderr, "s10-gpu-assist: %d tensors on the GPU, %.2f GiB\n", g_ntensors, g_gpu_bytes / 1073741824.0);
            }
            return e;
        }
    }
    return NULL;
}

static int s10ga_reserve(cl_mem * b, size_t * cap, size_t need) {
    if (*cap >= need) {
        return 1;
    }
    if (*b) {
        clReleaseMemObject(*b);
    }
    cl_int err;
    *b = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, need, NULL, &err);
    *cap = err == CL_SUCCESS ? need : 0;
    if (err != CL_SUCCESS) {
        *b = NULL;
    }
    return err == CL_SUCCESS;
}

// Queue the mat-vec kernel(s) for e's GPU rows, writing to out; *kev = event of the last dispatch.
// Multi-column kernel (mm4/mm8/mm2: 4 rows x mmc columns per item, each weight word loaded once for all mmc columns).
// Returns the first column it did not cover (a multiple of mmc).
static int64_t s10ga_enqueue_mmc(struct s10ga_entry * e, cl_kernel k4, int mmc, int nbi, int Ri, int lanes,
                                 int64_t ncols, cl_mem out, cl_event * kev) {
    cl_int err;
    const int64_t R = e->rows;
    const int s4 = (R % 8 == 0) ? 2 : 1;                     // rows are 4-aligned; two row slots when possible
    const int tr4 = 4 * s4, wg4 = lanes * s4;
    clSetKernelArg(k4, 0, sizeof(cl_mem), &e->w);
    clSetKernelArg(k4, 1, sizeof(cl_mem), &g_bdx);
    clSetKernelArg(k4, 2, sizeof(cl_mem), &g_bqx);
    clSetKernelArg(k4, 3, sizeof(cl_mem), &g_bdx);
    clSetKernelArg(k4, 4, sizeof(cl_mem), &out);
    clSetKernelArg(k4, 5, sizeof(int), &nbi);
    clSetKernelArg(k4, 6, sizeof(int), &Ri);
    clSetKernelArg(k4, 7, sizeof(int), &lanes);
    clSetKernelArg(k4, 8, wg4 * 4 * mmc * sizeof(float), NULL);
    clSetKernelArg(k4, 11, sizeof(cl_mem), &g_bas);
    const int64_t rl4 = S10GA_MAX_ROWS / mmc / tr4 * tr4;  // a column group does mmc x the work of one column
    const int64_t rp4 = R < rl4 ? R : (rl4 > 0 ? rl4 : tr4);
    int64_t qp4 = S10GA_MAX_ROWS / mmc / rp4;
    if (qp4 < 1) qp4 = 1;
    const int64_t nq = ncols / mmc;
    for (int64_t q0 = 0; q0 < nq; q0 += qp4) {
        const int64_t nqq = nq - q0 < qp4 ? nq - q0 : qp4;
        for (int64_t r0 = 0; r0 < R; r0 += rp4) {
            const int64_t nr = R - r0 < rp4 ? R - r0 : rp4;
            const int group_base = (int) (r0 / tr4), col_base = (int) (q0 * mmc);
            clSetKernelArg(k4, 9, sizeof(int), &group_base);
            clSetKernelArg(k4, 10, sizeof(int), &col_base);
            const size_t global[2] = { (size_t) (nr / tr4) * wg4, (size_t) nqq };
            const size_t local[2]  = { wg4, 1 };
            if (*kev) { clReleaseEvent(*kev); *kev = NULL; }
            err = clEnqueueNDRangeKernel(g_q, k4, 2, NULL, global, local, 0, NULL, kev);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("s10-gpu-assist: mm%d enqueue failed (%d)\n", mmc, err);
                GGML_ABORT("s10-gpu-assist");
            }
        }
    }
    return nq * mmc;
}

static void s10ga_enqueue_mv(struct s10ga_entry * e, int64_t nb, int64_t ncols, cl_mem out, cl_event * kev) {
    cl_int err;
    const int64_t R = e->rows;
    const int nbi = (int) nb, Ri = (int) R, lanes = S10GA_LANES;
    // Long K with many rows favors four rows per item. Other measured shares favor eight.
    // Keep the final block batch uniform and bound every load on this Mesa path.
    const int tail = nb % 4 != 0;
    const int tile = tail || (nb >= 128 && R >= 4096) ? 4 : 8;
    int slots = g_slots ? g_slots : (tile == 4 ? 2 : (R >= 4096 ? 4 : 1));
    // A user override or an unaligned share must still cover every cached row.
    if (R % (tile * slots) != 0) slots = 1;
    const int tile_rows = tile * slots, wg = lanes * slots;
    // Prompts (4+ columns, no K tail): the mm4 kernel loads each weight word once for 4 columns; 2-3 columns
    // (speculative-decoding checks) use mm2 the same way when enabled.
    int64_t col0 = 0;
    if (g_kern_mm4 && !tail && ncols >= g_mmc) {
        col0 = s10ga_enqueue_mmc(e, g_kern_mm4, g_mmc, nbi, Ri, lanes, ncols, out, kev);
    } else if (g_kern_mm2 && !tail && ncols >= 2) {
        col0 = s10ga_enqueue_mmc(e, g_kern_mm2, 2, nbi, Ri, lanes, ncols, out, kev);
    }
    if (col0 == ncols) {
        return;
    }
    cl_kernel kernel = tail ? g_kern_tail : (tile == 4 ? g_kern : g_kern8);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &e->w);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_bdx);   // unused fp32 activations
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &g_bqx);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &g_bdx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 5, sizeof(int), &nbi);
    clSetKernelArg(kernel, 6, sizeof(int), &Ri);
    clSetKernelArg(kernel, 7, sizeof(int), &lanes);
    clSetKernelArg(kernel, 8, wg * tile * sizeof(float), NULL);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), &g_bas);

    // each dispatch covers at most S10GA_MAX_ROWS rows x columns
    const int64_t row_limit = S10GA_MAX_ROWS / tile_rows * tile_rows;
    const int64_t rows_per = R < row_limit ? R : row_limit;
    int64_t cols_per = S10GA_MAX_ROWS / rows_per;
    if (cols_per < 1) cols_per = 1;
    for (int64_t c0 = col0; c0 < ncols; c0 += cols_per) {
        const int64_t nc = ncols - c0 < cols_per ? ncols - c0 : cols_per;
        for (int64_t r0 = 0; r0 < R; r0 += rows_per) {
            const int64_t nr = R - r0 < rows_per ? R - r0 : rows_per;
            const int group_base = (int) (r0 / tile_rows), col_base = (int) c0;
            clSetKernelArg(kernel, 9, sizeof(int), &group_base);
            clSetKernelArg(kernel, 10, sizeof(int), &col_base);
            const size_t global[2] = { (size_t) (nr / tile_rows) * wg, (size_t) nc };
            const size_t local[2]  = { wg, 1 };
            if (*kev) { clReleaseEvent(*kev); *kev = NULL; }
            err = clEnqueueNDRangeKernel(g_q, kernel, 2, NULL, global, local, 0, NULL, kev);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("s10-gpu-assist: enqueue failed (%d)\n", err);
                GGML_ABORT("s10-gpu-assist");
            }
        }
    }
}

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

// Direct path: the same dispatch geometry as s10ga_enqueue_mmc / s10ga_enqueue_mv, queued with s10d_add.
static int64_t s10ga_direct_mmc(struct s10ga_entry * e, enum s10d_kernel k, int mmc, int nbi, int Ri, int lanes,
                                int64_t ncols, const struct s10d_buf * out) {
    const int64_t R = e->rows;
    const int s4 = (R % 8 == 0) ? 2 : 1, tr4 = 4 * s4, wg4 = lanes * s4;
    const int64_t rl4 = S10GA_MAX_ROWS / mmc / tr4 * tr4;
    const int64_t rp4 = R < rl4 ? R : (rl4 > 0 ? rl4 : tr4);
    int64_t qp4 = S10GA_MAX_ROWS / mmc / rp4;
    if (qp4 < 1) qp4 = 1;
    const int64_t nq = ncols / mmc;
    for (int64_t q0 = 0; q0 < nq; q0 += qp4) {
        const int64_t nqq = nq - q0 < qp4 ? nq - q0 : qp4;
        for (int64_t r0 = 0; r0 < R; r0 += rp4) {
            const int64_t nr = R - r0 < rp4 ? R - r0 : rp4;
            if (!s10d_add(k, &e->dw, &g_dqx, &g_ddx, out, &g_das, nbi, Ri, wg4, wg4 * 4 * mmc * (int) sizeof(float),
                          (int) (nr / tr4), (int) nqq, (int) (r0 / tr4), (int) (q0 * mmc))) {
                GGML_ABORT("s10-gpu-assist: direct mm%d dispatch failed", mmc);
            }
        }
    }
    return nq * mmc;
}

static void s10ga_direct_mv(struct s10ga_entry * e, int64_t nb, int64_t ncols, const struct s10d_buf * out) {
    const int64_t R = e->rows;
    const int nbi = (int) nb, Ri = (int) R, lanes = S10GA_LANES;
    const int tile = 8;                      // lookup keeps tensors that need the 4-row kernel off the direct path
    int slots = g_slots ? g_slots : (R >= 4096 ? 4 : 1);
    if (R % (tile * slots) != 0) slots = 1;
    const int tile_rows = tile * slots, wg = lanes * slots;
    int64_t col0 = 0;
    if (g_d_mm4 && ncols >= 4) {
        col0 = s10ga_direct_mmc(e, S10D_MM4, 4, nbi, Ri, lanes, ncols, out);
    } else if (g_d_mm2 && ncols >= 2) {
        col0 = s10ga_direct_mmc(e, S10D_MM2, 2, nbi, Ri, lanes, ncols, out);
    }
    if (col0 == ncols) {
        return;
    }
    const int64_t row_limit = S10GA_MAX_ROWS / tile_rows * tile_rows;
    const int64_t rows_per = R < row_limit ? R : row_limit;
    int64_t cols_per = S10GA_MAX_ROWS / rows_per;
    if (cols_per < 1) cols_per = 1;
    for (int64_t c0 = col0; c0 < ncols; c0 += cols_per) {
        const int64_t nc = ncols - c0 < cols_per ? ncols - c0 : cols_per;
        for (int64_t r0 = 0; r0 < R; r0 += rows_per) {
            const int64_t nr = R - r0 < rows_per ? R - r0 : rows_per;
            if (!s10d_add(S10D_MV8, &e->dw, &g_dqx, &g_ddx, out, &g_das, nbi, Ri, wg, wg * tile * (int) sizeof(float),
                          (int) (nr / tile_rows), (int) nc, (int) (r0 / tile_rows), (int) c0)) {
                GGML_ABORT("s10-gpu-assist: direct mv dispatch failed");
            }
        }
    }
}

static int64_t s10ga_submit_direct(struct s10ga_entry * e, const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    const int64_t K = src0->ne[0], nb = K / 128, R = e->rows, ncols = src1->ne[1];
    if (s10d_pending() && !s10d_wait()) {    // the inputs below are shared with any chain still running
        GGML_ABORT("s10-gpu-assist: direct GPU job failed");
    }
    int8_t * qx = g_hqx = s10ga_host(g_hqx, &g_hcap_qx, (size_t) ncols * K);
    float  * dx = g_hdx = s10ga_host(g_hdx, &g_hcap_dx, (size_t) ncols * nb * sizeof(float));
    g_has = s10ga_host(g_has, &g_hcap_as, (size_t) ncols * nb * sizeof(int32_t));
    for (int64_t c = 0; c < ncols; c++) {
        const float * x = (const float *) ((const char *) src1->data + c * src1->nb[1]);
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (g_qneon) {
            s10ga_quant128_neon(x, nb, qx + c * K, dx + c * nb, g_has + c * nb);
        } else
#endif
        {
            s10ga_quant128_scalar(x, nb, qx + c * K, dx + c * nb, g_has + c * nb);
        }
    }
    memcpy(g_dqx.cpu, qx, (size_t) ncols * K);
    memcpy(g_ddx.cpu, dx, (size_t) ncols * nb * sizeof(float));
    memcpy(g_das.cpu, g_has, (size_t) ncols * nb * sizeof(int32_t));
    s10ga_direct_mv(e, nb, ncols, &g_dy);
    if (g_fuse && e->partner && e->partner->rows > 0 &&
        s10d_buf_reserve(&g_dy2, (size_t) ncols * e->partner->rows * sizeof(float))) {
        s10ga_direct_mv(e->partner, nb, ncols, &g_dy2);   // same activations: the partner's rows go in the same chain
        g_pend = e->partner; g_pend_x = src1->data; g_pend_ncols = ncols;
    }
    if (!s10d_submit()) {
        GGML_ABORT("s10-gpu-assist: direct submit failed");
    }
    return R;
}

static int64_t s10ga_submit_impl(struct s10ga_entry * e, const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    if (g_direct) {
        return s10ga_submit_direct(e, src0, src1);
    }
    const int64_t K = src0->ne[0], nb = K / 128, R = e->rows, ncols = src1->ne[1];
    cl_int err;
    const double t0 = s10ga_now();
    int8_t * qx; float * dx;
    if (g_xfer) {
        qx = g_hqx = s10ga_host(g_hqx, &g_hcap_qx, (size_t) ncols * K);
        dx = g_hdx = s10ga_host(g_hdx, &g_hcap_dx, (size_t) ncols * nb * sizeof(float));
    } else {
        qx = clEnqueueMapBuffer(g_q, g_bqx, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t) ncols * K, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) { return 0; }
        dx = clEnqueueMapBuffer(g_q, g_bdx, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, (size_t) ncols * nb * sizeof(float), 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) { clEnqueueUnmapMemObject(g_q, g_bqx, qx, 0, NULL, NULL); return 0; }
    }
    g_has = s10ga_host(g_has, &g_hcap_as, (size_t) ncols * nb * sizeof(int32_t));
#if defined(__ARM_NEON) && defined(__aarch64__)
    static int checked;
    static int8_t * check_qx;
    static float * check_dx;
    static int32_t * check_as;
    static size_t cap_qx, cap_dx, cap_as;
    const int qcheck = g_qcheck && checked < 2000;
    if (qcheck) {
        checked++;
        check_qx = s10ga_host(check_qx, &cap_qx, (size_t) K);
        check_dx = s10ga_host(check_dx, &cap_dx, (size_t) nb * sizeof(float));
        check_as = s10ga_host(check_as, &cap_as, (size_t) nb * sizeof(int32_t));
        GGML_ASSERT(check_qx && check_dx && check_as);
    }
#endif
    const double t1 = s10ga_now();
    for (int64_t c = 0; c < ncols; c++) {
        const float * x = (const float *) ((const char *) src1->data + c * src1->nb[1]);
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (g_qneon) {
            s10ga_quant128_neon(x, nb, qx + c * K, dx + c * nb, g_has + c * nb);
        } else
#endif
        {
            s10ga_quant128_scalar(x, nb, qx + c * K, dx + c * nb, g_has + c * nb);
        }
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (qcheck) {
            if (g_qneon) s10ga_quant128_scalar(x, nb, check_qx, check_dx, check_as);
            else         s10ga_quant128_neon(x, nb, check_qx, check_dx, check_as);
            if (memcmp(qx + c * K, check_qx, (size_t) K) ||
                memcmp(dx + c * nb, check_dx, (size_t) nb * sizeof(float)) ||
                memcmp(g_has + c * nb, check_as, (size_t) nb * sizeof(int32_t))) {
                GGML_ABORT("s10-gpu-assist: QCHECK mismatch for %s", src0->name);
            }
        }
#endif
    }
    const double t2 = s10ga_now();
    if (g_xfer) {
        clEnqueueWriteBuffer(g_q, g_bqx, CL_FALSE, 0, (size_t) ncols * K, qx, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_q, g_bdx, CL_FALSE, 0, (size_t) ncols * nb * sizeof(float), dx, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_q, g_bas, CL_FALSE, 0, (size_t) ncols * nb * sizeof(int32_t), g_has, 0, NULL, NULL);
    } else {
        clEnqueueUnmapMemObject(g_q, g_bqx, qx, 0, NULL, NULL);
        clEnqueueUnmapMemObject(g_q, g_bdx, dx, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_q, g_bas, CL_FALSE, 0, (size_t) ncols * nb * sizeof(int32_t), g_has, 0, NULL, NULL);
    }
    const double t3 = s10ga_now();

    s10ga_enqueue_mv(e, nb, ncols, g_by, &g_last_ev);
    if (g_spin && g_hprof) { g_t_done = 0; clSetEventCallback(g_last_ev, CL_COMPLETE, s10ga_done_cb, NULL); }
    if (g_xfer) {
        g_hy = s10ga_host(g_hy, &g_hcap_y, (size_t) ncols * R * sizeof(float));
        err = clEnqueueReadBuffer(g_q, g_by, CL_FALSE, 0, (size_t) ncols * R * sizeof(float), g_hy, 0, NULL, &g_read_ev);
        GGML_ASSERT(err == CL_SUCCESS);
        if (g_hprof && !g_spin) { g_t_done = 0; clSetEventCallback(g_read_ev, CL_COMPLETE, s10ga_done_cb, NULL); }
    }
    if (g_fuse && g_spin && g_xfer && e->partner && e->partner->rows > 0) {
        struct s10ga_entry * p = e->partner;
        const size_t ybytes = (size_t) ncols * p->rows * sizeof(float);
        if (s10ga_reserve(&g_by2, &g_cap_y2, ybytes)) {
            g_hy2 = s10ga_host(g_hy2, &g_hcap_y2, ybytes);
            clFlush(g_q);   // submit this call's own kernel + read before queueing the partner
            g_pend_kev = NULL;
            s10ga_enqueue_mv(p, nb, ncols, g_by2, &g_pend_kev);
            err = clEnqueueReadBuffer(g_q, g_by2, CL_FALSE, 0, ybytes, g_hy2, 0, NULL, &g_pend_rev);
            GGML_ASSERT(err == CL_SUCCESS);
            g_pend = p; g_pend_x = src1->data; g_pend_ncols = ncols;
            if (++g_n_queued % 200 == 0 && getenv("GGML_S10_GPU_FUSEDBG")) fprintf(stderr, "s10-gpu-assist fuse: queued %ld used %ld dropped %ld\n", g_n_queued, g_n_used, g_n_dropped);
        }
    }
    clFlush(g_q);
    g_t_flush = s10ga_now();
    const double t4 = s10ga_now();
    g_t[0] += t1 - t0; g_t[1] += t2 - t1; g_t[2] += t3 - t2; g_t[3] += t4 - t3; g_tbegin_end = t4;
    return R;
}

int64_t ggml_s10ga_begin(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    const double tplan = g_hprof || g_state < 0 ? s10ga_now() : 0.0;
    // checked before init, so a model without PTQ1_0 weights never opens an OpenCL context
    if (src0->type != GGML_TYPE_PTQ1_0) {
        return 0;
    }
    if (g_state < 0) {
        g_state = s10ga_init();
    }
    if (g_state == 0 || src1->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src0->ne[0] % 128 != 0) {
        return 0;
    }
    struct s10ga_entry * e = s10ga_lookup(src0);
    if (!e || e->rows == 0) {
        return 0;
    }

    const int64_t K = src0->ne[0], nb = K / 128, R = e->rows, ncols = src1->ne[1];
    if (g_fake) { g_t_flush = s10ga_now(); return R; }
    g_cur_is_pend = 0;
    if (g_pend) {
        if (g_pend == e && g_pend_x == src1->data && g_pend_ncols == ncols) {
            g_cur_is_pend = 1;           // queued by the gate call; end() collects it
            g_n_used++;
            if (g_hprof) { g_h_plan += s10ga_now() - tplan; g_h_collect++; }
            return R;
        }
        g_n_dropped++;
        if (getenv("GGML_S10_GPU_FUSEDBG") && g_n_dropped < 5) fprintf(stderr, "s10-gpu-assist fuse drop: pending %s x=%p n=%lld, got %s x=%p n=%lld\n", g_pend->name, g_pend_x, (long long) g_pend_ncols, e->name, src1->data, (long long) ncols);
        if (g_direct) {
            s10d_wait();                      // not consumed: let it finish and drop it
        } else {
        clWaitForEvents(1, &g_pend_rev);  // not consumed: let it finish and drop it
        clReleaseEvent(g_pend_kev); clReleaseEvent(g_pend_rev);
        }
        g_pend = NULL;
    }
    if (g_direct) {
        if (!s10d_buf_reserve(&g_dqx, (size_t) ncols * K) ||
            !s10d_buf_reserve(&g_ddx, (size_t) ncols * nb * sizeof(float)) ||
            !s10d_buf_reserve(&g_das, (size_t) ncols * nb * sizeof(int32_t)) ||
            !s10d_buf_reserve(&g_dy,  (size_t) ncols * R * sizeof(float))) {
            return 0;
        }
    } else
    if (!s10ga_reserve(&g_bqx, &g_cap_qx, (size_t) ncols * K) ||
        !s10ga_reserve(&g_bdx, &g_cap_dx, (size_t) ncols * nb * sizeof(float)) ||
        !s10ga_reserve(&g_bas, &g_cap_as, (size_t) ncols * nb * sizeof(int32_t)) ||
        !s10ga_reserve(&g_by,  &g_cap_y,  (size_t) ncols * R * sizeof(float))) {
        return 0;
    }
    const double plan_time = g_hprof ? s10ga_now() - tplan : 0.0;
    if (!g_late) {
        const int64_t rows = s10ga_submit_impl(e, src0, src1);
        if (g_hprof && rows) g_h_plan += plan_time;
        return rows;
    }
    if (g_hprof) g_h_plan += plan_time;
    g_def_e = e; g_def_src0 = src0; g_def_src1 = src1; g_deferred = 1;
    return R;
}

void ggml_s10ga_submit(void) {
    if (!g_deferred) return;
    g_deferred = 0;
    // Late mode requires XFER=1, so no map failure can change the published row count.
    const int64_t rows = s10ga_submit_impl(g_def_e, g_def_src0, g_def_src1);
    GGML_ASSERT(rows == g_def_e->rows);
}

static void s10ga_check(const struct ggml_tensor * dst, int64_t R) {
    static int calls = 0;
    if (calls++ >= 64) {
        return;
    }
    const struct ggml_tensor * src0 = dst->src[0], * src1 = dst->src[1];
    const int64_t K = src0->ne[0];
    ggml_to_float_t deq = ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float;
    float * row = malloc(K * sizeof(float));
    // every column up to 4 (the multi-column kernels write columns 1-3 too)
    const int64_t ncols = src1->ne[1] < 4 ? src1->ne[1] : 4;
    double se = 0, ss = 0, worst = 0;
    for (int64_t c = 0; c < ncols; c++) {
        const float * x = (const float *) ((const char *) src1->data + c * src1->nb[1]);
        const float * y = (const float *) ((const char *) dst->data + c * dst->nb[1]);
        double sec = 0, ssc = 0;
        for (int64_t r = 0; r < R; r += 7) {
            deq((const char *) src0->data + r * src0->nb[1], row, K);
            double ref = 0;
            for (int64_t k = 0; k < K; k++) ref += (double) row[k] * x[k];
            sec += (y[r] - ref) * (y[r] - ref);
            ssc += ref * ref;
        }
        if (ssc > 0 && sqrt(sec / ssc) > worst) worst = sqrt(sec / ssc);
        se += sec; ss += ssc;
    }
    free(row);
    fprintf(stderr, "s10-gpu-assist check: %-28s rows %6lld/%6lld  cols %lld  nrmse %.2e (worst column %.2e)\n",
                  src0->name, (long long) R, (long long) src0->ne[1], (long long) ncols, ss > 0 ? sqrt(se / ss) : 0.0, worst);
}

void ggml_s10ga_end(struct ggml_tensor * dst, int64_t R) {
    if (g_fake) {
        g_h_cpu += s10ga_now() - g_t_flush;
        if (++g_h_n % 400 == 0) fprintf(stderr, "s10-gpu-assist fake: flush->cpu done %.3f ms/call\n", g_h_cpu / g_h_n * 1e3);
        return;
    }
    const int64_t ncols = dst->ne[1];
    if (g_direct) {
        const uint8_t * y;
        if (g_cur_is_pend) {                 // computed in the previous call's chain, which its end() waited for
            y = g_dy2.cpu;
            g_pend = NULL; g_cur_is_pend = 0;
        } else {
            if (!s10d_wait()) {
                GGML_ABORT("s10-gpu-assist: direct GPU job failed");
            }
            y = g_dy.cpu;
        }
        for (int64_t c = 0; c < ncols; c++) {
            memcpy((char *) dst->data + c * dst->nb[1], y + (size_t) c * R * sizeof(float), R * sizeof(float));
        }
        if (g_check) {
            s10ga_check(dst, R);
        }
        return;
    }
    cl_int err;
    if (g_cur_is_pend) {
        cl_int st = CL_QUEUED;
        do {
            err = clGetEventInfo(g_pend_kev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
            GGML_ASSERT(err == CL_SUCCESS && st >= 0);
        } while (st != CL_COMPLETE);
        do {
            err = clGetEventInfo(g_pend_rev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
            GGML_ASSERT(err == CL_SUCCESS && st >= 0);
        } while (st != CL_COMPLETE);
        for (int64_t c = 0; c < ncols; c++) {
            memcpy((char *) dst->data + c * dst->nb[1], g_hy2 + c * R, R * sizeof(float));
        }
        clReleaseEvent(g_pend_kev); clReleaseEvent(g_pend_rev);
        g_pend = NULL; g_cur_is_pend = 0;
        if (g_check) {
            s10ga_check(dst, R);
        }
        return;
    }
    const double t0 = s10ga_now();
    g_tcpu += t0 - g_tbegin_end;
    const float * y;
    if (g_spin) {
        cl_int st = CL_QUEUED;
        do {
            err = clGetEventInfo(g_last_ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
            GGML_ASSERT(err == CL_SUCCESS && st >= 0);
        } while (st != CL_COMPLETE);
    }
    const double tk = s10ga_now();
    if (g_xfer) {
        // spin on the read's status: a blocking wait parks this thread and adds a slow wake-up through the
        // driver's queue thread, and lets schedutil lower the core clock
        cl_int st = CL_QUEUED;
        do {
            err = clGetEventInfo(g_read_ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, NULL);
            GGML_ASSERT(err == CL_SUCCESS && st >= 0);
        } while (st != CL_COMPLETE);
        if (g_hprof) {
            const double tw = s10ga_now();
            while (g_t_done == 0) { }
            g_h_lat += g_t_done - g_t_flush; g_h_cpu += t0 - g_t_flush; g_h_wait += tw - (g_spin ? tk : t0);
            g_h_cpu_wait += fmax(0.0, g_t_done - t0);
            g_h_gpu_idle += fmax(0.0, t0 - g_t_done);
            if (g_h_n % 400 == 0) g_h_t0cpu = sched_getcpu();
            if (++g_h_n % 400 == 0) {
                fprintf(stderr, "s10-gpu-assist hprof (ms/call over %ld): flush->%s complete %.3f  flush->cpu done %.3f  %s->read seen %.3f | plan %.3f quant %.3f write %.3f enqueue %.3f | cpu-waits %.3f gpu-idle %.3f | submits %ld of %ld late %d qneon %d t0cpu %d\n",
                        g_h_n + g_h_collect, g_spin ? "kernel" : "read", g_h_lat / g_h_n * 1e3, g_h_cpu / g_h_n * 1e3, g_spin ? "kernel seen" : "cpu done", g_h_wait / g_h_n * 1e3,
                        g_h_plan / (g_h_n + g_h_collect) * 1e3, g_t[1] / g_h_n * 1e3, g_t[2] / g_h_n * 1e3, g_t[3] / g_h_n * 1e3,
                        g_h_cpu_wait / g_h_n * 1e3, g_h_gpu_idle / g_h_n * 1e3,
                        g_h_n, g_h_n + g_h_collect, g_late, g_qneon, g_h_t0cpu);
            }
        }
        clReleaseEvent(g_read_ev);
        g_read_ev = NULL;
        y = g_hy;
    } else {
        clFinish(g_q);
        y = clEnqueueMapBuffer(g_q, g_by, CL_TRUE, CL_MAP_READ, 0, (size_t) ncols * R * sizeof(float), 0, NULL, NULL, &err);
        GGML_ASSERT(err == CL_SUCCESS);
    }
    const double t1 = s10ga_now();
    for (int64_t c = 0; c < ncols; c++) {
        memcpy((char *) dst->data + c * dst->nb[1], y + c * R, R * sizeof(float));
    }
    if (!g_xfer && !g_spin) {
        clEnqueueUnmapMemObject(g_q, g_by, (void *) y, 0, NULL, NULL);
    }
    g_t[4] += t1 - t0; g_t[5] += s10ga_now() - t1;
    if (g_prof && g_last_ev) {
        cl_ulong a = 0, b = 0, qd = 0, sb = 0;
        clGetEventProfilingInfo(g_last_ev, CL_PROFILING_COMMAND_QUEUED, sizeof qd, &qd, NULL);
        clGetEventProfilingInfo(g_last_ev, CL_PROFILING_COMMAND_SUBMIT, sizeof sb, &sb, NULL);
        clGetEventProfilingInfo(g_last_ev, CL_PROFILING_COMMAND_START, sizeof a, &a, NULL);
        clGetEventProfilingInfo(g_last_ev, CL_PROFILING_COMMAND_END, sizeof b, &b, NULL);
        g_tgpu += (b - a) * 1e-9; g_tq2sub += (sb - qd) * 1e-9; g_tsub2start += (a - sb) * 1e-9;
    }
    if (g_prof && ++g_calls % 400 == 0) {
        fprintf(stderr, "s10-gpu-assist prof (ms/call over %ld): cpu-share %.3f gpu-kernel(last dispatch) %.3f queued->submit %.3f submit->start %.3f | map %.3f quant %.3f unmap/write %.3f enqueue %.3f wait+read %.3f copy %.3f\n",
                g_calls, g_tcpu/g_calls*1e3, g_tgpu/g_calls*1e3, g_tq2sub/g_calls*1e3, g_tsub2start/g_calls*1e3, g_t[0]/g_calls*1e3, g_t[1]/g_calls*1e3, g_t[2]/g_calls*1e3, g_t[3]/g_calls*1e3, g_t[4]/g_calls*1e3, g_t[5]/g_calls*1e3);
    }
    if (g_check) {
        s10ga_check(dst, R);
    }
}
