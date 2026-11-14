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
#include "ggml-impl.h"

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
};

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
static double     g_min_bytes;   // GGML_S10_GPU_MINMB: only tensors at least this big (MiB) get GPU rows
static cl_context g_ctx;
static cl_command_queue g_q;
static cl_kernel  g_kern, g_kern8, g_kern_tail;
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
    if (!(g_frac > 0.0f && g_frac < 1.0f)) {
        return 0;
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

    cl_program prog = clCreateProgramWithSource(g_ctx, 1, &s10ga_kernel_src, NULL, &err);
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
            const int64_t rows = ((int64_t) (M * g_frac)) / S10GA_TILE_ROWS * S10GA_TILE_ROWS;
            if (rows < S10GA_TILE_ROWS || (double) ggml_nbytes(src0) < g_min_bytes) {
                return e;
            }
            if (g_fake && getenv("GGML_S10_GPU_FAKE_NOALLOC")) {
                e->rows = rows;
                return e;
            }
            const size_t bytes = (size_t) rows * nb * 34;
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
    for (int64_t c0 = 0; c0 < ncols; c0 += cols_per) {
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

int64_t ggml_s10ga_begin(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
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
            return R;
        }
        g_n_dropped++;
        if (getenv("GGML_S10_GPU_FUSEDBG") && g_n_dropped < 5) fprintf(stderr, "s10-gpu-assist fuse drop: pending %s x=%p n=%lld, got %s x=%p n=%lld\n", g_pend->name, g_pend_x, (long long) g_pend_ncols, e->name, src1->data, (long long) ncols);
        clWaitForEvents(1, &g_pend_rev);  // not consumed: let it finish and drop it
        clReleaseEvent(g_pend_kev); clReleaseEvent(g_pend_rev);
        g_pend = NULL;
    }
    if (!s10ga_reserve(&g_bqx, &g_cap_qx, (size_t) ncols * K) ||
        !s10ga_reserve(&g_bdx, &g_cap_dx, (size_t) ncols * nb * sizeof(float)) ||
        !s10ga_reserve(&g_bas, &g_cap_as, (size_t) ncols * nb * sizeof(int32_t)) ||
        !s10ga_reserve(&g_by,  &g_cap_y,  (size_t) ncols * R * sizeof(float))) {
        return 0;
    }


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
    const double t1 = s10ga_now();
    for (int64_t c = 0; c < ncols; c++) {
        const float * x = (const float *) ((const char *) src1->data + c * src1->nb[1]);
        for (int64_t b = 0; b < nb; b++) {
            float amax = 0.0f;
            for (int j = 0; j < 128; j++) {
                amax = fmaxf(amax, fabsf(x[b * 128 + j]));
            }
            const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            dx[c * nb + b] = d;
            int32_t sum = 0;
            for (int j = 0; j < 128; j++) {
                const int8_t q = (int8_t) lrintf(x[b * 128 + j] * id);
                qx[c * K + b * 128 + j] = q;
                sum += q;
            }
            g_has[c * nb + b] = sum;
        }
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

static void s10ga_check(const struct ggml_tensor * dst, int64_t R) {
    static int calls = 0;
    if (calls++ >= 64) {
        return;
    }
    const struct ggml_tensor * src0 = dst->src[0], * src1 = dst->src[1];
    const int64_t K = src0->ne[0];
    ggml_to_float_t deq = ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float;
    float * row = malloc(K * sizeof(float));
    const float * x = (const float *) src1->data;
    const float * y = (const float *) dst->data;
    double se = 0, ss = 0;
    for (int64_t r = 0; r < R; r += 7) {
        deq((const char *) src0->data + r * src0->nb[1], row, K);
        double ref = 0;
        for (int64_t k = 0; k < K; k++) ref += (double) row[k] * x[k];
        se += (y[r] - ref) * (y[r] - ref);
        ss += ref * ref;
    }
    free(row);
    fprintf(stderr, "s10-gpu-assist check: %-28s rows %6lld/%6lld  nrmse %.2e\n", src0->name,
                  (long long) R, (long long) src0->ne[1], ss > 0 ? sqrt(se / ss) : 0.0);
}

void ggml_s10ga_end(struct ggml_tensor * dst, int64_t R) {
    if (g_fake) {
        g_h_cpu += s10ga_now() - g_t_flush;
        if (++g_h_n % 400 == 0) fprintf(stderr, "s10-gpu-assist fake: flush->cpu done %.3f ms/call\n", g_h_cpu / g_h_n * 1e3);
        return;
    }
    const int64_t ncols = dst->ne[1];
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
            if (++g_h_n % 400 == 0) {
                fprintf(stderr, "s10-gpu-assist hprof (ms/call over %ld): flush->%s complete %.3f  flush->cpu done %.3f  %s->read seen %.3f\n",
                        g_h_n, g_spin ? "kernel" : "read", g_h_lat / g_h_n * 1e3, g_h_cpu / g_h_n * 1e3, g_spin ? "kernel seen" : "cpu done", g_h_wait / g_h_n * 1e3);
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
