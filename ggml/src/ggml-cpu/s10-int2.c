// S10 spec C: re-encode CPU PTQ1_0 rows to 2-bit codes and run them through the s2 kernels.
// See s10-int2.h. Real path: aarch64 + dotprod; stubs elsewhere.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "s10-int2.h"
#include "ggml-impl.h"
#include "quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#define S10I2_TABLE 4096

static struct ggml_s10i2 g_tab[S10I2_TABLE];
static int     g_state = -1;   // -1 not init, 0 off, 1 on
static char    g_sel[512];
static int     g_sel_all;
static int     g_check;
static int     g_keep_pages;
static long    g_max_mb;       // 0 = no cap
static double  g_min_bytes;
static int     g_pf_pairs;     // 0 = off, N = N row pairs ahead
static size_t  g_total_bytes;  // codes + scales allocated
static size_t  g_over_bytes;   // +6 B per block vs PTQ1_0
static int     g_ntensors;
static int     g_check_n;      // kernel bit-checks done (first 64)
static int     g_declined_logged;
static int     g_cap_logged;

// Optional shared pool (GGML_S10_INT2_POOL + _POOL_MB): bump-allocate 64-byte aligned, then anonymous.
static uint8_t * g_pool;
static size_t    g_pool_size;
static size_t    g_pool_used;
static int       g_pool_exhausted;

static size_t s10i2_align64(size_t n) {
    return (n + 63) & ~(size_t) 63;
}

static void * s10i2_alloc(size_t n) {
    n = s10i2_align64(n);
    if (g_pool && g_pool_used + n <= g_pool_size) {
        void * p = g_pool + g_pool_used;
        g_pool_used += n;
        return p;
    }
    if (g_pool && !g_pool_exhausted) {
        g_pool_exhausted = 1;
        fprintf(stderr, "s10-int2: pool exhausted, using anonymous memory\n");
    }
    void * p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

static void s10i2_log_stats(void) {
    fprintf(stderr, "s10-int2: %d tensors, %.2f GiB of codes (+%.0f MiB over PTQ1_0)\n",
            g_ntensors, g_total_bytes / 1073741824.0, g_over_bytes / 1048576.0);
}

static int s10i2_name_match(const char * name) {
    if (g_sel_all) {
        return 1;
    }
    const char * p = g_sel;
    while (*p) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        if (!*p) {
            break;
        }
        const char * start = p;
        while (*p && *p != ',') {
            p++;
        }
        size_t len = (size_t) (p - start);
        while (len > 0 && start[len - 1] == ' ') {
            len--;
        }
        if (len == 0) {
            continue;
        }
        if (len == 3 && strncmp(start, "all", 3) == 0) {
            return 1;
        }
        if (len == 4 && strncmp(start, "head", 4) == 0) {
            if (strcmp(name, "output.weight") == 0) {
                return 1;
            }
            continue;
        }
        char tok[256];
        if (len >= sizeof(tok)) {
            len = sizeof(tok) - 1;
        }
        memcpy(tok, start, len);
        tok[len] = 0;
        if (strstr(name, tok)) {
            return 1;
        }
    }
    return 0;
}

static void s10i2_init_pool(void) {
    const char * path = getenv("GGML_S10_INT2_POOL");
    if (!path || !path[0]) {
        return;
    }
    const char * mb_s = getenv("GGML_S10_INT2_POOL_MB");
    if (!mb_s || atol(mb_s) <= 0) {
        fprintf(stderr, "s10-int2: GGML_S10_INT2_POOL set but GGML_S10_INT2_POOL_MB missing/invalid, pool off\n");
        return;
    }
    const size_t bytes = (size_t) atol(mb_s) * 1048576ull;
    const int fd = open(path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "s10-int2: open pool %s failed, pool off\n", path);
        return;
    }
    void * p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "s10-int2: mmap pool %s (%zu bytes) failed, pool off\n", path, bytes);
        return;
    }
    g_pool = (uint8_t *) p;
    g_pool_size = bytes;
    g_pool_used = 0;
    fprintf(stderr, "s10-int2: pool %s %.0f MiB\n", path, bytes / 1048576.0);
}

static int s10i2_init(void) {
    const char * s = getenv("GGML_S10_INT2");
    if (!s || !s[0]) {
        return 0;
    }
    snprintf(g_sel, sizeof(g_sel), "%s", s);
    g_sel_all = strcmp(g_sel, "all") == 0;
    g_check = getenv("GGML_S10_INT2_CHECK") != NULL;
    g_keep_pages = getenv("GGML_S10_INT2_KEEP_PAGES") != NULL;
    g_max_mb = getenv("GGML_S10_INT2_MAXMB") ? atol(getenv("GGML_S10_INT2_MAXMB")) : 0;
    g_min_bytes = getenv("GGML_S10_INT2_MINMB") ? atof(getenv("GGML_S10_INT2_MINMB")) * 1048576.0 : 6.0 * 1048576.0;
    {
        const char * p = getenv("GGML_S10_INT2_PF");
        g_pf_pairs = p ? atoi(p) : 1;
        if (g_pf_pairs < 0) {
            g_pf_pairs = 0;
        }
    }
    s10i2_init_pool();
    return 1;
}

static void s10i2_pageout(const struct ggml_tensor * src0, int64_t row0, int64_t rows) {
    if (g_keep_pages || rows <= 0) {
        return;
    }
    const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t start = (uintptr_t) src0->data + (uintptr_t) row0 * src0->nb[1];
    const uintptr_t end   = (uintptr_t) src0->data + (uintptr_t) (row0 + rows) * src0->nb[1];
    const uintptr_t a = (start + pg - 1) & ~(pg - 1);
    const uintptr_t b = end & ~(pg - 1);
    if (b > a) {
#ifdef MADV_PAGEOUT
        madvise((void *) a, b - a, MADV_PAGEOUT);
#else
        madvise((void *) a, b - a, MADV_DONTNEED);
#endif
    }
}

static void s10i2_check_rows(const struct ggml_tensor * src0, const struct ggml_tensor * src1,
                             const struct ggml_s10i2 * e) {
    const int n = (int) src0->ne[0];
    const int nb = e->nb;
    // heap, not stack: this runs on the compute thread (musl threads get 128 KiB of stack)
    int8_t * q0 = malloc(32768), * q1 = malloc(32768);
    float * d0 = malloc(256 * sizeof(float)), * d1 = malloc(256 * sizeof(float));
    uint8_t * ybuf = malloc((32768 / QK8_0) * sizeof(block_q8_0));
    float * tmpf = malloc(32768 * sizeof(float));
    GGML_ASSERT(q0 && q1 && d0 && d1 && ybuf && tmpf);
    for (int64_t r = 0; r < e->rows; ++r) {
        const void * ptq = (const char *) src0->data + (e->row0 + r) * src0->nb[1];
        const uint8_t * codes = e->codes + r * nb * 32;
        const ggml_half * scales = (const ggml_half *) (e->scales + r * nb);
        ggml_ptq1_0_decode_row_s10(n, ptq, q0, d0);
        ggml_s10i2_decode_row(n, codes, scales, q1, d1);
        if (memcmp(q0, q1, (size_t) n) != 0 || memcmp(d0, d1, (size_t) nb * sizeof(float)) != 0) {
            GGML_ABORT("s10-int2 CHECK: decode mismatch on %s row %lld", src0->name, (long long) (e->row0 + r));
        }
    }
    if (g_check_n >= 64 || src1->ne[1] < 1) {
        goto done;
    }
    // First 64 conversions: one Q8_0 column from src1, compare both kernels bit-for-bit (before page-out).
    memcpy(tmpf, src1->data, (size_t) n * sizeof(float));
    quantize_row_q8_0(tmpf, ybuf, n);
    for (int64_t r = 0; r + 1 < e->rows; r += 2) {
        float s_s2[2], s_ptq[2];
        const void * ptq0 = (const char *) src0->data + (e->row0 + r) * src0->nb[1];
        ggml_vec_dot_s10i2_q8_0_x2(n, s_s2,
                e->codes + r * nb * 32, e->codes + (r + 1) * nb * 32,
                (const ggml_half *) (e->scales + r * nb), (const ggml_half *) (e->scales + (r + 1) * nb),
                ybuf, 0);
        ggml_vec_dot_ptq1_0_q8_0_x2(n, s_ptq, ptq0, src0->nb[1], ybuf, 0);
        if (memcmp(s_s2, s_ptq, sizeof(s_s2)) != 0) {
            GGML_ABORT("s10-int2 CHECK: kernel mismatch on %s rows %lld/%lld",
                       src0->name, (long long) (e->row0 + r), (long long) (e->row0 + r + 1));
        }
    }
    if (e->rows & 1) {
        const int64_t r = e->rows - 1;
        float s_s2[2], s_ptq;
        const void * ptq0 = (const char *) src0->data + (e->row0 + r) * src0->nb[1];
        ggml_vec_dot_s10i2_q8_0_x2(n, s_s2,
                e->codes + r * nb * 32, e->codes + r * nb * 32,
                (const ggml_half *) (e->scales + r * nb), (const ggml_half *) (e->scales + r * nb),
                ybuf, 0);
        ggml_vec_dot_ptq1_0_q8_0(n, &s_ptq, 0, ptq0, 0, ybuf, 0, 1);
        if (memcmp(&s_s2[0], &s_ptq, sizeof(float)) != 0) {
            GGML_ABORT("s10-int2 CHECK: kernel mismatch on %s row %lld", src0->name, (long long) (e->row0 + r));
        }
    }
    g_check_n++;
done:
    free(q0); free(q1); free(d0); free(d1); free(ybuf); free(tmpf);
}

static struct ggml_s10i2 * s10i2_lookup(const struct ggml_tensor * src0, const struct ggml_tensor * src1, int64_t gpu_rows) {
    size_t h = ((uintptr_t) src0->data >> 6) % S10I2_TABLE;
    for (int n = 0; n < S10I2_TABLE; n++, h = (h + 1) % S10I2_TABLE) {
        if (g_tab[h].key == src0->data) {
            return &g_tab[h];
        }
        if (g_tab[h].key == NULL) {
            struct ggml_s10i2 * e = &g_tab[h];
            e->key = src0->data;
            e->codes = NULL;
            e->scales = NULL;
            e->row0 = gpu_rows;
            e->rows = 0;
            e->nb = (int) (src0->ne[0] / 128);
            e->pf = 0;

            const int64_t M = src0->ne[1];
            const int64_t rows = M - gpu_rows;
            if (rows <= 0) {
                return e;
            }
            const int nb = e->nb;
            const size_t codes_bytes = (size_t) rows * nb * 32;
            const size_t scales_bytes = (size_t) rows * nb * sizeof(ggml_half);
            const size_t need = s10i2_align64(codes_bytes) + scales_bytes;
            const size_t need_mib_num = g_total_bytes + need;
            if (g_max_mb > 0 && need_mib_num > (size_t) g_max_mb * 1048576ull) {
                if (!g_cap_logged) {
                    g_cap_logged = 1;
                    s10i2_log_stats();
                }
                return e;   // rows stays 0
            }

            uint8_t * mem = (uint8_t *) s10i2_alloc(need);
            if (!mem) {
                fprintf(stderr, "s10-int2: alloc %zu bytes for %s failed\n", need, src0->name);
                return e;
            }
            uint8_t * codes = mem;
            ggml_half * scales = (ggml_half *) (mem + s10i2_align64(codes_bytes));

            int8_t tmp_q[32768] __attribute__((aligned(64)));
            float tmp_d[256];
            for (int64_t r = 0; r < rows; ++r) {
                const void * ptq = (const char *) src0->data + (gpu_rows + r) * src0->nb[1];
                ggml_s10i2_encode_row((int) src0->ne[0], ptq,
                                      codes + r * nb * 32, scales + r * nb, tmp_q, tmp_d);
            }

            e->codes = codes;
            e->scales = (const uint16_t *) scales;
            e->row0 = gpu_rows;
            e->rows = rows;
            e->pf = g_pf_pairs > 0 ? (size_t) g_pf_pairs * 2 * (size_t) nb * 32 : 0;

            if (g_check) {
                s10i2_check_rows(src0, src1, e);
            }
            s10i2_pageout(src0, gpu_rows, rows);

            g_total_bytes += need;
            g_over_bytes  += (size_t) rows * nb * 6;
            g_ntensors++;
            if ((g_ntensors % 64) == 0) {
                s10i2_log_stats();
            }
            return e;
        }
    }
    return NULL;
}

const struct ggml_s10i2 * ggml_s10i2_get(const struct ggml_tensor * src0, const struct ggml_tensor * src1, int64_t gpu_rows) {
    if (g_state < 0) {
        g_state = s10i2_init();
    }
    if (g_state == 0) {
        return NULL;
    }
    if (src0->type != GGML_TYPE_PTQ1_0 || src1->type != GGML_TYPE_F32) {
        return NULL;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return NULL;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return NULL;
    }
    if (src0->ne[0] % 128 != 0 || src0->ne[0] > 32768) {
        return NULL;
    }
    if ((double) ggml_nbytes(src0) < g_min_bytes) {
        return NULL;
    }
    if (!s10i2_name_match(src0->name)) {
        return NULL;
    }

    struct ggml_s10i2 * e = s10i2_lookup(src0, src1, gpu_rows);
    if (!e || e->rows <= 0) {
        return NULL;
    }
    if (gpu_rows < e->row0) {
        if (!g_declined_logged) {
            g_declined_logged = 1;
            fprintf(stderr, "s10-int2: GPU rows %lld < cached row0 %lld for %s; using PTQ1_0 this call\n",
                    (long long) gpu_rows, (long long) e->row0, src0->name);
        }
        return NULL;
    }
    return e;
}

void ggml_s10i2_chunk(const struct ggml_s10i2 * e, int n, const void * y, size_t y_stride,
                      struct ggml_tensor * dst, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    const int nb = e->nb;
    const size_t pf = e->pf;
#define S10I2_C(r) (e->codes + ((r) - e->row0) * (size_t) nb * 32)
#define S10I2_S(r) ((const ggml_half *) (e->scales + ((r) - e->row0) * (size_t) nb))

    const int64_t ncols = c1 - c0;
    if (ncols < 4) {
        for (int64_t c = c0; c < c1; ++c) {
            const void * yc = (const char *) y + c * y_stride;
            float * dcol = (float *) ((char *) dst->data + c * dst->nb[1]);
            int64_t r = r0;
            for (; r + 1 < r1; r += 2) {
                float s[2];
                ggml_vec_dot_s10i2_q8_0_x2(n, s, S10I2_C(r), S10I2_C(r + 1), S10I2_S(r), S10I2_S(r + 1), yc, pf);
                dcol[r] = s[0];
                dcol[r + 1] = s[1];
            }
            if (r < r1) {
                float s[2];
                ggml_vec_dot_s10i2_q8_0_x2(n, s, S10I2_C(r), S10I2_C(r), S10I2_S(r), S10I2_S(r), yc, pf);
                dcol[r] = s[0];
            }
        }
        return;
    }

    int8_t q0[32768] __attribute__((aligned(64)));
    int8_t q1[32768] __attribute__((aligned(64)));
    float d0[256], d1[256];
    const int64_t ncols4 = ncols / 4 * 4;
    int64_t r = r0;
    for (; r + 1 < r1; r += 2) {
        ggml_s10i2_decode_row(n, S10I2_C(r), S10I2_S(r), q0, d0);
        ggml_s10i2_decode_row(n, S10I2_C(r + 1), S10I2_S(r + 1), q1, d1);
        for (int64_t c = c0; c < c0 + ncols4; c += 4) {
            float s0[4], s1[4];
            ggml_ptq1_0_dot_dec_2x4_s10(n, q0, q1, d0, d1, (const char *) y + c * y_stride, y_stride, s0, s1);
            for (int k = 0; k < 4; ++k) {
                float * dcol = (float *) ((char *) dst->data + (c + k) * dst->nb[1]);
                dcol[r] = s0[k];
                dcol[r + 1] = s1[k];
            }
        }
        for (int64_t c = c0 + ncols4; c < c1; ++c) {
            float s[2];
            ggml_vec_dot_s10i2_q8_0_x2(n, s, S10I2_C(r), S10I2_C(r + 1), S10I2_S(r), S10I2_S(r + 1),
                                       (const char *) y + c * y_stride, pf);
            float * dcol = (float *) ((char *) dst->data + c * dst->nb[1]);
            dcol[r] = s[0];
            dcol[r + 1] = s[1];
        }
    }
    if (r < r1) {
        for (int64_t c = c0; c < c1; ++c) {
            float s[2];
            ggml_vec_dot_s10i2_q8_0_x2(n, s, S10I2_C(r), S10I2_C(r), S10I2_S(r), S10I2_S(r),
                                       (const char *) y + c * y_stride, pf);
            float * dcol = (float *) ((char *) dst->data + c * dst->nb[1]);
            dcol[r] = s[0];
        }
    }
#undef S10I2_C
#undef S10I2_S
}

#else  // !aarch64 || !dotprod

const struct ggml_s10i2 * ggml_s10i2_get(const struct ggml_tensor * src0, const struct ggml_tensor * src1, int64_t gpu_rows) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(gpu_rows);
    return NULL;
}

void ggml_s10i2_chunk(const struct ggml_s10i2 * e, int n, const void * y, size_t y_stride,
                      struct ggml_tensor * dst, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    GGML_UNUSED(e); GGML_UNUSED(n); GGML_UNUSED(y); GGML_UNUSED(y_stride);
    GGML_UNUSED(dst); GGML_UNUSED(r0); GGML_UNUSED(r1); GGML_UNUSED(c0); GGML_UNUSED(c1);
    GGML_ABORT("ggml_s10i2_chunk needs aarch64+dotprod");
}

#endif
