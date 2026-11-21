// S10 GPU assist without Rusticl (GGML_S10_GPU_DIRECT=1): Panfrost compute jobs for the s10-gpu-assist kernels,
// built and submitted directly. Per call: one DRM_IOCTL_PANFROST_SUBMIT of a chain [compute job(s)..., cache flush],
// then the caller spins on the flush job's status word. Measured on the S10 (s10-wifi/gpu/direct/README.md):
// ~2x faster per call than Rusticl (e.g. FFN up/gate share 1391 -> ~735 us), bit-identical output.
//
// Layout facts (Mesa 26.2.6, src/panfrost/genxml/v7.xml, gallium/drivers/panfrost/pan_jm.c, lib/pan_desc.h):
//   Compute Job = Job Header (+0, 32 B) + Invocation (+32) + Parameters (+40) + Draw (+64, 128 B)
//   Draw words: 6 uniform buffers, 10 samplers, 12 push uniforms, 14 renderer state, 28 thread storage
//   Uniform buffer entry = (count - 1) | (address >> 4) << 12
//   Local Storage: word1 = log2(WLS instances) | WLS size scale << 8; WLS base pointer at +16
//   Push uniforms of all three kernels (found by diffing captures): w0 nb, w1 nr, w2 local size x, w3-4 w,
//   w5-6 qx, w7-8 dx, w9-10 y, w12 group_base, w13 col_base, w14-15 asum. Uniform buffer 1 = {local x, 1, 1, 127}.
//   Completion: the job manager writes a job's status before the kernel's end-of-chain L2 flush, so the chain ends
//   with a CACHE_FLUSH job (shader-core LS clean, job-manager clean, L2 clean) and we spin on its status.
#include "s10-direct.h"
#include "s10-direct-shaders.h"
#include "s10-panfrost-drm.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define S10D_MAX_JOBS   16          // dispatches per chain; a full chain is submitted and the next page started, so a
                                    // long prompt is several short submits (Panfrost's job timeout is per chain)
#define S10D_SLOT       0x200       // descriptor bytes per dispatch
#define S10D_PAGES      8           // ring of descriptor pages; chains run in submit order, a page is reused only
                                    // after its own chain's flush job reported done
#define S10D_MAX_BOS    64

static int      s10d_fd = -1;
static struct s10d_buf s10d_sh[S10D_NKERNELS], s10d_state, s10d_wls, s10d_tls, s10d_desc[S10D_PAGES];
static uint32_t s10d_sync;
static unsigned s10d_cores;         // max shader core id + 1 (WLS is per core)
static int      s10d_page = -1, s10d_njobs, s10d_inflight;   // inflight: a submitted chain not yet waited for
static uint32_t s10d_bos[S10D_MAX_BOS]; static int s10d_nbos;
static volatile uint32_t * s10d_flush_status;                 // last submitted chain's flush job
static volatile uint32_t * s10d_page_flush[S10D_PAGES];        // each page's last flush job (NULL = page free)

static double s10d_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int64_t s10d_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (int64_t) t.tv_sec * 1000000000 + t.tv_nsec; }
static int s10d_ioctl(unsigned long req, void * arg) { int r; do { r = ioctl(s10d_fd, req, arg); } while (r == -1 && (errno == EINTR || errno == EAGAIN)); return r; }
static void w32(uint8_t * p, uint32_t v) { memcpy(p, &v, 4); }
static void w64(uint8_t * p, uint64_t v) { memcpy(p, &v, 8); }
static unsigned log2_ceil(unsigned v) { unsigned r = 0; while ((1u << r) < v) r++; return r; }
static unsigned npot(unsigned v) { return 1u << log2_ceil(v); }

int s10d_buf_alloc(struct s10d_buf * b, size_t size) {
    memset(b, 0, sizeof *b);
    size = (size + 4095) & ~(size_t) 4095;
    struct drm_panfrost_create_bo c = { .size = (uint32_t) size, .flags = PANFROST_BO_NOEXEC };
    if (b == &s10d_sh[0] || b == &s10d_sh[1] || b == &s10d_sh[2]) c.flags = 0;   // shaders must be executable
    if (s10d_ioctl(DRM_IOCTL_PANFROST_CREATE_BO, &c)) return 0;
    struct drm_panfrost_mmap_bo m = { .handle = c.handle };
    if (s10d_ioctl(DRM_IOCTL_PANFROST_MMAP_BO, &m)) goto fail;
    void * p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, s10d_fd, (off_t) m.offset);
    if (p == MAP_FAILED) goto fail;
    *b = (struct s10d_buf) { c.handle, c.offset, p, size };     // new GEM pages are zero-filled by the kernel
    return 1;
fail: {
        struct drm_gem_close gc = { .handle = c.handle };
        s10d_ioctl(DRM_IOCTL_GEM_CLOSE, &gc);
        return 0;
    }
}

void s10d_buf_free(struct s10d_buf * b) {
    if (!b->handle) return;
    if (b->cpu) munmap(b->cpu, b->size);
    struct drm_gem_close gc = { .handle = b->handle };
    s10d_ioctl(DRM_IOCTL_GEM_CLOSE, &gc);
    memset(b, 0, sizeof *b);
}

int s10d_buf_reserve(struct s10d_buf * b, size_t need) {
    if (b->handle && b->size >= need) return 1;
    if (s10d_inflight) s10d_wait();
    s10d_buf_free(b);
    return s10d_buf_alloc(b, need + need / 2 + 4096);
}

int s10d_init(void) {
    if (s10d_fd >= 0) return 1;
    const char * dev = getenv("GGML_S10_GPU_DIRECT_DEV") ? getenv("GGML_S10_GPU_DIRECT_DEV") : "/dev/dri/renderD128";
    s10d_fd = open(dev, O_RDWR | O_CLOEXEC);
    if (s10d_fd < 0) { fprintf(stderr, "s10-direct: %s: %s\n", dev, strerror(errno)); return 0; }
    char name[32] = {0};
    struct drm_version v = { .name = name, .name_len = sizeof name - 1 };
    struct drm_panfrost_get_param gp = { .param = DRM_PANFROST_PARAM_GPU_PROD_ID }, cores = { .param = DRM_PANFROST_PARAM_SHADER_PRESENT };
    if (s10d_ioctl(DRM_IOCTL_VERSION, &v) || strcmp(name, "panfrost") ||
        s10d_ioctl(DRM_IOCTL_PANFROST_GET_PARAM, &gp) || s10d_ioctl(DRM_IOCTL_PANFROST_GET_PARAM, &cores)) {
        fprintf(stderr, "s10-direct: %s is not panfrost\n", dev); goto fail;
    }
    // Mali-G76 product id 0x7211 (Bifrost v7); the embedded shaders and descriptors are for that GPU only.
    if ((gp.value >> 16) != 0x7211 && gp.value != 0x7211) {
        fprintf(stderr, "s10-direct: GPU product 0x%llx is not a Mali-G76, direct path off\n", (unsigned long long) gp.value); goto fail;
    }
    s10d_cores = cores.value ? 64 - (unsigned) __builtin_clzll(cores.value) : 1;
    const unsigned char * src[S10D_NKERNELS] = { s10d_shader_mv8, s10d_shader_mm2, s10d_shader_mm4 };
    const size_t len[S10D_NKERNELS] = { sizeof s10d_shader_mv8, sizeof s10d_shader_mm2, sizeof s10d_shader_mm4 };
    for (int k = 0; k < S10D_NKERNELS; k++) {
        if (!s10d_buf_alloc(&s10d_sh[k], len[k] + 256)) goto fail;   // + prefetch slack past the last clause
        memcpy(s10d_sh[k].cpu, src[k], len[k]);
    }
    if (!s10d_buf_alloc(&s10d_state, 4096)) goto fail;
    for (int k = 0; k < S10D_NKERNELS; k++) {                     // one 64-byte renderer state per kernel
        memcpy(s10d_state.cpu + 64 * k, s10d_rsd_template, 64);
        w64(s10d_state.cpu + 64 * k, s10d_sh[k].va);
    }
    // WLS: largest kernel local size (2048 B, mm4) x max instances per core (64) x cores
    if (!s10d_buf_alloc(&s10d_wls, (size_t) 2048 * 64 * s10d_cores)) goto fail;
    // thread-local stack (mm kernels spill): 16 << size field bytes per thread x 2048 threads (headroom over the
    // G76's per-core limit) x cores, as pan_get_total_stack_size; one buffer for all kernels
    unsigned tmax = 0; for (int k = 0; k < S10D_NKERNELS; k++) if (s10d_tls_used[k] && s10d_tls_size[k] > tmax) tmax = s10d_tls_size[k];
    if (!s10d_buf_alloc(&s10d_tls, ((size_t) 16 << tmax) * 2048 * s10d_cores)) goto fail;
    for (int p = 0; p < S10D_PAGES; p++) if (!s10d_buf_alloc(&s10d_desc[p], (size_t) S10D_SLOT * (S10D_MAX_JOBS + 1))) goto fail;
    struct drm_syncobj_create sc = { 0 };
    if (s10d_ioctl(DRM_IOCTL_SYNCOBJ_CREATE, &sc)) goto fail;
    s10d_sync = sc.handle;
    fprintf(stderr, "s10-direct: panfrost GPU 0x%llx, %u core slots, direct submission on\n", (unsigned long long) gp.value, s10d_cores);
    return 1;
fail:
    close(s10d_fd); s10d_fd = -1;
    return 0;
}

static void s10d_add_bo(uint32_t h) {
    for (int i = 0; i < s10d_nbos; i++) if (s10d_bos[i] == h) return;
    if (s10d_nbos < S10D_MAX_BOS) s10d_bos[s10d_nbos++] = h;
}

int s10d_add(enum s10d_kernel k, const struct s10d_buf * w, const struct s10d_buf * qx, const struct s10d_buf * dx,
             const struct s10d_buf * y, const struct s10d_buf * as, int nb, int nr, int wg, int local_bytes,
             int groups_x, int groups_y, int group_base, int col_base) {
    if (k < 0 || k >= S10D_NKERNELS || wg < 1 || groups_x < 1 || groups_y < 1) return 0;
    if (s10d_njobs == S10D_MAX_JOBS && !s10d_submit()) return 0;
    if (s10d_njobs == 0) {
        s10d_page = (s10d_page + 1) % S10D_PAGES;
        if (s10d_page_flush[s10d_page]) {                       // the page's previous chain must be finished
            const double lim = s10d_now() + 2.0;
            while (s10d_page_flush[s10d_page][0] == 0 && s10d_now() < lim) { }
            s10d_page_flush[s10d_page] = NULL;
        }
        s10d_nbos = 0;
        s10d_add_bo(s10d_desc[s10d_page].handle); s10d_add_bo(s10d_state.handle); s10d_add_bo(s10d_wls.handle);
        s10d_add_bo(s10d_tls.handle);
    }
    s10d_add_bo(s10d_sh[k].handle);
    s10d_add_bo(w->handle); s10d_add_bo(qx->handle); s10d_add_bo(dx->handle); s10d_add_bo(y->handle); s10d_add_bo(as->handle);

    const struct s10d_buf * d = &s10d_desc[s10d_page];
    uint8_t * s = d->cpu + (size_t) S10D_SLOT * s10d_njobs;
    const uint64_t sva = d->va + (uint64_t) S10D_SLOT * s10d_njobs;
    memset(s, 0, S10D_SLOT);
    // +0x00 sampler, +0x20 uniform buffer 1 data, +0x30 uniform buffer table, +0x40 push uniforms (160 B),
    // +0xe0 thread storage (32 B, 64-aligned at +0x100 below), +0x140 compute job (192 B)
    memcpy(s, s10d_sampler, 32);
    w32(s + 0x20, (uint32_t) wg); w32(s + 0x24, 1); w32(s + 0x28, 1); w32(s + 0x2c, 127);
    w64(s + 0x30, 0);                                                     // uniform buffer 0: unused (null in Mesa's job too)
    w64(s + 0x38, (((sva + 0x20) >> 4) << 12) | 0);                     // uniform buffer 1: 1 entry
    uint8_t * pu = s + 0x40;
    w32(pu + 0, (uint32_t) nb); w32(pu + 4, (uint32_t) nr); w32(pu + 8, (uint32_t) wg);
    w64(pu + 12, w->va); w64(pu + 20, qx->va); w64(pu + 28, dx->va); w64(pu + 36, y->va);
    w32(pu + 48, (uint32_t) group_base); w32(pu + 52, (uint32_t) col_base); w64(pu + 56, as->va);
    // thread storage: WLS instances = min(npot(groups x) * npot(groups y), npot(ceil(32 / wg) * 32)) (G76:
    // 1024 threads / 32 tasks per core; pan_calc_wls_instances), size = npot(max(local, 128))
    unsigned inst = npot((unsigned) groups_x) * npot((unsigned) groups_y);
    const unsigned maxi = npot(((32u + (unsigned) wg - 1) / (unsigned) wg) * 32u);
    if (inst > maxi) inst = maxi;
    const unsigned wls = npot(local_bytes > 128 ? (unsigned) local_bytes : 128u);
    if ((size_t) wls * inst * s10d_cores > s10d_wls.size) return 0;
    uint8_t * tls = s + 0x100;
    w32(tls + 0, s10d_tls_size[k]);                                   // TLS size (stack per thread = 16 << n bytes)
    w32(tls + 4, log2_ceil(inst) | (log2_ceil(wls) + 1) << 8);
    if (s10d_tls_used[k]) w64(tls + 8, s10d_tls.va);                  // TLS base pointer
    w64(tls + 16, s10d_wls.va);
    // compute job
    uint8_t * j = s + 0x140;
    w32(j + 16, 1u | (4u << 1) | (1u << 8) | ((uint32_t) (s10d_njobs + 1) << 16));   // 64b, Compute, Barrier, Index
    w32(j + 20, (uint32_t) s10d_njobs);                                               // Dependency 1: previous job
    // Invocation (pan_pack_work_groups_compute): sizes (wg,1,1), counts (groups_x, groups_y, 1)
    const unsigned vals[6] = { (unsigned) wg, 1, 1, (unsigned) groups_x, (unsigned) groups_y, 1 };
    unsigned sh[7] = {0}; uint32_t packed = 0;
    for (int i = 0; i < 6; i++) { packed |= (vals[i] - 1) << sh[i]; sh[i + 1] = sh[i] + log2_ceil(vals[i]); }
    w32(j + 32, packed);
    w32(j + 36, sh[1] | sh[2] << 5 | sh[3] << 10 | sh[4] << 16 | sh[5] << 22 | sh[3] << 28);
    w32(j + 40, (log2_ceil((unsigned) wg + 1) + 1 + 1) << 26);                        // Job Task Split
    uint8_t * dr = j + 64;
    w32(dr + 0, 3u);                                     // four components per vertex, 64b descriptor (as Mesa)
    w64(dr + 4 * 6, sva + 0x30);                         // uniform buffers
    w64(dr + 4 * 10, sva);                               // samplers
    w64(dr + 4 * 12, sva + 0x40);                        // push uniforms
    w64(dr + 4 * 14, s10d_state.va + 64 * k);            // renderer state
    w64(dr + 4 * 28, sva + 0x100);                       // thread storage
    if (s10d_njobs > 0) w64(s - S10D_SLOT + 0x140 + 24, sva + 0x140);   // previous job's Next
    s10d_njobs++;
    return 1;
}

int s10d_submit(void) {
    if (s10d_njobs == 0) return 1;
    const struct s10d_buf * d = &s10d_desc[s10d_page];
    uint8_t * f = d->cpu + (size_t) S10D_SLOT * s10d_njobs;
    const uint64_t fva = d->va + (uint64_t) S10D_SLOT * s10d_njobs;
    memset(f, 0, 64);
    w32(f + 16, 1u | (3u << 1) | (1u << 8) | ((uint32_t) (s10d_njobs + 1) << 16));   // 64b, Cache flush, Barrier
    w32(f + 20, (uint32_t) s10d_njobs);                                               // after the last compute job
    w32(f + 32, 1u | (1u << 16));                                                     // clean shader core LS + job manager
    w32(f + 36, 1u);                                                                  // L2 clean
    w64(d->cpu + (size_t) S10D_SLOT * (s10d_njobs - 1) + 0x140 + 24, fva);          // last compute job -> flush job
    s10d_flush_status = (volatile uint32_t *) f;
    s10d_page_flush[s10d_page] = s10d_flush_status;
    __sync_synchronize();
    struct drm_panfrost_submit s = { .jc = d->va + 0x140, .out_sync = s10d_sync,
                                     .bo_handles = (uintptr_t) s10d_bos, .bo_handle_count = (uint32_t) s10d_nbos };
    s10d_njobs = 0;
    if (s10d_ioctl(DRM_IOCTL_PANFROST_SUBMIT, &s)) { perror("s10-direct: SUBMIT"); return 0; }
    s10d_inflight = 1;
    return 1;
}

int s10d_pending(void) { return s10d_inflight; }

int s10d_wait(void) {
    if (s10d_njobs && !s10d_submit()) return 0;                // queued dispatches not submitted yet
    if (!s10d_inflight) return 1;
    const double lim = s10d_now() + 2.0;
    uint32_t st;
    while ((st = s10d_flush_status[0]) == 0) {
        if (s10d_now() > lim) {                                   // fall back to the kernel's fence once
            struct drm_syncobj_wait w = { .handles = (uintptr_t) &s10d_sync, .count_handles = 1, .timeout_nsec = s10d_now_ns() + 2000000000LL };
            s10d_ioctl(DRM_IOCTL_SYNCOBJ_WAIT, &w);
            st = s10d_flush_status[0];
            break;
        }
    }
    s10d_inflight = 0;
    if (st != 1) {
        fprintf(stderr, "s10-direct: chain ended with status 0x%x\n", st);
        return 0;
    }
    return 1;
}
