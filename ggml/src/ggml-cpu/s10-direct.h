// S10 GPU assist without Rusticl: build and submit Panfrost compute jobs for the s10-gpu-assist kernels directly.
// Mali-G76 (Bifrost v7) only. Shaders are pre-compiled (s10-direct-shaders.h); descriptors follow Mesa 26.2.6
// genxml v7.xml and pan_jm.c. Notes, captures and measurements: s10-wifi/gpu/direct/README.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct s10d_buf {
    uint32_t  handle;   // GEM handle on the s10d DRM fd (0 = none)
    uint64_t  va;       // GPU address
    uint8_t * cpu;      // CPU mapping (write-combined)
    size_t    size;
};

enum s10d_kernel { S10D_MV8 = 0, S10D_MM2 = 1, S10D_MM4 = 2, S10D_NKERNELS };

int  s10d_init(void);                                  // 1 = ready, 0 = unavailable (no panfrost / not Bifrost v7)
int  s10d_buf_alloc(struct s10d_buf * b, size_t size); // 1 = ok; zero-filled
void s10d_buf_free(struct s10d_buf * b);
int  s10d_buf_reserve(struct s10d_buf * b, size_t need); // grow (free + alloc) when smaller than need

// Queue one dispatch of KERNEL into the pending chain. Arguments as the OpenCL kernels take them
// (s10-gpu-assist-kernel.h); local_bytes = size of the __local tmp argument; groups_x/groups_y = work-groups.
int  s10d_add(enum s10d_kernel k, const struct s10d_buf * w, const struct s10d_buf * qx, const struct s10d_buf * dx,
              const struct s10d_buf * y, const struct s10d_buf * as, int nb, int nr, int wg, int local_bytes,
              int groups_x, int groups_y, int group_base, int col_base);
int  s10d_submit(void);   // chain + cache-flush job -> one DRM_IOCTL_PANFROST_SUBMIT; 1 = ok
int  s10d_wait(void);     // spin until the last submitted chain's results are in memory; 1 = ok
// Ticket of the chain just submitted: wait for that chain only (chains run in submit order), so a caller can use the
// first chain's results while a later one (e.g. the gate/up partner) still runs. Valid until 8 more chains.
typedef const volatile uint32_t * s10d_ticket;
s10d_ticket s10d_last_ticket(void);
int  s10d_wait_ticket(s10d_ticket t);
int  s10d_pending(void);  // a submitted chain has not been waited for yet

#ifdef __cplusplus
}
#endif
