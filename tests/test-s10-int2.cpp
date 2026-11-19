// S10 int2 CPU kernel: the s2 path must be bit-identical to the lean-v2 PTQ1_0 path, and the 2x2 (two-column)
// kernel bit-identical to the 2-row kernel called once per column.
//  (a) decode(encode(row)) == ggml_ptq1_0_decode_row_s10(row): bytes and scales
//  (b) ggml_vec_dot_s10i2_q8_0_x2 floats == ggml_vec_dot_ptq1_0_q8_0_x2 floats, bit for bit
// Random valid PTQ1_0 blocks use the codeword construction of the microbench (kernels/microbench/ptq1bench.c:886-887):
// a base-3 byte holding 5 trits is ceil(256 * sum(t_i * 3^(4-i)) / 243), the qh bytes hold 4 trits: ceil(256 * sum / 81).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#if defined(__aarch64__)
extern "C" {
typedef uint16_t ggml_half;
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_ptq1_0_decode_row_s10(int n, const void * vx, int8_t * q, float * d);
void ggml_vec_dot_ptq1_0_q8_0_x2(int n, float * s, const void * vx, size_t bx, const void * vy, size_t prefetch);
void ggml_s10i2_encode_row(int n, const void * ptq_row, uint8_t * codes, ggml_half * scales, int8_t * tmp_q, float * tmp_d);
void ggml_s10i2_decode_row(int n, const uint8_t * codes, const ggml_half * scales, int8_t * q, float * d);
void ggml_vec_dot_s10i2_q8_0_x2(int n, float * s, const uint8_t * c0, const uint8_t * c1,
                                const ggml_half * d0, const ggml_half * d1, const void * vy, size_t prefetch);
void ggml_vec_dot_s10i2_q8_0_x2c2(int n, float * s, const uint8_t * c0, const uint8_t * c1,
                                  const ggml_half * d0, const ggml_half * d1, const void * vya, const void * vyb, size_t prefetch);
}
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(block_ptq1_0) == 28, "PTQ1_0 block");
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block");

static uint8_t codeword(std::mt19937 & g, int ntrits, int div) {
    int v = 0;
    for (int i = 0; i < ntrits; ++i) v = v * 3 + (int) (g() % 3);
    return (uint8_t) ((256 * v + div - 1) / div);
}

static uint16_t rand_half(std::mt19937 & g) {
    for (;;) {                                   // finite, non-NaN fp16, either sign, wide exponent range
        uint16_t h = (uint16_t) (g() & 0xFFFF);
        if (((h >> 10) & 0x1F) != 0x1F) return h;
    }
}

int main() {
    std::mt19937 g(12345);
    int fails = 0, rows_checked = 0;
    for (int n : {128, 5120, 6144, 17408}) {
        const int nb = n / 128;
        for (int rep = 0; rep < 20; ++rep) {
            std::vector<block_ptq1_0> w(2 * nb);
            for (auto & b : w) {
                for (int j = 0; j < 24; ++j) b.qs[j] = codeword(g, 5, 243);
                for (int j = 0; j < 2; ++j)  b.qh[j] = codeword(g, 4, 81);
                b.d = rand_half(g);
            }
            std::vector<float> x(n); std::uniform_real_distribution<float> u(-4.0f, 4.0f);
            for (auto & v : x) v = u(g);
            std::vector<block_q8_0> y(n / 32);
            quantize_row_q8_0(x.data(), y.data(), n);

            std::vector<uint8_t> codes(2 * nb * 32); std::vector<ggml_half> sc(2 * nb);
            std::vector<int8_t> tq(n), q_ref(n), q_s2(n); std::vector<float> td(nb), d_ref(nb), d_s2(nb);
            for (int r = 0; r < 2; ++r) {
                ggml_s10i2_encode_row(n, &w[r * nb], &codes[r * nb * 32], &sc[r * nb], tq.data(), td.data());
                ggml_ptq1_0_decode_row_s10(n, &w[r * nb], q_ref.data(), d_ref.data());
                ggml_s10i2_decode_row(n, &codes[r * nb * 32], &sc[r * nb], q_s2.data(), d_s2.data());
                if (memcmp(q_ref.data(), q_s2.data(), n) || memcmp(d_ref.data(), d_s2.data(), nb * sizeof(float))) {
                    fprintf(stderr, "FAIL (a) n=%d rep=%d row=%d\n", n, rep, r); fails++;
                }
                rows_checked++;
            }
            float s_ref[2], s_s2[2];
            ggml_vec_dot_ptq1_0_q8_0_x2(n, s_ref, w.data(), nb * sizeof(block_ptq1_0), y.data(), 0);
            ggml_vec_dot_s10i2_q8_0_x2(n, s_s2, &codes[0], &codes[nb * 32], &sc[0], &sc[nb], y.data(), rep & 1 ? 2 * nb * 32 : 0);
            if (memcmp(s_ref, s_s2, sizeof s_ref)) {
                fprintf(stderr, "FAIL (b) n=%d rep=%d: %.9g %.9g vs %.9g %.9g\n", n, rep, s_ref[0], s_ref[1], s_s2[0], s_s2[1]);
                fails++;
            }
            // (c) 2 rows x 2 columns in one pass == the 2-row kernel once per column
            std::vector<float> x2(n); for (auto & v : x2) v = u(g);
            std::vector<block_q8_0> y2(n / 32);
            quantize_row_q8_0(x2.data(), y2.data(), n);
            float s_b[2], s_c2[4];
            ggml_vec_dot_s10i2_q8_0_x2(n, s_b, &codes[0], &codes[nb * 32], &sc[0], &sc[nb], y2.data(), 0);
            ggml_vec_dot_s10i2_q8_0_x2c2(n, s_c2, &codes[0], &codes[nb * 32], &sc[0], &sc[nb], y.data(), y2.data(),
                                          rep & 1 ? 2 * nb * 32 : 0);
            if (memcmp(s_c2, s_s2, 2 * sizeof(float)) || memcmp(s_c2 + 2, s_b, 2 * sizeof(float))) {
                fprintf(stderr, "FAIL (c) n=%d rep=%d: %.9g %.9g %.9g %.9g vs %.9g %.9g %.9g %.9g\n", n, rep,
                        s_c2[0], s_c2[1], s_c2[2], s_c2[3], s_s2[0], s_s2[1], s_b[0], s_b[1]);
                fails++;
            }
        }
    }
    printf("test-s10-int2: %d rows, %s\n", rows_checked, fails ? "FAILED" : "all bit-identical");
    return fails ? 1 : 0;
}
#else
int main() { printf("test-s10-int2: skipped (needs aarch64)\n"); return 0; }
#endif
