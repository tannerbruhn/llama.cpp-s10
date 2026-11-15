# Working on llama.cpp-s10

This is a fork for one device: the Samsung Galaxy S10 (Exynos 9820, Mali-G76, 8 GB) under postmarketOS. Humans and
AI coding agents are both welcome to write code here. Upstream's contributor and AI rules apply only when sending
code to upstream (ggml-org/llama.cpp or PrismML-Eng/llama.cpp): follow their CONTRIBUTING.md then.

## Where things are
- S10 CPU kernels for PTQ1_0 / Q4_0: `ggml/src/ggml-cpu/arch/arm/quants.c`, `ggml/src/ggml-cpu/ggml-cpu.c`
  (look for `s10` / `S10`).
- Mali GPU assist: `ggml/src/ggml-cpu/s10-gpu-assist.{c,h}`, CMake option `GGML_CPU_S10_GPU_ASSIST`; runtime
  switches are `GGML_S10_GPU_*` environment variables.
- Router default model: `LLAMA_ROUTER_DEFAULT_MODEL` in `tools/server/server-models.cpp`.
- Recipes, server config and measured numbers: https://github.com/tannerbruhn/s10-llm-recipes

## Build (on the phone, aarch64 musl)
```sh
cmake -B build-mali -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_S10_GPU_ASSIST=ON
cmake --build build-mali -j4 --target llama-server llama-bench llama-perplexity test-backend-ops
```
The GPU assist needs Mesa Rusticl on Panfrost (with the Bifrost int8-dot patch from the recipes repo) through
`OCL_ICD_VENDORS`, and `RUSTICL_ENABLE=panfrost`.

## Rules
1. **Measure every speed change on the phone.** `llama-bench` from a dropped page cache, the phone below 50 C before
   each run, alternating A/B runs (noise is about +/-0.03 tok/s). Report the charger state with the numbers.
2. **One commit per step, with its measured gain** (or "no gain, not adopted") in the commit message.
3. **Kernel changes must not change results.** Check greedy output against the CPU-only path, and perplexity
   (wikitext-2, 12 x 512) before and after.
4. **New paths go behind a switch** (environment variable or compile-time define) until measured better, and the
   old path stays reachable for A/B.
5. **Heat:** never disable the phone's thermal guard to get a number. GPU clocks only from Samsung's DVFS table
   (156-702 MHz); other rates are rejected by the firmware.
6. Commit as `tannerbruhn <tannerbruhn@users.noreply.github.com>`, without co-author or tool trailers.
