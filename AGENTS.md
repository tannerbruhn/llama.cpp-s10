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
- int2 CPU kernel (experimental): `ggml/src/ggml-cpu/s10-int2.{c,h}`, kernels in `arch/arm/quants.c`
  (`ggml_s10i2_*`), switches `GGML_S10_INT2*`; bit-identity test `tests/test-s10-int2.cpp`.
- Recipes, server config and measured numbers: https://github.com/tannerbruhn/s10-llm-recipes

## Build (on the phone, aarch64 musl)
```sh
F="-DGGML_PTQ1_LEAN=0 -DGGML_PTQ1_LEAN2=1 -DGGML_PTQ1_PREFETCH=0"
cmake -B build-mali -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_CPU_S10_GPU_ASSIST=ON -DCMAKE_C_FLAGS="$F" -DCMAKE_CXX_FLAGS="$F"
cmake --build build-mali -j4 --target llama-server llama-bench llama-perplexity test-backend-ops
```
All of these flags matter: without `GGML_PTQ1_LEAN2=1` the S10 PTQ1_0 kernels are compiled out (PTQ1_0 then aborts
or runs about 2x slower); `GGML_NATIVE=ON` on a newer Arm build machine enables i8mm, which the S10 cannot run.
The GPU assist also needs `clang23-headers` (Alpine) at runtime, or Rusticl cannot compile its kernel and the
assist silently turns off.
If the phone cannot download the server's web UI during the build (the `llama-ui-assets` step fails with
"UI: download ... failed" or "missing required asset(s)"), configure with `-DLLAMA_USE_PREBUILT_UI=OFF
-DLLAMA_BUILD_UI=OFF` and delete `build-mali/tools/ui/dist` if a partial download is there: the server then has no
embedded UI, which is what `--no-webui` serves anyway.
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
