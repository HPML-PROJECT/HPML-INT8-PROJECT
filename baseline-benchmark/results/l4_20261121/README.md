# WS1 main run: NVIDIA L4, 2026-11-21

Full-grid benchmark of V0 (vLLM unfused), V1 (vLLM fused CUDA) and V2 (the team's built-in Triton kernel).

## Summary

- **All 1,440 correctness checks pass**, for every variant, dtype, shape and residual mode.
- **V1 is slower than V0 only while the working set fits in the L4's 48 MB L2.** With a warm L2, V1 takes up to 2.31× as long as V0 (BF16, residual add, H=2048, T=1024; 2.82× without residual). Once the working set (about 5 bytes × H per token) passes 48 MB, V1 is faster (0.62–0.69×). In cold-cache runs, V1 is never meaningfully slower.
- **V2 is the fastest variant**: geometric-mean speedup over V0 of 1.73× warm and 1.37× cold, against 1.06× and 1.20× for V1.

## Setup

| | |
|---|---|
| Hardware | NVIDIA L4 (Ada, sm_89, 58 SMs, 48 MB L2, 22 GB usable), Google Cloud, driver 620.70.06 |
| Software | vLLM 0.34.0, PyTorch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0, Python 3.13.5 |
| Grid | T ∈ {1, 2, 4, …, 16384} (15 values) × H ∈ {2048, 4096, 5120, 8192} × FP16, BF16 × residual off/on × warm/cold × V0, V1, V2 = 1,440 timing rows |
| Inputs | seed 1234, per-shape CPU generator (see the folder README) |
| Warm timing | 25 untimed calls to size the graph, then a CUDA graph of N back-to-back calls (N for ~2 ms, max 500); 3 warm-up calls before capture and 1 warm-up replay; 50 timed replays; latency = median of replay time / N |
| Cold timing | 1 warm-up call; 100 individually timed calls, each preceded by writing a 384 MB buffer to evict L2 |
| Clocks | Default boost clock under the 72 W power limit; 2040 MHz at the end of the run (71.8 °C, 34.5 W) |

Command: `python bench_rmsnorm_int8.py --residual both --out-dir results_l4` (all other settings default; full arguments in `env.json`).

## Correctness (`correctness.csv`)

1,440 of 1,440 comparisons pass: each variant against the FP32 reference (720) and pairwise (720).

| Check | Worst case | Limit |
|---|---|---|
| Dequantized error vs. FP32 reference | 2.16 INT8 steps (V1, BF16, T=4096, H=4096, residual) | 2.5 |
| INT8 mismatch, V1 vs. V0 | 12.3% (BF16, T=2, H=4096, residual) | 15% |
| Largest INT8 difference | 3 steps (V2 vs. V1, BF16, T=512, H=2048, residual) | 3 |
| Scale difference | 1.39% (V2 vs. V1, BF16, T=64, H=2048, residual) | 1.6% |
| Updated residual vs. reference | 0 (exact) | atol 1e-2 |

**Why BF16 shows more INT8 mismatches.** V1 vs. V0 mismatches have a median of 0.83% in FP16 but 6.11% in BF16 (maximum 2.47% vs. 12.3%). The likely cause: BF16 keeps only 8 significant bits, so the two kernels' different rounding points (V0 rounds the normalized output to BF16 before quantizing it in a separate kernel; V1 quantizes inside one kernel) move values across INT8 rounding boundaries more often than in FP16. All differences are 1–3 INT8 steps and within tolerance. The vLLM issue reported 0.2–1.1% mismatches, in line with our FP16 numbers.

## Performance (`bench_results.csv`, plots in `plots/`)

**V1 / V0 time, BF16, residual add (>1 means V1 is slower):**

| Cache | H | T=1 | 64 | 128 | 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|
| warm | 2048 | 0.77 | 0.99 | 1.13 | **1.97** | **2.31** | **1.83** | **1.59** | 0.68 | 0.66 |
| warm | 4096 | 0.82 | 1.05 | 1.17 | **1.75** | **1.46** | 1.15 | 0.69 | 0.66 | 0.65 |
| warm | 5120 | 0.85 | 1.10 | 1.25 | **1.69** | 1.11 | 0.83 | 0.66 | 0.65 | 0.65 |
| warm | 8192 | 0.85 | 1.09 | 1.29 | **1.37** | 0.86 | 0.69 | 0.63 | 0.62 | 0.62 |
| cold | 2048 | 1.00 | 1.00 | 0.94 | 0.96 | 0.87 | 0.86 | 0.81 | 0.68 | 0.61 |
| cold | 4096 | 0.88 | 1.00 | 0.89 | 0.90 | 0.84 | 0.80 | 0.70 | 0.66 | 0.65 |
| cold | 5120 | 1.00 | 1.11 | 0.93 | 0.90 | 0.76 | 0.80 | 0.67 | 0.65 | 0.65 |
| cold | 8192 | 1.00 | 1.00 | 0.98 | 0.88 | 0.79 | 0.70 | 0.64 | 0.63 | 0.63 |

**The crossover follows the L2 size.** The working set is about 5 bytes × H per token (x and the residual read, the residual written, the INT8 output). V1 stops being slower right where that passes 48 MB:

| H | Last warm T where V1 is slower | Working set there | First T where V1 is faster | Working set there |
|---|---|---|---|---|
| 2048 | 4096 | 42 MB | 8192 | 84 MB |
| 4096 | 2048 | 42 MB | 4096 | 84 MB |
| 5120 | 1024 | 26 MB | 2048 | 52 MB |
| 8192 | 512 | 21 MB | 1024 | 42 MB |

At H = 8192, V1 is already faster at T = 1024 (42 MB), just below the L2 size. Elsewhere the switch happens as the working set crosses 48 MB.

**Geometric-mean speedup over V0** (all 240 shapes per mode; from `run.log`):

| | V1 | V2 |
|---|---|---|
| warm | 1.06× (0.35–2.48×) | 1.73× (1.06–5.32×) |
| cold | 1.20× (0.87–2.48×) | 1.37× (1.08–2.33×) |

The largest slowdown is V1 at 2.82× V0's time (BF16, no residual, warm, H=2048, T=1024).

**Effective bandwidth** (minimum bytes ÷ median time) is a sanity check. Cold runs reach at most 248 GB/s, about 83% of the L4's ~300 GB/s, as expected for DRAM-bound kernels. Warm runs reach up to 1,655 GB/s, which is only possible because the data is served from L2.

## Measurement notes

- **V2** is the team's Triton baseline kernel, built into `bench_rmsnorm_int8.py`. Other Triton implementations can be benchmarked in its place with `--v2-impl module:function`.
- **Clocks:** the GPU ran at its default boost clock (2040 MHz) under the L4's 72 W power limit. A 200-repetition rerun of the shapes near the L2 limit gave the same V1/V0 direction at every shape, with warm ratios within about ±20%; the tables report the medians from this run.
- **Shapes near the L2 limit** show two timing levels, depending on how much of the working set stays in L2. For V1 at H=2048, T=4096 (warm), the minimum is 144 µs and the median 173 µs (V1/V0 = 1.36 by minimums, 1.59 by medians). `bench_results.csv` includes `min_us`, `p10_us` and `p90_us` for every row.
- **Cold timings** are single calls measured with CUDA events, whose resolution is about 0.5 µs. Cold ratios are therefore most precise from T ≈ 128 upward, where kernel times are tens of microseconds or more.
- **Warm timing** replays up to 500 back-to-back calls per graph, with inputs restored before each replay. The kernels' runtime does not depend on input values, and correctness is checked separately on fresh inputs.

## Files

| File | Contents |
|---|---|
| `bench_results.csv` | 1,440 timing rows |
| `correctness.csv` | 1,440 correctness comparisons |
| `env.json` | GPU, software versions, all arguments and tolerances |
| `run.log` | Console log of the runs on the VM, including environment setup |
| `plots/` | Latency vs. tokens and speedup vs. V0, per dtype × residual × cache mode (16 PNGs) |
