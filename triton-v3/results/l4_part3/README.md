# Part 3: Custom Triton kernel (V3) and initial optimizations (NVIDIA L4, midpoint)

## Setup
- **V3** is our fused Triton kernel (residual add + RMSNorm + dynamic per-token INT8). It was added to the Part 1
  benchmark script, so it uses the same inputs, reference, tolerances, CUDA-graph timing and cold-cache flush as V0–V2.
- **Seven configurations**, `r<rows per program>w<warps>[p]`:
  - `r1w8`: same design as V2 (control)
  - `r1w4`, `r1w2`, `r1w1`: fewer warps per row, so fewer block-wide barriers
  - `r2w2`, `r4w4`: one warp per row, several rows per program
  - `r1w4p`: persistent grid
  These test the fixes suggested by the Part 2 profile of V1.
- **Shapes:** BF16 with residual add; H = 2048 and 4096; T = 8 to 16384; warm (100 samples) and cold (100 single calls).
- **Clock:** SM clock locked at **1200 MHz**, chosen to avoid the L4's 72 W power cap. It was power-capped in 2% of
  samples, versus about 65% at 2040 MHz.
- **Comparisons:** V3 vs V0, V1 and V2 come from the same run. Absolute times are not comparable with Part 1 (2040 MHz).

## Correctness
- **Main run:** 880/880 comparisons passed (V3: 784/784). V3's max dequantized error vs FP32 was 1.38 INT8 steps
  (V0: 1.42).
- **Edge cases** (`test_edge_cases.py`): **364/364 V3 checks passed.** Cases: T = 1, 3, 7, 1000, 1023;
  H = 5120 and 3000 (masked tail); all-zero rows (q = 0, finite scale); large values with a 3×10⁴ outlier;
  BF16 and FP16; with and without residual. V0, V1 and V2 also passed every case.

## Results (warm cache, median µs, best V3 config per shape)

| H | T | V0 | V1 | V2 | best V3 | config | V3 vs V0 | V3 vs V1 | V3 vs V2 |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | 5.6 | 4.5 | 3.0 | 3.0 | r1w8 | 1.85× | 1.49× | 0.99× |
| 2048 | 256 | 8.7 | 17.8 | 5.8 | 5.8 | r1w8 | 1.49× | 3.05× | 0.99× |
| 2048 | 1024 | 22.8 | 61.4 | 13.5 | 13.4 | r1w8 | 1.70× | 4.58× | 1.01× |
| 2048 | 2048 | 39.8 | 121.0 | 23.4 | 23.0 | r1w8 | 1.73× | 5.25× | 1.02× |
| 2048 | 4096 | 111.3 | 240.0 | 60.9 | 55.9 | r1w1 | 1.99× | 4.29× | 1.09×* |
| 2048 | 16384 | 1566.7 | 1086.5 | 974.3 | 973.8 | r2w2 | 1.61× | 1.12× | 1.00× |
| 4096 | 256 | 13.1 | 22.5 | 9.6 | 9.9 | r1w4 | 1.32× | 2.26× | 0.96× |
| 4096 | 1024 | 38.5 | 77.3 | 24.1 | 24.6 | r1w4p | 1.57× | 3.14× | 0.98× |
| 4096 | 2048 | 111.0 | 152.6 | 61.2 | 53.7 | r1w4p | 2.07× | 2.84× | 1.14×* |
| 4096 | 16384 | 3088.4 | 1993.7 | 1923.6 | 1948.7 | r1w4 | 1.58× | 1.02× | 0.99× |

\* Bimodal shapes, see below. These "wins" over V2 are within the timing spread and should not be claimed.

**Each config: geometric-mean speedup over all shapes**

| config | warm vs V0 | warm vs V2 | cold vs V0 | cold vs V2 |
|---|---|---|---|---|
| r1w8 (V2 design) | **1.60×** | **0.98×** | **1.35×** | **0.99×** |
| r1w4 | 1.56× | 0.95× | 1.33× | 0.98× |
| r1w4p (persistent) | 1.54× | 0.94× | 1.29× | 0.94× |
| r1w2 | 1.50× | 0.92× | 1.31× | 0.96× |
| r1w1 | 1.32× | 0.80× | 1.22× | 0.89× |
| r2w2 | 1.31× | 0.80× | 1.27× | 0.93× |
| r4w4 | 1.22× | 0.75× | 1.23× | 0.90× |

## Findings
1. **V3 beats both vLLM paths, but only matches V2, not beats it.** V3 vs V0: 1.3–2.0× warm, 1.2–1.6× cold.
   V3 vs V1: up to 5.25× warm. The best V3 config is within about ±5% of V2 everywhere, outside the bimodal shapes.
2. **The fixes Part 2 suggested for V1 do not help a Triton kernel; they hurt.** Fewer warps per row and one warp
   per row (`r2w2`, `r4w4`) are 8–25% slower than the 8-warp design in warm cache. A persistent grid gives no gain.
   With a cold cache, all configs are within ~10% (DRAM-bound). So the plain one-row-per-program Triton design
   already avoids what makes V1 slow. V1's problem is specific to its CUDA implementation: 1024-thread blocks,
   narrow loads, ~2.3× the instructions. This still needs confirming by profiling V2/V3 with the Part 2 toolkit.
3. **Clock sensitivity confirms Part 2's diagnosis of V1.** From 2040 to 1200 MHz (1.70× slower clock), V1's
   medium-size warm time grew 1.44–1.66×, almost in proportion to the clock: instruction-bound. V0 barely changed
   at T ≥ 2048 (0.94–1.05×): memory/L2-bound. At T = 16384, nothing changed (DRAM-bound). So the V1 slowdown is
   larger at lower clocks: 4.6–5.3× slower than V3 at 1200 MHz vs ~2–2.3× slower than V0 at 2040 MHz.
4. **The bimodal timings from Part 1 had two different causes.**
   - At a steady clock, V0's and V1's spread collapses to 0.1–1.5%. Their Part 1 jumps came from power-cap throttling.
   - The Triton kernels (V2 and all V3 configs) are still bimodal at a steady clock, but only at the two shapes where
     each activation tensor is 16 MB (H=2048/T=4096 and H=4096/T=2048). They switch between ~45 µs and ~60 µs.
     There, the fused kernel's working set (about 40 MB) sits just under the L4's 48 MB L2. That is consistent with
     H2, and would need L2 hit rates (Part 2 toolkit) to confirm.

## Limitations
- **V2 is not the issue author's kernel** (AshrithaG/int8-linear). It is the benchmark's built-in Triton kernel, so
  "V3 vs an existing Triton kernel" is currently a comparison between two kernels the team wrote. The author's
  `add_rmsnorm_quant` (in `int8_linear/fused_quant.py`) will be added as V2 for the four-way comparison.
- One GPU (L4), one clock setting (1200 MHz) for this run. L40S/A6000 are pending.
- Configurations are a fixed grid, not autotuned per shape.

## Next steps
1. Add the author's kernel as V2 (adapter writing directly into the output buffers, so no copy is timed).
2. Profile V2/V3 with the Part 2 toolkit (load width, instructions, L2 hit rate) to explain finding 2 and the
   16 MB bimodality.
3. Targeted V3 work: per-shape config selection, cache-eviction hints for the residual stream, and `num_stages`.
4. Hand V3 (r1w8 or per-shape best) to Part 4 for the end-to-end integration test.
