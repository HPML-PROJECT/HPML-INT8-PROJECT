# triton-v3: our fused Triton kernel (V3) and initial optimizations (WS3)

V3 fuses **residual add + RMSNorm + dynamic per-token INT8 quantization** into one Triton kernel. It is added as a
fourth variant to the WS1 benchmark, so V3 uses exactly the same inputs, FP32 reference, tolerances, CUDA-graph timing
and cold-cache flush as V0–V2 (see `../baseline-benchmark/README.md`).

| ID | Implementation |
|---|---|
| V0 | vLLM unfused: `fused_add_rms_norm` (or `rms_norm`) → `dynamic_scaled_int8_quant` |
| V1 | vLLM fused CUDA kernel: `rms_norm_dynamic_per_token_quant` |
| V2 | Triton fused kernel built into the benchmark (**not yet** the issue author's kernel) |
| V3 | **ours**, one variant per configuration (below) |

## Files
| File | What it is |
|---|---|
| `bench_rmsnorm_int8.py` | WS1 benchmark **plus V3** (`--variants V0 V1 V2 V3`, `--v3-configs ...`). Not the same file as `../baseline-benchmark/bench_rmsnorm_int8.py`, which is kept exactly as used for the WS1 results |
| `test_edge_cases.py` | V3 correctness on edge cases (V0/V1/V2 alongside for reference) |
| `run_part3_l4.sh` | runs everything on the L4 VM: locked clock, clock/power log, benchmark, summary, results tarball |
| `summarize_v3.py` | best V3 config per shape, each config vs V0/V1/V2, correctness and clock-stability summary → `part3_summary.md` |
| `plot_results.py` | latency and speedup plots for V0–V3 (V3 = best config per shape, band = range over configs), and `v3_configs.png` comparing the configs |
| `results/l4_part3/` | L4 results; its `README.md` has the findings |

## V3 configurations
Names are `r<rows per program>w<warps>[p]`:

| Config | Meaning | Tests (from the WS2 profile of V1) |
|---|---|---|
| `r1w8` | 1 row per program, 8 warps (V2's design) | control: should match V2 |
| `r1w4`, `r1w2`, `r1w1` | 1 row, fewer warps | fewer block-wide barriers, smaller blocks |
| `r2w2`, `r4w4` | several rows per program, one warp per row | no block-wide barrier in the row reductions |
| `r1w4p` | `p` = persistent grid (4 × #SMs programs looping over rows) | launch/scheduling overhead |

All configs share V2's numerics: FP32 math, rounding to the activation dtype after normalizing and after × weight,
scale = absmax / 127, round-half-to-even, clamp to [−128, 127].

## How to run (L4 VM, from this folder)
Use the WS1 Python environment (`../baseline-benchmark/requirements.txt`). By default the scripts use `./.venv`, else
`$VENV`, else `~/rmsnorm_int8_bench/.venv`.
```bash
QUICK=1 bash run_part3_l4.sh      # ~2 min smoke test (H=2048, T=256 and 4096)
bash run_part3_l4.sh              # full run (~10-15 min): BF16, residual on, H=2048/4096, T=8..16384, warm + cold
python test_edge_cases.py         # < 1 min: edge cases, writes edge_cases.csv, exit code 0 = all V3 checks passed
python summarize_v3.py results_l4_part3   # re-summarize an existing run
```
`run_part3_l4.sh` locks the SM clock at **1200 MHz** (`CLOCK=...` to change). At 2040 MHz the L4 hits its 72 W power
cap, which makes timings jump. It logs clocks and power to `<out>/gpu_log.csv` and prints the results tarball path.
Do not run anything else on the GPU at the same time.

## Results
See [`results/l4_part3/README.md`](results/l4_part3/README.md). In short:
- 880/880 correctness checks and 364/364 edge-case checks passed.
- V3 is 1.3–2.0× faster than V0 and up to 5.3× faster than V1 (warm).
- V3 matches V2 but does not beat it. The fixes suggested by the WS2 profile make a Triton kernel slower.
