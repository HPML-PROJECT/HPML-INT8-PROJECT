# WS1: Baseline implementation and benchmarking

Benchmarks the existing RMSNorm + dynamic per-token INT8 quantization implementations, checks that they produce the same outputs, and measures their latency on the NVIDIA L4.

## Variants

| ID | Implementation | Kernels per call |
|---|---|---|
| **V0** | vLLM unfused: `torch.ops._C.rms_norm` (or `fused_add_rms_norm` with residual) → `torch.ops._C.dynamic_scaled_int8_quant` | 2 |
| **V1** | vLLM fused CUDA kernel: `torch.ops._C.rms_norm_dynamic_per_token_quant` | 1 |
| **V2** | The team's built-in Triton kernel (in `bench_rmsnorm_int8.py`): one row per program, `num_warps` = 4 / 8 / 16 by hidden size. An external kernel can replace it with `--v2-impl module:function` | 1 |

All variants share one interface: inputs `x [T, H]`, weight `w [H]`, optional residual `[T, H]`; outputs INT8 `q [T, H]` and FP32 scales `[T, 1]`. With a residual, `residual` is updated in place to `x + residual` (vLLM semantics).

## What the script does

1. **Inputs:** deterministic, generated on the CPU with a per-shape seed (base seed 1234), so they are bit-identical on every machine: `x ~ N(0,1)`, `w = 1 + 0.1·N(0,1)`, `residual ~ N(0,1)`.
2. **Cross-check:** every variant against an FP32 PyTorch reference (with vLLM's rounding points), and pairwise (V1 vs. V0, V2 vs. V0, V2 vs. V1).
3. **Timing** (CUDA events, calls captured in CUDA graphs):
   - **warm:** a graph of N back-to-back calls (N chosen for about 2 ms of GPU time, at most 500), replayed `--reps` times after warm-up; latency = replay time / N. Inputs are restored before each replay, outside the timed region.
   - **cold:** before every call, a buffer of at least 8 × L2 (384 MB on the L4) is written to evict L2; each call is timed on its own, `--cold-iters` times.
4. **Outputs:** `bench_results.csv`, `correctness.csv`, `env.json` and plots (via `plot_results.py`).

## Running it

```bash
pip install -r requirements.txt
python bench_rmsnorm_int8.py --quick --out-dir results/quick          # smoke test (~1 min)
python bench_rmsnorm_int8.py --residual both --out-dir results/<run>  # full grid, as in results/l4_20261121
python bench_rmsnorm_int8.py --plot-only --out-dir results/<run>      # re-plot an existing CSV
```

The default grid is T ∈ {1, 2, 4, …, 16384}, H ∈ {2048, 4096, 5120, 8192}, FP16 and BF16. `--residual` defaults to `off`; the main run used `both`. The script exits with code 2 if any correctness check fails.

## Shared conventions

| | |
|---|---|
| eps | 1e-6 |
| Seed | 1234 |
| INT8 vs. INT8 | at most 3 steps apart, on at most 15% of elements |
| Scales | rtol 1.6e-2, atol 1e-6 |
| Dequantized output vs. FP32 reference | within 2.5 INT8 steps (× that token's scale) |
| Updated residual | atol 1e-2, rtol 1e-3 |

**`bench_results.csv` columns:** `variant, impl, gpu, dtype, num_tokens, hidden_size, residual, cache_mode, timing_method, inner_iters, samples, median_us, mean_us, std_us, min_us, p10_us, p90_us, eff_bandwidth_GBps, speedup_vs_v0, check_passed, torch_version, vllm_version, triton_version, timestamp`

- `eff_bandwidth_GBps` = minimum bytes the operation must move (x, w, residual read and write, INT8 output, scales) ÷ median time.
- `speedup_vs_v0` = V0 median ÷ this variant's median (>1 means faster than V0).

**`correctness.csv` columns:** `variant, compared_to, dtype, num_tokens, hidden_size, residual, q_max_abs_diff, q_mismatch_frac, scale_max_rel_diff, dequant_max_err_lsb, residual_max_abs_diff, passed, note`

## Results

- [results/l4_20261121/](results/l4_20261121/README.md): main run on the NVIDIA L4, full grid, warm and cold.
