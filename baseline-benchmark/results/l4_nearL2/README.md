# Rerun near the L2 limit (NVIDIA L4, 2026-11-21)

## Why this rerun
- In the main run (`../l4_20261121`), a few warm-cache shapes gave timings that jumped between two levels, and those
  are the shapes behind the headline ratios. Example: V1 at H=2048, T=4096 had min 144 µs but median 173 µs, so the
  slowdown read 1.36× by minimum and 1.59× by median.
- This rerun repeats the shapes around the L2 limit with **4× more samples** and logs the GPU clock and power.
  The aim is to check that the ratios are stable and to find out why some timings jump.

## What was run
| | |
|---|---|
| GPU | NVIDIA L4 (sm_89, 58 SMs, 48 MB L2), Google Cloud g2 VM |
| Software | vLLM 0.34.0, PyTorch 2.14.0+cu130, Triton 3.8.0, Python 3.13 (see `env.json`) |
| Script | `bench_rmsnorm_int8.py`, same version and tolerances as the main run |
| Shapes | BF16; H = 2048, 4096; T = 1024, 2048, 4096, 8192; residual off and on |
| Variants | V0 (vLLM unfused), V1 (vLLM fused CUDA), V2 (built-in Triton kernel, see main-run caveats) |
| Timing | CUDA graphs + CUDA events; warm: **200** samples (main run: 50); cold: 100 single calls with an 8×L2 flush |
| Clock | SM clock locked at 2040 MHz (`LOCK_CLOCKS=1`); nvidia-smi logged every 0.5 s → `gpu_log.csv` |
| Command | `LOCK_CLOCKS=1 bash run_l4.sh --dtypes bf16 --hidden 2048 4096 --tokens 1024 2048 4096 8192 --reps 200 --out-dir results_l4_nearL2` |

Correctness: **96/96** cross-checks passed.

## Results
Regenerate the tables with `python summarize_ws1.py` (writes `../ws1_summary.md`).

**Headline ratios are stable** (BF16, residual on, warm; V1/V0 by median, with p10 in brackets):

| Shape | Main run | This rerun |
|---|---|---|
| H=2048, T=1024 | 2.31× slower | 2.12× (2.49×) slower |
| H=2048, T=2048 | 1.83× slower | 2.29× (2.38×) slower |
| H=2048, T=4096 | 1.59× slower | 1.56× (1.57×) slower |
| H=2048, T=8192 | 0.68× (V1 faster) | 0.67× (0.67×) |
| H=4096, T=1024 | 1.46× slower | 1.54× (1.55×) slower |
| H=4096, T=2048 | 1.15× slower | 1.08× (1.10×), borderline |
| H=4096, T=4096 | 0.69× (V1 faster) | 0.68× (0.68×) |

**Reproducibility:** for V1/V0, **31 of 32** configurations are on the same side of 1 in both runs. For V2/V0 it
is 32 of 32. The median change in the ratio is 2–3%. The only V1 flip is a tie (H=2048, T=1024, cold, no residual:
1.02 vs 1.00), not a real reversal.

## Clock and power: why timings jumped
- **The L4 hit its 72 W power cap.** Even with the clock "locked" at 2040 MHz, the software power cap was active in
  **78% of busy samples**, and the SM clock dropped as low as **780 MHz**. Power peaked at 73.7 W, with no thermal
  slowdown. `nvidia-smi -lgc` does not override the power cap.
- **Follow-up at a steady clock** (Part 3 run, `results_l4_part3`, SM clock locked at 1200 MHz, power cap active in
  only 2% of samples):
  - **V0 and V1:** the spread (p90 − p10)/median fell to **0.1–1.5%**. Their jumps in the 2040 MHz runs came from
    **power throttling**.
  - **Triton kernels (V2, V3):** still jumped between ~45 µs and ~60 µs, but only at the two shapes where each
    activation tensor is 16 MB (H=2048/T=4096, H=4096/T=2048). There the fused kernel's working set (~40 MB) sits
    just under the 48 MB L2. That fits the cache-residency hypothesis (H2), and is to be confirmed with L2 hit rates.
- **For reporting:** quote the median and p10 together at the shapes flagged in the spread table of
  `ws1_summary.md`. Use a clock below the power limit (e.g., 1200 MHz) for future timing runs.

## Files
| File | Content |
|---|---|
| `bench_results.csv` | one row per variant × shape × residual × cache mode (median/mean/std/min/p10/p90 µs, bandwidth, speedup) |
| `correctness.csv` | every cross-check comparison (vs FP32 reference and pairwise) |
| `env.json` | GPU, versions, arguments, tolerances |
| `gpu_log.csv` | nvidia-smi every 0.5 s: SM clock, power, temperature, throttle reasons (0x4 = software power cap) |
| `plots/` | latency and speedup vs tokens, per residual mode and cache mode |
