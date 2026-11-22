# Part 4: vLLM model-level evaluation (Qwen3-1.7B, NVIDIA L4, midpoint)

## Setup
- **Models:** `nytopop/Qwen3-1.7B.w8a8` (W8A8 INT8, detected as `compressed-tensors`) and `Qwen/Qwen3-1.7B` (BF16).
- **Engine:** stock vLLM 0.34.0, normal mode (torch.compile + CUDA graphs). An extra W8A8 profile was taken in eager mode.
- **Software and GPU:** PyTorch 2.14.0+cu130, the same environment as Parts 1–3. One NVIDIA L4. SM clock locked at
  1200 MHz; power-capped in 13% of busy samples, minimum 915 MHz, no thermal throttling.
- **Workloads:**
  - Prefill: 8 × 512 and 1 × 2048 tokens.
  - Decode: batch 1, 8 and 64; 128-token prompts; exactly 128 output tokens. Latency per output token (TPOT) is
    computed as (t₁₂₈ − t₁)/127.
  - 3 warm-ups and 5 repeats (medians reported). Prefix caching off.
- **Perplexity:** WikiText-2 raw test, 146 non-overlapping 2048-token windows (298,862 tokens scored), computed through
  the same vLLM kernels that are benchmarked.
- **Profile:** PyTorch profiler, GPU kernels only. Decode time = (prefill + 32 decode steps) − prefill.

## 1. W8A8 vs BF16

| Workload | BF16 | W8A8 | W8A8 gain |
|---|---|---|---|
| Prefill 8×512 (tok/s) | 15,329 | 20,546 | **1.34×** |
| Prefill 1×2048 (tok/s) | 14,589 | 18,919 | **1.30×** |
| Decode b=1 (TPOT ms) | 15.03 | 10.72 | **1.40×** |
| Decode b=8 (TPOT ms) | 16.50 | 11.65 | **1.42×** |
| Decode b=64 (TPOT ms) | 23.20 | 19.32 | **1.20×** |
| WikiText-2 PPL | 16.70 | 16.40 | −0.30 (−1.8%) |

Repeat-to-repeat variation is small: under 0.1% for W8A8, and up to 3.5% for BF16 prefill.

## 2. Where the time goes (W8A8, share of GPU kernel time, normal mode)

| Phase | RMSNorm | INT8 quant | norm + quant | **fusable** (norm → quant) | GEMM | attention |
|---|---|---|---|---|---|---|
| Prefill 8×512 | 8.8% | 6.4% | 15.2% | **7.6%** | 74.1% | 1.5% |
| Decode b=1 | 3.5% | 3.0% | 6.5% | **2.9%** | 90.9% | 0.9% |
| Decode b=8 | 3.4% | 2.8% | 6.2% | **2.9%** | 90.1% | 0.8% |
| Decode b=64 | 2.4% | 2.1% | 4.5% | **2.1%** | 91.0% | 0.6% |

- **Only 2 of the 4 quantizations per layer are fusable with RMSNorm:** the inputs to qkv_proj and gate_up_proj. The
  o_proj input follows attention; the down_proj input follows SiLU·mul.
- **vLLM still does not fuse INT8 norm + quant.** The log shows the norm+quant fusion pass is enabled
  (`Enabled custom fusions: norm_quant, act_quant`). Yet prefill runs **112 separate**
  `vllm::dynamic_scaled_int8_quant_kernel` launches: 4 per layer × 28 layers, one before every INT8 GEMM.
  torch.compile only merges residual add + RMSNorm into a Triton kernel. This confirms the gap the project targets,
  on the current vLLM.
- **Methodology note.** Compiled kernel names include every nearby op, for example
  `triton_red_fused_dynamic_scaled_int8_quant_fused_add_rms_norm`. Inductor names kernels after the FX nodes in their
  region, so the `dynamic_scaled_int8_quant` part plausibly comes from allocating the quant op's output; we did not
  inspect the generated code. What the results do show is that the kernel does not quantize: the separate CUDA quant
  kernel still runs 112 times, unchanged. Our first
  quick run therefore mislabelled them as "already fused" and reported 0% fusable time. The classifier now counts a
  compiled kernel as quantization only if it contains the rounding/clamping math. The self-test covers these names.

## 3. Amdahl bound (end-to-end speedup from fusing norm + quant)

| Phase | Fusable share f | Bandwidth-model bound (s = 1.57×) | **Measured V3 vs V0 kernel speedup*** | Fused region free (s = ∞) |
|---|---|---|---|---|
| Prefill 8×512 (4,096 tokens, H=2048) | 7.6% | 1.028× | **1.017–1.039×** (s = 1.28× cold – 1.99× warm) | 1.082× |
| Decode b=1–64 (T = 1–64) | 2.1–2.9% | 1.008–1.011× | **~1.010–1.014×** (s ≈ 1.5–1.96×) | 1.022–1.030× |

\* From the Part 1/3 kernel benchmarks at the matching token counts (H=2048). **Caveat:** those speedups are against
vLLM's *CUDA* norm and quant kernels (V0). In the default compiled model, the norm half is an Inductor-generated
Triton kernel (residual add + RMSNorm) followed by vLLM's CUDA quant kernel. So the real end-to-end gain depends on
how fast Inductor's norm kernel already is, and these columns are estimates, not measurements.

**H4 is supported.** Even a perfect fused kernel can speed up end-to-end inference by at most ~8% (prefill) and ~2–3%
(decode) on this model and GPU. With the measured kernel speedups, it is about **1.7–3.9% (prefill)** and
**1.0–1.4% (decode)**. Kernel-level
gains of 1.5–2× (Parts 1 and 3) shrink to a few percent end to end, because INT8 GEMMs take 74–91% of the time.

## 4. Integration (see [`INTEGRATION_PROPOSAL.md`](INTEGRATION_PROPOSAL.md), [`results/integration_map.md`](results/integration_map.md))
- The hooks were found in the installed vLLM 0.34.0: the RMSNorm layer, `scaled_int8_quant`, the
  `CompressedTensorsW8A8Int8` scheme with `CutlassInt8ScaledMMLinearKernel`, the fusion-pass modules, custom-op
  registration and plugin entry points.
- **Proposed approach:**
  - **A, a plugin with layer-level fusion,** for the first end-to-end measurement. The fused op returns (q, scales)
    and the W8A8 linear layer skips its own quantization.
  - **B, an INT8 pattern added to the existing fusion pass,** as the path that could go upstream.
- Validation: the kernel-count check (2 fewer launches per layer), PPL, and prefill/decode speed with and without it.

## Caveats
- **W8A8 PPL is lower than BF16 (16.40 vs 16.70).** That is unexpected, since quantization usually adds error. The
  evaluation itself is consistent: same tokens, same 146 windows, same scoring. The most likely cause is that **the
  two checkpoints are not the same model**, for example the W8A8 one quantized from `Qwen3-1.7B-Base` rather than the
  post-trained `Qwen3-1.7B`. Calibration data close to WikiText is another possibility. **Check `base_model` on the
  W8A8 checkpoint's Hugging Face page before claiming "no accuracy loss".** Until then, report the two PPLs without
  attributing the difference to quantization.
- **The end-to-end baseline is not WS1's V0.** In compiled mode, vLLM's CUDA RMSNorm kernel is replaced by an
  Inductor Triton kernel (residual add + RMSNorm). Only the INT8 quant kernel is the same as in V0 (112 launches per
  forward pass, 4 per layer). Kernel-level speedups from WS1/WS3 therefore do not transfer one-to-one. The eager-mode
  profile (`results/profile_w8a8_eager.json`) uses the plain vLLM kernels, as in V0.
- **Only the 8×512 prefill was profiled.** The 1×2048 prefill was benchmarked but not broken down.
- The busy/wall ratio slightly above 1 (decode b=64: 1.00–1.04) is profiler rounding.
- One GPU (L4), one model (1.7B), offline (non-concurrent) runs, clock locked at 1200 MHz. About 10–13% of
  clock-log samples show the power cap, mostly during the perplexity runs (~975 MHz). The timing benchmarks ran at a
  steady 1200 MHz.
- TPOT and decode profiles use subtraction, so small noise is possible at batch 1.
- The model-level gain is a bound, not a measurement: V3 is not yet integrated (next step).

## Files (in `results/`)
| File | Content |
|---|---|
| `bench_w8a8.json`, `bench_bf16.json` | prefill throughput and decode TPOT, all repeats |
| `ppl_w8a8.json`, `ppl_bf16.json` | WikiText-2 perplexity |
| `profile_*.json`, `profile_*_kernels.csv` | GPU kernel-time breakdown per phase (normal and eager mode) |
| `amdahl.json`, `amdahl.md` | fusable share and Amdahl bounds |
| `REPORT.md` | auto-generated report from the run |
| `integration_map.md` | vLLM hooks found for integration |
| `env.json`, `clock_setting.json`, `gpu_log.csv` | versions, clock lock, nvidia-smi log |

Code: [`../vllm-evaluation/`](../vllm-evaluation/) (run with `run_l4_vm.sh`).
