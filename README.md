# Why Is Fused RMSNorm + INT8 Quantization Slow?

**Diagnosing a kernel-fusion gap in vLLM across GPU generations**

COMS 6998 High-Performance Machine Learning · Columbia University · Fall 2026

| Team member | UNI |
|---|---|
| Nrushad Joshi | naj2139 |
| Akriti Agarwal | aa5807 |
| Anisha Gupta | ag5213 |
| Toshini Agrawal | ta2828 |

---

## Overview

We study W8A8 inference in vLLM: INT8 weights, plus INT8 activations quantized per token at runtime. In this setup, every transformer layer runs **RMSNorm** and then **INT8 quantization** before the attention (QKV) projection and before the MLP gate/up projections. Both steps are small and limited by memory bandwidth. Fusing them into one kernel should remove the intermediate normalized tensor and cut global-memory traffic, while still producing the residual update, the INT8 output and the per-token scales.

vLLM currently has two problems here:

1. **No INT8 fusion.** vLLM's `torch.compile` fusion pass has RMSNorm + quant patterns for FP8 and NVFP4, but none for INT8. W8A8 INT8 models never get this fusion.
2. **The existing fused kernel is slower at prefill.** vLLM's fused CUDA kernel `rms_norm_dynamic_per_token_quant` beats the unfused pair at decode sizes. At prefill sizes it is up to **2× slower**: 48.6 µs vs. 24.6 µs at 4,096 tokens with hidden size 2048 on an RTX 4090 ([vLLM #60499](https://github.com/vllm-project/vllm/issues/60499)). Nobody has explained why.

This project takes a profile-first approach. We aim to:

- **Bound** the possible end-to-end gain with Amdahl's Law.
- **Diagnose** the slowdown with Nsight Compute.
- **Compare** four implementations on an NVIDIA L4 GPU on Google Cloud (with other GPU generations if available), reporting kernel-level and end-to-end results separately.

The main deliverable is a **per-GPU explanation of where the time and bytes go**, which can guide the choice and tuning of INT8 fusion kernels in vLLM.

## Research question

Why is the fused RMSNorm + INT8 kernel slow at large token counts? Does this hold across GPU generations? How do vLLM's fused CUDA kernel, an existing Triton kernel and our tuned Triton variants compare, both at the kernel level and end to end (including compiler-generated fusion where feasible)?

## Hypotheses (to be tested, not assumed)

- **H1 (data movement):** At large token counts, the fused kernel is limited by how it moves data (scalar or poorly coalesced loads, extra passes per row), not by arithmetic or reduction latency. Low bandwidth alone does not prove this, so we confirm it with stall analysis and ablations.
- **H2 (cache residency):** In the unfused path, a ~16 MB intermediate tensor passes between the two kernels. It can stay in L2 on the L4 (48 MB L2), but would not on a GPU with a small L2 such as the A6000 (6 MB). Whether it does shifts the token count where fusion breaks even. We test this with measured L2 hit rates and break-even points under both cache protocols.
- **H3 (architecture dependence):** Differences in memory access patterns, reduction scheduling and hardware resource use make the break-even points depend on the GPU architecture. Targeted Triton optimizations can move them.
- **H4 (Amdahl):** End-to-end gains will be much smaller than kernel-level gains, because normalization and quantization are only a small part of total inference time.
- **Correctness:** We measure errors in the per-token scales, the updated residual and the dequantized output against an FP32 reference, compare with the unfused INT8 path, and check WikiText-2 perplexity.

## Implementations compared

| ID | Implementation |
|---|---|
| V0 | Unfused pair: `fused_add_rms_norm`, then `scaled_int8_quant` |
| V1 | vLLM's fused CUDA kernel (`rms_norm_dynamic_per_token_quant`) |
| V2 | The issue author's Triton kernel ([AshrithaG/int8-linear](https://github.com/AshrithaG/int8-linear)) |
| V3 | Our tuned Triton variants (this repo) |

## Plan

| Step | What we do | Tools | Output |
|---|---|---|---|
| 0 | Reproduce the issue's microbenchmark on the L4 (sm_89, same architecture as the RTX 4090) for 1–16,384 tokens, hidden sizes 2048 and 4096, with cold and warm L2 | PyTorch, CUDA graphs | Confirms or refutes the 2× slowdown on datacenter hardware |
| 1 | Profile vLLM prefill and decode with Qwen3-1.7B W8A8 and measure the time share of norm + quant | PyTorch Profiler, Nsight Systems | Time breakdown and Amdahl bound on end-to-end gain |
| 2 | Diagnose the fused kernel vs. the unfused pair: DRAM vs. L2 traffic, load width, instructions, occupancy, stalls; ablations that isolate each cause | Nsight Compute, roofline | Evidence for or against each cause, including a possibly missing vectorized read ([vLLM PR #26234](https://github.com/vllm-project/vllm/pull/26234)) |
| 3 | Write fused Triton variants that vary row/block scheduling, warp count, reduction strategy and register use, tuned per GPU | Triton | Optimized configurations with measured tradeoffs |
| 4 | Compare V0–V3 on the L4 (other GPUs if available) | Same harness on each GPU | Kernel time and three bandwidth measures vs. token count, per GPU |
| 5 | Patch the best variants into vLLM; measure prefill throughput, decode latency and WikiText-2 perplexity | vLLM, W&B | End-to-end speedup vs. the Amdahl bound |

## Current focus (midpoint)

Work toward the midpoint report is split into four workstreams:

1. **Baseline implementation and benchmarking:** run the existing fused and unfused implementations, benchmark latency and produce comparison plots.
2. **GPU profiling and bottleneck analysis:** analyze memory traffic, occupancy and performance bottlenecks.
3. **Custom Triton kernel and optimization:** implement our fused kernel (V3), validate its correctness and test first optimizations.
4. **vLLM model-level evaluation:** run Qwen3 inference benchmarks, measure how much RMSNorm and quantization affect end-to-end time, and identify where a fused kernel can be integrated.

## Methodology

We follow a **benchmark → profile → change → re-benchmark** loop and change one thing at a time.

**Kernel-level and end-to-end numbers are always reported and labeled separately.** A kernel speedup is never presented as an end-to-end speedup.

### Timing protocols (reported separately)

- **Warm cache:** median time over CUDA-graph replays after warm-up.
- **Cold cache:** rotate through complete working sets (input, residual, norm weights, INT8 output and scales) whose total size is far larger than L2. No eviction work happens inside the timed region.

We confirm cache state with measured L2 hit rates and DRAM traffic rather than inferring it from buffer sizes.

### Metrics (logged to Weights & Biases)

- Time per call (µs)
- Three bandwidth figures, kept separate:
  - minimum required bytes ÷ time (input, residual in and out, norm weights, INT8 output, scales)
  - actual DRAM bytes ÷ time
  - L2 traffic and hit rate
- Occupancy and warp-stall reasons (Nsight Compute)
- Prefill tokens/s and time per output token
- INT8 mismatch rate and WikiText-2 perplexity
- BF16 vs. W8A8 INT8 inference speed and perplexity, and fusion speedups measured against the unfused INT8 baseline

## Setup

- **Hardware:** NVIDIA L4 on Google Cloud Platform (Ada, sm_89; 24 GB GDDR6, ~300 GB/s, 48 MB L2, 58 SMs, 72 W). It has the same architecture as the RTX 4090 in vLLM #60499, so the kernel code paths match, but its memory bandwidth is about 3× lower. We therefore compare fused/unfused ratios and break-even token counts with the issue, not absolute µs. Roofline and %-of-peak figures use the L4's own peak bandwidth. The L4 is power-limited (72 W), so we record SM clocks and power during runs. All variants are compared on the same GPU. Other generations (e.g. A6000 sm_86, L40S sm_89, H100 sm_90) are added for cross-GPU comparison if available.
- **Software:** vLLM (pinned version), PyTorch 2, Triton, CUDA, PyTorch Profiler, Nsight Systems, Nsight Compute, Weights & Biases.
- **Models:** the Qwen3-1.7B W8A8 INT8 checkpoint used in vLLM #60499 ([`nytopop/Qwen3-1.7B.w8a8`](https://huggingface.co/nytopop/Qwen3-1.7B.w8a8)), so our results are directly comparable, and the BF16 base model ([`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)) for the precision comparison. Secondary, if compatible and time allows: an 8B-class W8A8 model with hidden size 4096 (e.g. Llama-3.1-8B-Instruct-quantized.w8a8).
- **Workloads:** offline, non-concurrent runs.
  - Prefill: 8 prompts of 512 tokens, and 1 prompt of 2,048 tokens.
  - Decode: batch sizes 1, 8 and 64, with 128 output tokens.
  - We make no claims about concurrent serving.
- **Accuracy:** WikiText-2 perplexity.

## Challenges

- **Numerical correctness:** INT8 rounding and clamping differ slightly between implementations (the issue saw 0.2–1.1% of values differ by ≤2 steps). One INT8 step also means a different real error for each token, depending on its scale. So we check scales and dequantized error, not only the INT8 codes.
- **Fair baselines:** Every implementation must do identical work: same inputs, residual update, INT8 output, scales and in-place behavior. In-place kernels need fresh inputs on every call, which can bias timings, so we re-validate the issue's timing method instead of reusing it.
- **Cache-aware timing:** Repeated calls on the same input keep data in L2 (the L4's 48 MB L2 can hold a whole 16 MB activation), which inflates bandwidth. Flushing L2 inside CUDA-graph replay would distort the workload, so we use the two separate timing protocols above.
- **Integration:** vLLM has no INT8 fusion pattern, so end-to-end runs require patching the model's norm-to-linear path. This is the riskiest step, so we prototype it in weeks 2–3.

## Planned demo

- Kernel time vs. token count for each implementation and GPU
- Roofline and bandwidth plots
- Nsight Compute evidence for the slowdown, with vLLM's kernel and ours side by side
- End-to-end results compared against the Amdahl bound

## Limits of this study

Results cover one operation, one serving system, one main model and up to three NVIDIA GPUs. Related vLLM work ([#60224](https://github.com/vllm-project/vllm/pull/60224), [#45364](https://github.com/vllm-project/vllm/pull/45364)) and a similar planned kernel ([TileOPs #2392](https://github.com/tile-ai/TileOPs/issues/2392)) may change the baseline. We expect end-to-end gains to be small. The diagnosis stands even if our kernel does not win.

## Repository structure

```
.
├── README.md
├── .gitignore
└── gpu profling - bottleneck/    WS2: Nsight profiling toolkit
    ├── README.md                 how to run the sweep and read the tables
    ├── run_all.sh                end-to-end driver
    ├── slurm/job.sbatch          Slurm wrapper
    ├── scripts/                  counter check, nsys, ncu sweep
    ├── harness/profile_target.py V0/V1 calls (profile, verify, bench)
    ├── analysis/                 parse reports and write the bottleneck note
    ├── reports/                  ncu/ and nsys/ outputs (gitignored)
    ├── tables/                   generated comparison tables (gitignored)
    └── logs/                     per-step logs (gitignored)
```

Each workstream lives in its own directory, with source, job scripts and generated results kept in separate folders. Folders for the other workstreams will be added as their code lands.

### Running WS2 (profiling)

WS2 profiles V0 against V1 with Nsight Systems and Nsight Compute. Setup, the sweep and how to read the tables are in [gpu profling - bottleneck/README.md](<gpu profling - bottleneck/README.md>). On the GPU machine:

```bash
cd "gpu profling - bottleneck"
./run_all.sh
```

## Key references

- vLLM Issue [#60499](https://github.com/vllm-project/vllm/issues/60499): RMSNorm and INT8 quant are never fused for W8A8 models; the fused kernel is up to 2× slower at prefill (sm_89)
- vLLM [fusion `torch.compile` passes](https://docs.vllm.ai/en/latest/design/fusions/)
- vLLM PRs [#26234](https://github.com/vllm-project/vllm/pull/26234) (vectorized RMSNorm variance read), [#38026](https://github.com/vllm-project/vllm/pull/38026) (Inductor quant fusion), [#45364](https://github.com/vllm-project/vllm/pull/45364) (manual RMSNorm-quant fusion), [#60224](https://github.com/vllm-project/vllm/pull/60224) (vectorized RMSNorm-quant kernel)
- [AshrithaG/int8-linear](https://github.com/AshrithaG/int8-linear): Triton W8A8 INT8 kernels benchmarked against vLLM
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019
- Xiao et al., *SmoothQuant*, ICML 2023
- Dettmers et al., *LLM.int8()*, NeurIPS 2022
- Yao et al., *ZeroQuant*, NeurIPS 2022
- Ivanov et al., *Data Movement Is All You Need*, MLSys 2021
- Lin et al., *QServe*, MLSys 2025
- Kwon et al., *PagedAttention (vLLM)*, SOSP 2023
- Tillet et al., *Triton*, MAPL 2019
- Williams et al., *Roofline*, CACM 2009
