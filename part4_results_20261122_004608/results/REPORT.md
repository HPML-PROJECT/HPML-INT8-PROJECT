# End-to-end impact of RMSNorm + quantization: Qwen3-1.7B on NVIDIA L4

- GPU: NVIDIA L4 (cc 8.9), torch 2.14.0+cu130, CUDA 13.0, vLLM 0.34.0
- nvidia-smi snapshot: `NVIDIA L4, 620.70.06, 1200 MHz, 2040 MHz, 6251 MHz, 22.42 W, 72.00 W, 48.72, 3 MiB, 23034 MiB`

## 1. Prefill throughput (tokens/s, median; higher is better)

| workload | BF16 | W8A8 | W8A8 / BF16 |
|---|---|---|---|
| 8x512 | 15329 | 20546 | 1.34x |
| 1x2048 | 14589 | 18919 | 1.30x |

## 2. Decode latency (128 output tokens)

| batch | BF16 TPOT ms | W8A8 TPOT ms | W8A8/BF16 speedup | BF16 e2e s | W8A8 e2e s |
|---|---|---|---|---|---|
| 1 | 15.03 | 10.72 | 1.40x | 1.930 | 1.378 |
| 8 | 16.50 | 11.65 | 1.42x | 2.163 | 1.535 |
| 64 | 23.20 | 19.32 | 1.20x | 3.476 | 2.863 |

## 3. WikiText-2 perplexity (non-overlapping windows)

| model | PPL | tokens scored | windows | partial run |
|---|---|---|---|---|
| bf16 | 16.6974 | 298862 | 146 | False |
| w8a8 | 16.4015 | 298862 | 146 | False |

W8A8 - BF16 = -0.2958 PPL (-1.77%)

## 4. Model execution breakdown (share of GPU kernel time)


**w8a8** (mode `graphs`, quantization `compressed-tensors`)

| phase | norm | quant | fused n+q | norm+quant | fusable region | GEMM | attention | other | GPU busy / wall |
|---|---|---|---|---|---|---|---|---|---|
| prefill_8x512 | 8.8% | 6.4% | 0.0% | **15.2%** | 7.6% | 74.1% | 1.5% | 9.1% | 0.94 |
| decode_b1 | 3.5% | 3.0% | 0.0% | **6.5%** | 2.9% | 90.9% | 0.9% | 1.7% | 0.99 |
| decode_b8 | 3.4% | 2.8% | 0.0% | **6.2%** | 2.9% | 90.1% | 0.8% | 2.9% | 0.99 |
| decode_b64 | 2.4% | 2.1% | 0.0% | **4.5%** | 2.1% | 91.0% | 0.6% | 3.8% | 1.00 |

**bf16** (mode `graphs`, quantization `None`)

| phase | norm | quant | fused n+q | norm+quant | fusable region | GEMM | attention | other | GPU busy / wall |
|---|---|---|---|---|---|---|---|---|---|
| prefill_8x512 | 6.2% | 0.0% | 0.0% | **6.2%** | 0.0% | 86.0% | 1.2% | 6.7% | 0.95 |
| decode_b1 | 2.5% | 0.0% | 0.0% | **2.5%** | 0.0% | 95.7% | 0.6% | 1.2% | 1.00 |
| decode_b8 | 2.5% | 0.0% | 0.0% | **2.5%** | 0.0% | 94.8% | 0.6% | 2.1% | 1.00 |
| decode_b64 | 2.0% | 0.0% | 0.0% | **2.0%** | 0.0% | 94.4% | 0.5% | 3.1% | 1.01 |

## 5. Amdahl bound for fusing norm + quant (W8A8)

| phase | f_all (GPU) | f_fusable (GPU) | max speedup, fusable (s=inf) | speedup, fusable at s=11/7 (bandwidth bound) |
|---|---|---|---|---|
| prefill_8x512 | 15.21% | 7.57% | 1.0819x | 1.0283x |
| decode_b1 | 6.52% | 2.92% | 1.0301x | 1.0107x |
| decode_b8 | 6.24% | 2.91% | 1.0299x | 1.0107x |
| decode_b64 | 4.53% | 2.11% | 1.0216x | 1.0077x |

Full tables: `amdahl.md`. Kernel-level speedups are not end-to-end speedups.

## 6. Integration
See `INTEGRATION_PROPOSAL.md` and `integration_map.md` (hooks found in the installed vLLM).

