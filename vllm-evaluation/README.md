# End-to-end impact of RMSNorm + quantization on real inference (Qwen3-1.7B, NVIDIA L4)

**Objective:** understand how much of real vLLM inference time goes to RMSNorm and activation quantization, and what
that implies for a fused Triton kernel (V3).
**Hardware:** one **NVIDIA L4** (Ada, sm_89, 24 GB, ~300 GB/s, 48 MB L2). Every script checks for an L4 and exits otherwise
(override: `--allow-non-l4` or `ALLOW_NON_L4=1`; the check matches "L4", not "L40S").

| Task | Covered by |
|---|---|
| Stock vLLM W8A8 (`nytopop/Qwen3-1.7B.w8a8`) and BF16 (`Qwen/Qwen3-1.7B`) inference | `common.py` (`make_llm`), `setup_env.sh` |
| Fixed workloads: prefill 8x512 + 1x2048; decode batch 1/8/64 x 128 tokens | `bench_perf.py` |
| Prefill throughput and decode latency (both precisions) | `bench_perf.py` -> `results/bench_<tag>.json` |
| WikiText-2 perplexity (both precisions) | `eval_ppl.py` -> `results/ppl_<tag>.json` |
| Locate RMSNorm + quantization kernels, time share | `profile_breakdown.py`, `analysis.py` |
| Amdahl bound | `amdahl.py` -> `results/amdahl.md` |
| Where V3 could be integrated | `inspect_integration.py`, `INTEGRATION_PROPOSAL.md` |
| Everything in one report | `make_report.py` -> `results/REPORT.md` |


## 0. On the team's Google Cloud L4 VM (recommended)
Reuse the Part 1 environment (`~/rmsnorm_int8_bench/.venv`, vLLM 0.34.0) so Parts 1–4 use the same vLLM and no second
multi-GB venv fills the disk:
```bash
cd ~/vllm-evaluation
QUICK=1 bash run_l4_vm.sh     # smoke test; check results/REPORT.md
bash run_l4_vm.sh             # full run; prints a results bundle path at the end
```
It locks the SM clock at 1200 MHz (the L4 throttles at its 72 W power cap at 2040 MHz; `CLOCK=0` to skip) and logs
clocks/power to `results/gpu_log.csv`. State the clock setting next to every number in the report.
`setup_env.sh` (below) is only for a fresh machine without the Part 1 environment.

## 1. Files
| File | Purpose |
|---|---|
| `setup_env.sh`, `requirements.txt` | create `.venv`, install vLLM (pin with `VLLM_VERSION`) and deps |
| `run_all.sh` | **driver**: all steps, logs, resumable |
| `vllm_eval/common.py` | L4 guard, prompts, vLLM factory, stats, JSON helpers |
| `vllm_eval/analysis.py` | kernel classifier, fusable-region detection, Amdahl (pure Python) |
| `vllm_eval/bench/` | `env_info`, prefill/decode benchmark, WikiText-2 perplexity |
| `vllm_eval/profile/` | torch.profiler GPU-kernel breakdown (prefill, decode b1/8/64) |
| `vllm_eval/reporting/` | Amdahl tables, aggregated report, offline self-test |
| `vllm_eval/integration/` | maps the integration hooks in the installed vLLM |
| `INTEGRATION_PROPOSAL.md` | proposed approach for V3 |

## 2. Setup (once)
```bash
# On the L4 machine (e.g. a Google Cloud g2-standard-8 VM with 1x L4 and the NVIDIA driver installed; need ~30 GB disk
# for the venv + model downloads and internet access to Hugging Face)
cd vllm-evaluation
VLLM_VERSION=<team-pinned-version> ./setup_env.sh      # leave empty only for a quick try
source .venv/bin/activate
# optional: huggingface-cli login
python -m vllm_eval.reporting.selftest                  # must print: selftest OK
```
`setup_env.sh` prints whether `rms_norm`, `fused_add_rms_norm`, `scaled_int8_quant` and
`rms_norm_dynamic_per_token_quant` exist in `vllm._custom_ops`.

## 3. Run
```bash
source .venv/bin/activate
QUICK=1 ./run_all.sh        # smoke test (few repeats, 8 ppl windows, 8 decode steps): validates the whole pipeline
./run_all.sh                # full run; results in results/, logs in logs/
```
Selected steps / redo: `ONLY="bench_w8a8 report" ./run_all.sh`, `FORCE=1 ./run_all.sh`. Extra flags for every script:
`EXTRA="--gpu-mem 0.8"`. Steps: `env selftest bench_w8a8 bench_bf16 ppl_w8a8 ppl_bf16 profile_w8a8 profile_bf16
profile_w8a8_eager amdahl integration report`. Each model runs in its own process so GPU memory is freed in between.

## 4. Step by step (what each does and how to read it)

### 4.1 Pipeline sanity: `python -m vllm_eval.bench.env_info`, then a one-off check
`bench_perf.py` prints `detected quantization = ...`. For the W8A8 model it must be a quantized scheme (e.g.
compressed-tensors), for BF16 `None`. If W8A8 shows `None`, the checkpoint was loaded unquantized: stop and check the model id.

### 4.2 Performance: `python -m vllm_eval.bench.bench_perf --tag w8a8` (and `--tag bf16`)
- **Prefill:** `8x512` (one batch of 8 prompts, 4,096 tokens) and `1x2048`, `max_tokens=1`. Throughput = prompt tokens / wall
  time (includes the single sampled token). Prompts are random token ids; prefix caching is **off**, and
  `max_num_batched_tokens=8192` so the 8x512 batch is not chunked.
- **Decode:** batch 1, 8, 64, prompt length 128 (`--prompt-len`), exactly 128 output tokens (`ignore_eos`, checked).
  Each repeat runs `max_tokens=1` and `max_tokens=128`; **TPOT = (t128 - t1) / 127** removes the prefill from the latency.
  Also reported: end-to-end time and decode tokens/s.
- 3 warm-ups, 5 repeats (median/stdev saved). nvidia-smi clocks/power are logged before and after: **check them**, the L4
  (72 W) can throttle, which shows up as noisy repeats.

### 4.3 Perplexity: `python -m vllm_eval.bench.eval_ppl --tag w8a8`
WikiText-2 raw test, joined with `"\n\n"`, split into non-overlapping 2048-token windows (remainder dropped).
vLLM returns each prompt token's log-prob (`prompt_logprobs=0`), so the number comes from the **same W8A8 kernels** that are
benchmarked. `--max-windows 8` gives a quick partial run (flagged `partial_run` in the JSON). Memory-heavy: lower
`--batch` (default 2) if you see OOM.

### 4.4 Profiling: `python -m vllm_eval.profile.profile_breakdown --tag w8a8`
Uses `torch.profiler` (GPU activity only) with the engine core in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set in
`common.py`). Phases: `prefill_8x512`, `decode_b1`, `decode_b8`, `decode_b64`. Decode-only time =
profile(prefill + N decode steps) - profile(prefill only), on identical prompts.
Kernels are classified by name into `norm`, `quant`, `norm_quant_fused`, `gemm`, `attention`, `activation`, `rope`,
`memory`, `other`. The **fusable region** is a quant kernel executed right after a norm kernel on the same stream:
exactly the qkv_proj and gate_up_proj sites (o_proj/down_proj quant follow attention / SiLU, q_norm/k_norm are followed by RoPE).
Run it normally (CUDA graphs on, the real configuration) and in `--enforce-eager` (clearer attribution): `run_all.sh`
does both for W8A8 (`results/profile_w8a8.json`, `results/profile_w8a8_eager.json`).
Top kernels per phase: `results/profile_<tag>[_mode]_<phase>_kernels.csv`. **Open these**: kernel names vary with the vLLM
version and compile mode (e.g. Inductor-generated `triton_*` kernels); fix regexes in `vllm_eval/analysis.py` if something that
should be norm/quant lands in `other`. `--custom-ops "+rms_norm"` forces vLLM's CUDA RMSNorm kernel instead of the compiled one.

### 4.5 Amdahl: `python -m vllm_eval.reporting.amdahl --profile w8a8`
Two fractions per phase: `f_all` (all norm+quant) is the optimistic bound; `f_fusable` is what a fused norm+quant kernel
can actually replace. Both on GPU-kernel-time and wall-clock basis. Speedup `S = 1/((1-f) + f/s)` for `s` in {1.25, 1.5, 2, 3,
11/7, infinity}. `11/7 = 1.57` is a bandwidth-model bound for BF16 (unfused add+norm+quant = 11 B/element vs fused = 7 B/element).
A kernel speedup is never an end-to-end speedup: report them separately.

### 4.6 Integration: `python -m vllm_eval.integration.inspect_integration`
Greps the installed vLLM for the hooks (RMSNorm layer, `scaled_int8_quant`, W8A8 linear scheme, fusion pass, custom-op
registration, plugin entry points) and checks which ops exist in `torch.ops._C`. Read `results/integration_map.md` next to
`INTEGRATION_PROPOSAL.md` (options A layer-level plugin, B compile fusion pass, C kernel swap; recommended A then B).

### 4.7 Report: `python -m vllm_eval.reporting.make_report` -> `results/REPORT.md`

## 5. Output layout
```
vllm-evaluation/
├── run_all.sh
├── setup_env.sh
├── requirements.txt
├── INTEGRATION_PROPOSAL.md
├── vllm_eval/
│   ├── common.py, analysis.py
│   ├── bench/               env, throughput, latency, perplexity
│   ├── profile/             torch.profiler breakdown
│   ├── reporting/           Amdahl, report, self-test
│   └── integration/         vLLM hook map
├── logs/                    per-step logs (gitignored)
├── .stamps/                 resume markers (gitignored)
└── results/                 gitignored measurements
    ├── env.json, bench_*.json, ppl_*.json
    ├── profile_*.json, profile_*_kernels.csv
    ├── amdahl.json, amdahl.md, integration_map.md, REPORT.md
    └── traces/              deleted unless --keep-trace
```

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| "GPU is not an NVIDIA L4" | run on the L4, or `--allow-non-l4` for a different GPU |
| CUDA OOM at load / during ppl | lower `--gpu-mem` (0.80), `--max-model-len`, or ppl `--batch 1` |
| W8A8 reports quantization `None` | wrong model id or loader; check `--model` and the vLLM log |
| Unknown argument errors from `LLM(...)` | vLLM API drift: adjust `make_llm` in `vllm_eval/common.py` (one place) |
| Profile shows no kernels / empty tables | engine ran in another process; make sure `VLLM_ENABLE_V1_MULTIPROCESSING=0` was not overridden |
| Large `other` share or norm/quant = 0 | open the kernels CSV, adjust regexes in `vllm_eval/analysis.py`, re-run `python -m vllm_eval.profile.profile_breakdown` |
| Noisy timings | check nvidia-smi clocks/power in the JSON, rerun with more `--repeats`, close other GPU jobs |
| Trace too large / slow | lower `--decode-steps` |

## 7. Caveats
- Single L4, one model, offline (non-concurrent) runs: no claims about concurrent serving.
- TPOT is derived by subtraction; profiler decode breakdown uses the same trick, so tiny negative noise is possible at small batch.
- Prefill throughput includes one decode/sampling step and scheduler overhead.
- Perplexity depends on the windowing protocol; compare only numbers produced by this script.
- The offline self-test checks the analysis logic only. The GPU scripts need a first run on the L4 (use `QUICK=1`); vLLM
  API names can differ between versions.
