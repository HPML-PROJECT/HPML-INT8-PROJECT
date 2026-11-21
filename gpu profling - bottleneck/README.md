# WS2: Why is fused RMSNorm + INT8 quantization slow? (profiling toolkit)

Workstream 2 of the HPML project *"Why Is Fused RMSNorm + INT8 Quantization Slow?"*.
**Objective:** understand why the fused and unfused implementations perform differently, using Nsight Compute and Nsight Systems.

**Deliverables produced by this toolkit**
1. Nsight reports labeled by variant, GPU, shape and cache mode
2. Memory and execution-resource comparison tables
3. An initial bottleneck analysis with supporting measurements

---

## Contents

1. [What is being profiled](#1-what-is-being-profiled)
2. [Folder contents](#2-folder-contents)
3. [Prerequisites](#3-prerequisites)
4. [One-time setup](#4-one-time-setup)
5. [Quick start](#5-quick-start)
6. [Step-by-step guide](#6-step-by-step-guide)
7. [Output layout and naming](#7-output-layout-and-naming)
8. [Configuration reference](#8-configuration-reference)
9. [Cold vs. warm cache](#9-cold-vs-warm-cache)
10. [Reading the tables](#10-reading-the-tables)
11. [Turning measurements into the bottleneck analysis](#11-turning-measurements-into-the-bottleneck-analysis)
12. [Running on several GPUs](#12-running-on-several-gpus)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations and caveats](#14-limitations-and-caveats)
15. [Final checklist](#15-final-checklist)

---

## 1. What is being profiled

Both variants perform the same work on a `(T tokens x H hidden)` BF16 activation: add the residual, apply RMSNorm with the weight, quantize to INT8 with one dynamic scale per token, and write the updated residual, the INT8 output and the scales.

| Variant | Calls (vLLM `_custom_ops`) | Kernels per call |
|---|---|---|
| **V0, unfused** | `fused_add_rms_norm(x, residual, w, eps)` then `scaled_int8_quant(x)` | 2 |
| **V1, vLLM fused** | `rms_norm_dynamic_per_token_quant(x, w, eps, torch.int8, residual=residual)` | 1 |

Shapes: hidden size **2048** (Qwen3-1.7B) and optionally **4096** (8B-class models). Token counts: **8, 1024, 4096** (small, medium, large). The `--no-residual` flag switches both variants to plain RMSNorm + quant.

Known issue being investigated (vLLM #60499): V1 beats V0 at decode sizes but is up to about 2x slower at prefill sizes on an RTX 4090. This toolkit measures why, on your GPUs.

## 2. Folder contents

| File | Purpose |
|---|---|
| `run_all.sh` | **Driver.** Runs every step in order, logs, resumable. |
| `slurm/job.sbatch` | Slurm wrapper that calls `run_all.sh` (edit partition/GPU lines). |
| `scripts/check_ncu_counters.sh` | Step 0: checks that Nsight Compute can read GPU counters. |
| `harness/profile_target.py` | Harness: V0/V1 calls with modes `profile`, `verify`, `bench`. |
| `scripts/run_nsys.sh` | Nsight Systems timelines and kernel summaries. |
| `scripts/run_ncu_sweep.sh` | Nsight Compute sweep over variants x tokens x cache modes. |
| `analysis/parse_ncu.py` | Converts `.ncu-rep` files into the memory and execution tables. |
| `analysis/analyze_bottlenecks.py` | Generates the first-pass bottleneck analysis from the tables. |

## 3. Prerequisites

- A **GPU compute node** (not a login node). Use the same GPU type as WS1 for each run.
- **CUDA toolkit with Nsight Compute (`ncu`) and Nsight Systems (`nsys`)** on `PATH`. Check with `ncu --version` and `nsys --version`.
- **Python environment with vLLM** (the version pinned by the team) and its matching PyTorch. The harness imports `vllm._custom_ops`, so vLLM's compiled `_C` extension must be installed (a normal pip install of vLLM provides it).
- **Permission to read GPU performance counters.** Shared clusters often block this. Step 0 checks it.
- `nvidia-smi` and `bash`. Python 3.9+.

## 4. One-time setup

The exact module names and Slurm options depend on the cluster, so the lines below use placeholders.

```bash
# 1. Get a GPU node (interactive). Adjust partition/account/GPU type for your cluster.
srun --gres=gpu:1 --time=03:00:00 --pty bash

# 2. Load CUDA tools and activate the vLLM environment
module load cuda                      # whatever provides ncu / nsys
source ~/venvs/vllm/bin/activate      # your pinned vLLM env

# 3. Go to the toolkit folder and make scripts executable
cd "gpu profling - bottleneck"
chmod +x run_all.sh scripts/*.sh slurm/job.sbatch harness/*.py analysis/*.py

# 4. Confirm the environment
nvidia-smi
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.get_device_name())"
python -c "from vllm import _custom_ops as o; print([n for n in dir(o) if 'rms' in n or 'int8' in n])"
```

The last command should list `fused_add_rms_norm`, `rms_norm`, `rms_norm_dynamic_per_token_quant` and `scaled_int8_quant`. If any are missing, see [Troubleshooting](#13-troubleshooting).

## 5. Quick start

```bash
# Full plan on the current GPU (hidden 2048 and 4096, tokens 8/1024/4096, cold + warm)
./run_all.sh

# or as a batch job
sbatch slurm/job.sbatch
```

When it finishes, open `tables/memory_table.md`, `tables/execution_table.md` and `tables/bottleneck_analysis.md`. Repeat on each GPU type ([section 12](#12-running-on-several-gpus)).

Useful variations:

```bash
HIDDENS="2048" ./run_all.sh                      # only Qwen3-1.7B shapes
TOKENS="8 256 1024 4096" ./run_all.sh            # different token counts
ONLY="verify bench" ./run_all.sh                 # run only selected steps
ONLY="ncu parse analyze" FORCE=1 ./run_all.sh    # redo selected steps
```

## 6. Step-by-step guide

`run_all.sh` performs steps 0 to 6. You can also run each step by hand. Every command below is run from the toolkit folder on a GPU node.

### Step 0: Check that Nsight Compute can read counters

```bash
./scripts/check_ncu_counters.sh
```
Prints the GPU, tool versions, and runs a tiny profile.
- **Success:** `RESULT: OK - Nsight Compute can read counters on this GPU.`
- **Failure:** `RESULT: FAIL (ERR_NVGPUCTRPERM)`. The driver blocks counters for normal users. Either ask the cluster admins to enable profiling for your account, or use the fallback: a **Google Cloud L4 VM** (Ada, sm_89, the same architecture as the L40S) and run the sweep with sudo, e.g. `sudo -E env "PATH=$PATH" ./scripts/run_ncu_sweep.sh`. Note in your report that the L4 has a different L2 size and bandwidth than the L40S.

Do not continue to step 4 until this passes. Steps 1 to 3 do not need counters.

### Step 1: Verify that V0 and V1 do identical work

```bash
python harness/profile_target.py --mode verify --tokens 4096 --hidden 2048
```
Compares both variants against an FP32 reference and against each other. Look at:
- `scale rel.err(max)` and `dequant ... err`: small for both variants.
- `residual max|diff|`: near zero (the updated residual must match).
- `v0 vs v1 INT8: mismatch rate`: small. The vLLM issue saw 0.2 to 1.1% of values differ by 2 steps or fewer, caused by rounding differences.

If the variants differ more than that, fix the comparison before profiling, because a profile of two different workloads is not a fair comparison. Run it for each token count and hidden size (`run_all.sh` does this).

### Step 2: Direction sanity check (optional but recommended)

```bash
python harness/profile_target.py --mode bench --tokens 4096 --hidden 2048
```
Quick CUDA-event timing. It confirms the direction of the effect (V1 slower than V0 at large T) before the long profiling runs. Event timing includes launch overhead, so **do not quote these numbers**; the authoritative times come from Nsight Compute.

### Step 3: Nsight Systems timelines

```bash
./scripts/run_nsys.sh 2048
```
Writes `reports/nsys/<variant>_<gpu>_H<hidden>_T<tokens>.nsys-rep` and a `.kern_sum.csv` kernel summary. Open the `.nsys-rep` in the Nsight Systems GUI. Check: V0 shows 2 kernels per call and V1 shows 1, plus the launch gaps and kernel durations.

### Step 4: Nsight Compute sweep

```bash
./scripts/run_ncu_sweep.sh 2048        # then again with 4096
```
For each variant x token count x cache mode it runs `ncu --set full` plus extra counters (DRAM and L2 bytes, L2 hit rate, load sectors and requests, registers, occupancy, stall reasons). Only kernels inside the NVTX range `target` are profiled, so the input-restore copies are excluded. Output: `reports/ncu/<variant>_<gpu>_H<hidden>_T<tokens>_<cold|warm>.ncu-rep`.

Open any report in the Nsight Compute GUI for the source and memory charts. The sweep can take a while because `--set full` replays each kernel many times. To shorten it, reduce `TOKENS`, `CACHES` or `VARIANTS` ([section 8](#8-configuration-reference)).

### Step 5: Build the comparison tables

```bash
python analysis/parse_ncu.py reports/ncu
```
Writes `tables/memory_table.{csv,md}` and `tables/execution_table.{csv,md}`. Each report is grouped by kernel (median over the 3 profiled calls). V0 gets a `TOTAL` row that sums its two kernels so it compares directly with V1. Column meanings are in [section 10](#10-reading-the-tables).

### Step 6: First-pass bottleneck analysis

```bash
python analysis/analyze_bottlenecks.py
```
Writes `tables/bottleneck_analysis.md`: for each GPU, hidden size, cache mode and token count it lists the time ratio (V1/V0), DRAM bytes ratio, V1 DRAM throughput, occupancy, top stall, load-width evidence and instruction ratio, plus a warm/cold table for the cache-residency hypothesis. The flags are heuristics; confirm them against the raw tables ([section 11](#11-turning-measurements-into-the-bottleneck-analysis)).

## 7. Output layout and naming

```
gpu profling - bottleneck/
├── run_all.sh                 driver
├── slurm/job.sbatch
├── scripts/                   ncu/nsys sweeps and the counter check
├── harness/profile_target.py  V0/V1 calls
├── analysis/                  parse_ncu.py, analyze_bottlenecks.py
├── logs/                      one log per step per GPU (gitignored)
├── .stamps/                   resume markers for run_all.sh (gitignored)
├── reports/
│   ├── ncu/<variant>_<gpu>_H<hidden>_T<tokens>_<cold|warm>.ncu-rep
│   └── nsys/<variant>_<gpu>_H<hidden>_T<tokens>.nsys-rep  (+ .kern_sum.csv)
└── tables/
    ├── memory_table.csv / .md
    ├── execution_table.csv / .md
    └── bottleneck_analysis.md
```
Example name: `v1_NVIDIA_L40S_H2048_T4096_cold.ncu-rep` = variant V1, NVIDIA L40S, hidden 2048, 4096 tokens, cache flushed.

## 8. Configuration reference

Environment variables (set on the command line before the script):

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `HIDDENS` | `2048 4096` | `run_all.sh` | Hidden sizes to run |
| `TOKENS` | `8 1024 4096` | all sweeps | Token counts |
| `VARIANTS` | `v0 v1` | sweeps | Variants to profile |
| `CACHES` | `cold warm` | `run_ncu_sweep.sh` | Cache modes |
| `ONLY` | all steps | `run_all.sh` | Steps to run: `counters verify bench nsys ncu parse analyze` |
| `FORCE` | `0` | `run_all.sh` | `1` = redo steps already done on this GPU |
| `PY` | `python` | sweeps | Python executable |
| `OUT` | `reports/ncu` or `reports/nsys` | sweeps | Output folder |
| `EXTRA` | empty | `run_ncu_sweep.sh` | Extra harness flags, e.g. `EXTRA="--no-residual"` |

Harness flags (`profile_target.py`): `--variant {v0,v1}`, `--tokens`, `--hidden`, `--dtype {bf16,fp16}`, `--no-residual`, `--mode {profile,verify,bench}`, `--iters` (profiled calls, default 3), `--warmup` (default 5), `--seed`.

## 9. Cold vs. warm cache

The cache mode changes what the profile means, so every report records it in its filename.

| Mode | ncu flag | Meaning |
|---|---|---|
| `cold` | `--cache-control all` (ncu default) | Caches are flushed before each replay pass, so data comes from DRAM. |
| `warm` | `--cache-control none` | No flushing, so data left in L2 by earlier passes may be reused. |

"Warm" only means "not flushed". **Check the measured L2 hit rate** instead of assuming the data was resident. This matters for hypothesis H2: the L40S has a large L2 (96 MB) that can hold a full intermediate tensor, while the A6000 has 6 MB.

## 10. Reading the tables

**`memory_table`**

| Column | Meaning |
|---|---|
| `variant`, `gpu`, `H`, `T`, `cache` | Configuration |
| `kernel` | Kernel name; `TOTAL` = sum over the variant's kernels |
| `time_us` | GPU kernel duration (Nsight Compute, base clocks) |
| `dram_MB`, `dram_rd_MB`, `dram_wr_MB` | DRAM traffic |
| `L2_MB`, `L2_hit_%` | L2 traffic and sector hit rate |
| `DRAM_%peak` | DRAM throughput as a percent of peak |
| `dram_GBps` | DRAM bytes / time |
| `min_bytes_MB` | Minimum bytes the whole operation must move (input, residual in/out, weight, INT8 output, scales) |
| `dram/min` | (TOTAL rows) DRAM bytes divided by the minimum; above 1 means extra traffic |

**`execution_table`**

| Column | Meaning |
|---|---|
| `regs/thr`, `block`, `grid` | Registers per thread, threads per block, blocks |
| `occ_ach_%`, `occ_theo_%` | Achieved and theoretical occupancy |
| `inst`, `gld_inst` | Executed instructions, global-load instructions |
| `ld_sectors/req` | Global-load sectors per request (about 4 = well coalesced for 32-bit loads) |
| `approx_B/ld_inst` | Approximate bytes moved per global-load instruction (higher = wider loads) |
| `top_stalls` | Top warp-stall reasons (warps stalled per issue-active cycle) |

## 11. Turning measurements into the bottleneck analysis

Start from `tables/bottleneck_analysis.md`, then write about one page covering:

1. **Where V1 stops winning.** Compare the time ratio at T = 8, 1024 and 4096. Is the crossover point the same on each GPU?
2. **H1: how the fused kernel moves data.** Look at `approx_B/ld_inst`, `ld_sectors/req`, the `inst` ratio, `DRAM_%peak`, occupancy and `top_stalls`. Low bandwidth alone does not prove scalar loads (the proposal cites about 40% DRAM throughput on sm_86), so look for agreement between load width, stalls and instruction counts.
3. **H2: cache residency.** Compare warm and cold results and the L2 hit rate. Does the GPU with the larger L2 behave differently?
4. **Verdict per hypothesis:** supported, contradicted or inconclusive, with the numbers that justify it.
5. **Hand-off:** list candidate causes for the ablation and Triton work (vectorized vs. scalar loads, one vs. two passes per row, register use).

## 12. Running on several GPUs

Run the same toolkit once on each GPU type (A6000, L40S, and H100 if available). File names include the GPU, so results never overwrite each other, and `.stamps/` is tracked per GPU.

```bash
# on an A6000 node
./run_all.sh
# on an L40S node (same folder)
./run_all.sh
# afterwards, on any machine that has all reports/ncu files (needs only ncu for parsing)
python analysis/parse_ncu.py reports/ncu && python analysis/analyze_bottlenecks.py
```
Copy `reports/` between machines if you ran on different nodes with separate folders.

## 13. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Could not import vllm._custom_ops` | Wrong environment. Activate the pinned vLLM env; check `python -c "import vllm"`. |
| `AttributeError` for `fused_add_rms_norm`, `scaled_int8_quant` or `rms_norm_dynamic_per_token_quant` | Op names or signatures differ in your vLLM version. Run the `dir(o)` command in [section 4](#4-one-time-setup) and edit `call_v0`/`call_v1` in `harness/profile_target.py`. |
| `ERR_NVGPUCTRPERM` | Counters blocked. See step 0 (admin request or L4 fallback). |
| ncu profiles nothing / empty reports | The NVTX filter did not match. In `scripts/run_ncu_sweep.sh` replace `--nvtx --nvtx-include "target/"` with `-k regex:"rms_norm\|int8_quant"`. |
| ncu error: unknown metric | A metric name is not supported on your GPU/ncu version. Remove it from the `METRICS` list in `scripts/run_ncu_sweep.sh`; the parser leaves missing columns empty. |
| Sweep is very slow | Reduce `TOKENS`, `CACHES` or `HIDDENS`, or switch `--set full` to `--set detailed`. |
| `parse_ncu.py`: `skip (name)` | A report does not follow the naming pattern. Use the sweep script's names. |
| Tables have empty or `nan` columns | The metric was not collected. Check the raw report in the GUI and the metric list. |
| `analyze_bottlenecks.py`: missing tables | Run `parse_ncu.py` first. |
| Step skipped unexpectedly | A stamp exists for this GPU. Use `FORCE=1`, or delete `.stamps/`. |

## 14. Limitations and caveats

- `ncu` locks clocks to base (`--clock-control base`), so absolute times are higher than real runs. Compare variants only within one GPU and one setup. Use `--clock-control none` if you prefer boost-clock numbers, and state it.
- Nsight Compute replays kernels and restores memory between passes, so the in-place ops are safe, but timing under replay is not the same as back-to-back CUDA-graph timing. Use this toolkit for **diagnosis**; use the WS1 benchmark for headline timings.
- Kernel names vary across vLLM versions; tables use the names `ncu` reports.
- The analysis flags use fixed thresholds (listed at the bottom of `bottleneck_analysis.md`). They suggest where to look and do not replace reading the numbers.
- Results cover one operation, one serving system and the GPUs you can access.

## 15. Final checklist

- [ ] `check_ncu_counters.sh` reports OK (or the L4 fallback is documented)
- [ ] `verify` logs show V0 and V1 do the same work
- [ ] Nsight reports exist for every variant x token count x cache mode x hidden size, named with variant, GPU, shape and cache mode
- [ ] `tables/memory_table.md` and `tables/execution_table.md` generated for each GPU
- [ ] `tables/bottleneck_analysis.md` reviewed; hypothesis verdicts written with measurements
- [ ] Cache mode and clock-control setting stated next to every reported number
