# WS2: Why is fused RMSNorm + INT8 quantization slow? (profiling toolkit)

Workstream 2 of the HPML project *"Why Is Fused RMSNorm + INT8 Quantization Slow?"*.
**Objective:** understand why the fused and unfused implementations perform differently, using Nsight Compute and Nsight Systems.

**Deliverables produced by this toolkit**
1. Nsight reports labeled by variant, GPU, shape and cache mode
2. Memory and execution-resource comparison tables
3. An initial bottleneck analysis with supporting measurements

---

## 0. Alignment with Part 1 (WS1 baseline) — read first

This toolkit is set up to measure **exactly what Part 1 benchmarked** (`rmsnorm_int8_bench`, results
`results_l4_20261121_213823`), so its numbers explain Part 1's results instead of a different workload.

| Setting | Part 1 | This toolkit |
|---|---|---|
| Inputs | CPU generator, per-shape seed, seed 1234; x ~ N(0,1), w = 1 + 0.1·N(0,1), residual ~ N(0,1) | **Identical** (same code); `verify` checks bit-identity against Part 1's script if it is in `$PART1_DIR` |
| dtype / eps / residual | BF16 (main), eps 1e-6, residual add on | Same defaults |
| Kernels | `torch.ops._C` fused_add_rms_norm + dynamic_scaled_int8_quant (V0); rms_norm_dynamic_per_token_quant (V1) | Same kernels via `vllm._custom_ops` |
| Reference / tolerances | FP32 with vLLM's rounding points; ≤3 INT8 steps on ≤15%, scales 1.6%, dequant ≤2.5 steps | Same; `verify` **fails the run** if they are not met |
| GPU / clocks | NVIDIA L4, SM clock locked at 2040 MHz | `run_l4_vm.sh` locks 2040 MHz and runs ncu with `--clock-control none` |
| Token counts | 1–16384; V1 slower at 128–4096 (warm), faster at ≥ 8192 | Default `8 256 1024 4096 16384`: both sides of the switch |
| Cold cache | 8×L2 buffer written before each call | ncu `--cache-control all` (flush before each replay) — same intent |
| Warm cache | back-to-back calls in a CUDA graph | ncu `--cache-control none`; **not identical**: ncu replays each kernel separately, so V0's quant kernel may lose L2 reuse. Check its L2 hit rate and `tables/part1_vs_ncu.md` |

`analysis/compare_part1.py` writes `tables/part1_vs_ncu.md`: Part 1's V1/V0 ratio next to Nsight Compute's for every
profiled shape, and flags shapes where they point different ways. Report those honestly.

**On the L4 VM (same VM and environment as Part 1):**
```bash
cd ~/ws2
QUICK=1 bash run_l4_vm.sh     # ~10 min setup test (H=2048, T=256 and 16384)
bash run_l4_vm.sh             # full plan (~1 h); prints the results bundle path at the end
```

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

Shapes: hidden size **2048** (Qwen3-1.7B) and **4096** (8B-class models). Token counts: **8, 256, 1024, 4096, 16384** (decode, Part 1's slow zone, and the large size where V1 wins). The `--no-residual` flag switches both variants to plain RMSNorm + quant.

Known issue being investigated (vLLM #60499): V1 beats V0 at decode sizes but is up to about 2x slower at prefill sizes on an RTX 4090. This toolkit measures why on our GPU: an **NVIDIA L4 on Google Cloud** (Ada, sm_89, the same architecture as the RTX 4090; 24 GB, ~300 GB/s, 48 MB L2, 58 SMs, 72 W).

## 2. Folder contents

| File | Purpose |
|---|---|
| `run_all.sh` | **Driver.** Runs every step in order, logs, resumable. |
| `slurm/job.sbatch` | Optional Slurm wrapper for running on a cluster later (not used on the L4 VM). |
| `scripts/check_ncu_counters.sh` | Step 0: checks that Nsight Compute can read GPU counters. |
| `harness/profile_target.py` | Harness: V0/V1 calls with modes `profile`, `verify`, `bench`. |
| `scripts/run_nsys.sh` | Nsight Systems timelines and kernel summaries. |
| `scripts/run_ncu_sweep.sh` | Nsight Compute sweep over variants x tokens x cache modes. |
| `analysis/parse_ncu.py` | Converts `.ncu-rep` files into the memory and execution tables. |
| `analysis/analyze_bottlenecks.py` | Generates the first-pass bottleneck analysis from the tables. |

## 3. Prerequisites

- The team's **Google Cloud L4 VM**, the same GPU as WS1.
- **CUDA toolkit with Nsight Compute (`ncu`) and Nsight Systems (`nsys`)** on `PATH`. Check with `ncu --version` and `nsys --version`.
- **Python environment with vLLM** (the version pinned by the team) and its matching PyTorch. The harness imports `vllm._custom_ops`, so vLLM's compiled `_C` extension must be installed (a normal pip install of vLLM provides it).
- **Permission to read GPU performance counters.** The NVIDIA driver blocks them for non-root users by default, so on the VM you may need `sudo` (step 0 checks this).
- `nvidia-smi` and `bash`. Python 3.9+.

## 4. One-time setup

### 4a. Create the VM (once per team)

- **Machine type:** a `g2-standard-*` instance. G2 is the GCP machine family that comes with an NVIDIA L4. `g2-standard-8` is enough for this toolkit.
- **Boot image:** a GCP Deep Learning VM image with CUDA 12.x, so the NVIDIA driver and CUDA toolkit are preinstalled. With a plain Ubuntu image, install the driver yourself.
- **Disk:** at least 100 GB. vLLM, PyTorch and the CUDA toolkit are large.
- **Cost:** stop the VM when you're not using it. A stopped VM costs only its disk.

### 4b. Prepare the environment on the VM

# 3. Go to the toolkit folder and make scripts executable
cd "gpu profling - bottleneck"
chmod +x run_all.sh scripts/*.sh slurm/job.sbatch harness/*.py analysis/*.py

```bash
# 1. Check the GPU and driver (expect "NVIDIA L4", compute capability 8.9)
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

# 2. Make ncu and nsys available. pip-installed vLLM/PyTorch do NOT include them.
export PATH=/usr/local/cuda/bin:$PATH        # add to ~/.bashrc
ncu --version && nsys --version
# If either is missing (pick the CUDA 12.x version that matches your driver):
#   sudo apt-get install -y cuda-nsight-compute-12-4 nsight-systems
# ncu must be 2022.3 or newer to support sm_89.

# 3. Create the pinned vLLM environment (use the team's pinned version)
python3 -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
pip install "vllm==<pinned-version>"

# 4. Get the code with git, not by copying files from a Windows laptop
#    (Windows copies can have CRLF line endings, which break the bash scripts)
git clone https://github.com/HPML-PROJECT/HPML-INT8-PROJECT.git
cd "HPML-INT8-PROJECT/gpu profling - bottleneck"
chmod +x run_all.sh scripts/*.sh harness/*.py analysis/*.py

# 5. Confirm the environment
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.get_device_name())"
python -c "from vllm import _custom_ops as o; print([n for n in dir(o) if 'rms' in n or 'int8' in n])"
```

The last command should list `fused_add_rms_norm`, `rms_norm`, `rms_norm_dynamic_per_token_quant` and `scaled_int8_quant`. If any are missing, see [Troubleshooting](#13-troubleshooting).

### 4c. Allow GPU performance counters (recommended)

By default the driver blocks performance counters for non-root users, so Nsight Compute fails with `ERR_NVGPUCTRPERM`. Running everything with `sudo` works, but then `reports/`, `tables/`, `logs/` and `.stamps/` are owned by root, and later runs without sudo can't overwrite them. Since we own the VM, enable counters for all users once instead:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
# after the reboot, confirm:
./scripts/check_ncu_counters.sh
```

If you already ran with sudo, take back ownership of the output files with `sudo chown -R $USER: reports tables logs .stamps`.

## 5. Quick start

```bash
# Full plan on the L4 (hidden 2048 and 4096, tokens 8/1024/4096, cold + warm)
./run_all.sh

# if step 0 reports ERR_NVGPUCTRPERM, run as root while keeping your PATH and venv
sudo -E env "PATH=$PATH" ./run_all.sh
```

The sweep takes a while, so run it inside `tmux` or `screen` so it survives an SSH disconnect. When it finishes, open `tables/memory_table.md`, `tables/execution_table.md` and `tables/bottleneck_analysis.md`.

Useful variations:

```bash
HIDDENS="2048" ./run_all.sh                      # only Qwen3-1.7B shapes
TOKENS="8 256 1024 4096" ./run_all.sh            # different token counts
ONLY="verify bench" ./run_all.sh                 # run only selected steps
ONLY="ncu parse analyze" FORCE=1 ./run_all.sh    # redo selected steps
```

## 6. Step-by-step guide

`run_all.sh` performs steps 0 to 6. You can also run each step by hand. Every command below is run from the toolkit folder on the L4 VM.

### Step 0: Check that Nsight Compute can read counters

```bash
./scripts/check_ncu_counters.sh
```
Prints the GPU, tool versions, and runs a tiny profile.
- **Success:** `RESULT: OK - Nsight Compute can read counters on this GPU.`
- **Failure:** `RESULT: FAIL (ERR_NVGPUCTRPERM)`. The driver blocks counters for non-root users. Since we own the VM, either:
  - run with sudo: `sudo -E env "PATH=$PATH" ./run_all.sh` (or the same prefix on any single script), or
  - allow counters for all users permanently: `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf`, then `sudo update-initramfs -u` and reboot the VM.

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
Example name: `v1_NVIDIA_L4_H2048_T4096_cold.ncu-rep` = variant V1, NVIDIA L4, hidden 2048, 4096 tokens, cache flushed.

## 8. Configuration reference

Environment variables (set on the command line before the script):

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `HIDDENS` | `2048 4096` | `run_all.sh` | Hidden sizes to run |
| `TOKENS` | `8 256 1024 4096 16384` | all sweeps | Token counts |
| `VARIANTS` | `v0 v1` | sweeps | Variants to profile |
| `CACHES` | `cold warm` | `run_ncu_sweep.sh` | Cache modes |
| `ONLY` | all steps | `run_all.sh` | Steps to run: `counters verify bench nsys ncu parse analyze` |
| `FORCE` | `0` | `run_all.sh` | `1` = redo steps already done on this GPU |
| `PY` | `python` | sweeps | Python executable |
| `OUT` | `reports/ncu` or `reports/nsys` | sweeps | Output folder |
| `EXTRA` | empty | `run_ncu_sweep.sh` | Extra harness flags, e.g. `EXTRA="--no-residual"` |
| `CLOCK_CONTROL` | `base` (`none` in `run_l4_vm.sh`) | `run_ncu_sweep.sh` | ncu clock control |
| `PART1_DIR` | `~/rmsnorm_int8_bench` | harness, `run_l4_vm.sh` | Part 1 folder (input check, venv, results CSV) |

Harness flags (`profile_target.py`): `--variant {v0,v1}`, `--tokens`, `--hidden`, `--dtype {bf16,fp16}`, `--no-residual`, `--mode {profile,verify,bench}`, `--iters` (profiled calls, default 3), `--warmup` (default 5), `--seed` (default 1234, as in Part 1).

## 9. Cold vs. warm cache

The cache mode changes what the profile means, so every report records it in its filename.

| Mode | ncu flag | Meaning |
|---|---|---|
| `cold` | `--cache-control all` (ncu default) | Caches are flushed before each replay pass, so data comes from DRAM. |
| `warm` | `--cache-control none` | No flushing, so data left in L2 by earlier passes may be reused. |

"Warm" only means "not flushed". **Check the measured L2 hit rate** instead of assuming the data was resident. This matters for hypothesis H2: the L4's 48 MB L2 can hold the full ~16 MB intermediate tensor (H = 2048, T = 4096), so warm-cache V0 may benefit from L2 reuse that a GPU with a small L2 (e.g. the A6000, 6 MB) would not get.

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

1. **Where V1 stops winning.** Compare the time ratio at T = 8, 1024 and 4096, and compare the crossover with the RTX 4090 numbers in vLLM #60499. Compare ratios, not absolute µs: the L4 has about 3× less DRAM bandwidth.
2. **H1: how the fused kernel moves data.** Look at `approx_B/ld_inst`, `ld_sectors/req`, the `inst` ratio, `DRAM_%peak`, occupancy and `top_stalls`. Low bandwidth alone does not prove scalar loads (the proposal cites about 40% DRAM throughput on sm_86), so look for agreement between load width, stalls and instruction counts.
3. **H2: cache residency.** Compare warm and cold results and the L2 hit rate. Does the V0 intermediate stay in the L4's L2 when warm?
4. **Verdict per hypothesis:** supported, contradicted or inconclusive, with the numbers that justify it.
5. **Hand-off:** list candidate causes for the ablation and Triton work (vectorized vs. scalar loads, one vs. two passes per row, register use).

## 12. Running on several GPUs

All current results come from the L4. If another GPU type becomes available later, run the same toolkit there. File names include the GPU, so results never overwrite each other, and `.stamps/` is tracked per GPU. On a Slurm cluster, use `sbatch slurm/job.sbatch` after editing its partition/GPU lines.

```bash
# copy reports to your laptop (the .ncu-rep files open in the Nsight Compute GUI)
gcloud compute scp --recurse "<vm-name>:~/HPML-INT8-PROJECT/gpu\ profling\ -\ bottleneck/reports" "./gpu profling - bottleneck/" --zone <zone>

# after collecting reports/ncu from every GPU into one folder (parsing needs only ncu)
python analysis/parse_ncu.py reports/ncu && python analysis/analyze_bottlenecks.py
```

## 13. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Could not import vllm._custom_ops` | Wrong environment. Activate the pinned vLLM env; check `python -c "import vllm"`. |
| `AttributeError` for `fused_add_rms_norm`, `scaled_int8_quant` or `rms_norm_dynamic_per_token_quant` | Op names or signatures differ in your vLLM version. Run the `dir(o)` command in [section 4](#4-one-time-setup) and edit `call_v0`/`call_v1` in `harness/profile_target.py`. |
| `ERR_NVGPUCTRPERM` | Counters blocked for non-root users. See step 0 (run with sudo, or set `NVreg_RestrictProfilingToAdminUsers=0`). |
| ncu profiles nothing / empty reports | The NVTX filter did not match. In `scripts/run_ncu_sweep.sh` replace `--nvtx --nvtx-include "target/"` with `-k regex:"rms_norm\|int8_quant"`. |
| ncu error: unknown metric | A metric name is not supported on your GPU/ncu version. Remove it from the `METRICS` list in `scripts/run_ncu_sweep.sh`; the parser leaves missing columns empty. |
| `ncu` or `nsys`: command not found (often under sudo) | Add `/usr/local/cuda/bin` to `PATH`, and keep it under sudo with `sudo -E env "PATH=$PATH" ...`. |
| Sweep is very slow | Reduce `TOKENS`, `CACHES` or `HIDDENS`, or switch `--set full` to `--set detailed`. |
| `parse_ncu.py`: `skip (name)` | A report does not follow the naming pattern. Use the sweep script's names. |
| Tables have empty or `nan` columns | The metric was not collected. Check the raw report in the GUI and the metric list. |
| `analyze_bottlenecks.py`: missing tables | Run `parse_ncu.py` first. |
| `/usr/bin/env: 'bash\r'` or `$'\r': command not found` | The scripts have Windows (CRLF) line endings. Clone with git on the VM, or run `sed -i 's/\r$//' run_all.sh scripts/*.sh`. |
| `Permission denied` writing `reports/`, `tables/` or `.stamps/` | An earlier run used sudo. Run `sudo chown -R $USER: reports tables logs .stamps`. |
| Step skipped unexpectedly | A stamp exists for this GPU. Use `FORCE=1`, or delete `.stamps/`. |

## 14. Limitations and caveats

- `ncu` locks clocks to base (`--clock-control base`), so absolute times are higher than real runs. Compare variants only within one GPU and one setup. Use `--clock-control none` if you prefer boost-clock numbers, and state it.
- Nsight Compute replays kernels and restores memory between passes, so the in-place ops are safe, but timing under replay is not the same as back-to-back CUDA-graph timing. Use this toolkit for **diagnosis**; use the WS1 benchmark for headline timings.
- Kernel names vary across vLLM versions; tables use the names `ncu` reports.
- The analysis flags use fixed thresholds (listed at the bottom of `bottleneck_analysis.md`). They suggest where to look and do not replace reading the numbers.
- Results cover one operation, one serving system and one GPU (the L4). The L4 is limited to 72 W; `ncu --clock-control base` keeps its clocks fixed during profiling, but WS1-style timings can drift as it heats up.

## 15. Final checklist

- [ ] `check_ncu_counters.sh` reports OK on the L4 (state whether sudo was needed)
- [ ] `verify` logs show V0 and V1 do the same work
- [ ] Nsight reports exist for every variant x token count x cache mode x hidden size, named with variant, GPU, shape and cache mode
- [ ] `tables/memory_table.md` and `tables/execution_table.md` generated
- [ ] `tables/bottleneck_analysis.md` reviewed; hypothesis verdicts written with measurements
- [ ] Cache mode and clock-control setting stated next to every reported number
