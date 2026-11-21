# HPML INT8 Project

Why is fused RMSNorm + INT8 quantization slow?

This repo holds the workstreams for that question. Each workstream lives in its own directory. Source, job scripts, and generated results stay in separate folders.

## Layout

```
.
├── README.md
├── .gitignore
└── ws2/                          Workstream 2: Nsight profiling toolkit
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

## Workstream 2

Compare the unfused path (V0: `fused_add_rms_norm` then `scaled_int8_quant`) with the fused path (V1: `rms_norm_dynamic_per_token_quant`) using Nsight Systems and Nsight Compute.

Setup, the sweep, and how to read the tables are in [ws2/README.md](ws2/README.md). From a GPU node:

```bash
cd ws2
./run_all.sh
```
