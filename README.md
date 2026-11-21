# HPML INT8 Project

Why is fused RMSNorm + INT8 quantization slow?

Each workstream lives in its own directory. Source stays separate from generated results.

## Layout

```
.
├── README.md
├── .gitignore
├── gpu profling - bottleneck/   Nsight profiling of the fused and unfused kernels
└── vllm-evaluation/             end-to-end vLLM benchmark and integration notes
    ├── README.md
    ├── INTEGRATION_PROPOSAL.md
    ├── run_all.sh
    ├── setup_env.sh
    ├── requirements.txt
    ├── vllm_eval/               bench, profile, reporting, integration
    ├── results/                 generated measurements (gitignored)
    └── logs/                    per-step logs (gitignored)
```

## Profiling

Compare the unfused path (V0) with the fused vLLM kernel (V1) using Nsight Systems and Nsight Compute.

Guide: [gpu profling - bottleneck/README.md](<gpu profling - bottleneck/README.md>).

```bash
cd "gpu profling - bottleneck"
./run_all.sh
```

## vLLM evaluation

Measure how much of Qwen3-1.7B inference on an NVIDIA L4 is RMSNorm and activation quantization, bound the end-to-end gain with Amdahl's law, and map where a fused kernel would plug into vLLM.

Guide: [vllm-evaluation/README.md](vllm-evaluation/README.md).

```bash
cd vllm-evaluation
VLLM_VERSION=<team-pinned-version> ./setup_env.sh
source .venv/bin/activate
python -m vllm_eval.reporting.selftest
QUICK=1 ./run_all.sh
```
