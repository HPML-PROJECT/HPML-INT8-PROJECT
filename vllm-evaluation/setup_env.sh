#!/usr/bin/env bash
# One-time setup on an NVIDIA L4 machine (e.g. Google Cloud g2-standard-* VM with an L4).
# Usage:  VLLM_VERSION=0.x.y ./setup_env.sh        (VLLM_VERSION empty = latest stable; PIN IT for the project)
set -euo pipefail
cd "$(dirname "$0")"

echo "== Checking GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "nvidia-smi failed: install the NVIDIA driver first."; exit 2; }
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qE '\bL4\b'; then
  echo "WARNING: GPU is not an NVIDIA L4. Scripts will refuse to run unless --allow-non-l4 / ALLOW_NON_L4=1 is set."
fi

echo "== Creating virtual environment (.venv) =="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

echo "== Installing vLLM ${VLLM_VERSION:-<latest>} =="
if [[ -n "${VLLM_VERSION:-}" ]]; then pip install "vllm==${VLLM_VERSION}"; else pip install vllm; fi
pip install -r requirements.txt

echo "== Verifying =="
python - << 'PY'
import torch, vllm
print("torch", torch.__version__, "| vllm", vllm.__version__)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0))
from vllm import _custom_ops as ops
need = ["rms_norm", "fused_add_rms_norm", "scaled_int8_quant", "rms_norm_dynamic_per_token_quant"]
print({n: hasattr(ops, n) for n in need})
PY

cat << 'MSG'

Setup done. Next:
  source .venv/bin/activate
  # optional (gated/rate-limited downloads):  huggingface-cli login
  python -m vllm_eval.reporting.selftest   # offline logic tests (no GPU needed)
  QUICK=1 ./run_all.sh        # smoke test, a few minutes
  ./run_all.sh                # full run
MSG
