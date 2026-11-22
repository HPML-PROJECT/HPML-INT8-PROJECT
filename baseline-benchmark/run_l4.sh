#!/usr/bin/env bash
# Setup + full baseline run on an NVIDIA L4 VM (Linux, NVIDIA driver installed).
#
#   bash run_l4.sh --quick     # smoke test, ~1-2 min
#   bash run_l4.sh             # full run: fp16+bf16, residual off+on, warm+cold
#
# Env knobs:
#   VLLM_VERSION=0.x.y   pin vLLM (recommended: whole team uses the same one)
#   USE_VENV=0           use the current python env instead of ./.venv (e.g. a conda env that already has vllm)
#   LOCK_CLOCKS=1        lock SM clocks with sudo nvidia-smi for steadier numbers
#   OUT=dir              output dir (default results_l4)
# Any extra args go to bench_rmsnorm_int8.py (see: python bench_rmsnorm_int8.py --help).
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-results_l4}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is)  run_l4.sh $* ==="

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv

# ---------------- python env ----------------
if [ "${USE_VENV:-1}" = "1" ]; then
  if [ ! -d .venv ]; then
    if ! python3 -m venv .venv 2>/dev/null; then
      echo "python3-venv missing, installing..."
      sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip
      python3 -m venv .venv
    fi
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pip install -q --upgrade pip
if ! python -c "import vllm" 2>/dev/null; then
  if [ -n "${VLLM_VERSION:-}" ]; then pip install "vllm==${VLLM_VERSION}"; else pip install vllm; fi
fi
python -c "import matplotlib" 2>/dev/null || pip install -q matplotlib

# ---------------- preflight ----------------
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
import vllm, vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
C = torch.ops._C
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | vllm {vllm.__version__} | {torch.cuda.get_device_name(0)}")
for op in ["rms_norm", "fused_add_rms_norm", "dynamic_scaled_int8_quant", "rms_norm_dynamic_per_token_quant"]:
    print(f"  torch.ops._C.{op:36s}", "OK" if hasattr(C, op) else "MISSING")
try:
    import triton; print("  triton", triton.__version__)
except Exception as e:
    print("  triton MISSING:", e)
EOF

# ---------------- optional clock lock ----------------
if [ "${LOCK_CLOCKS:-0}" = "1" ]; then
  sudo nvidia-smi -pm 1 || true
  MAXCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
  sudo nvidia-smi -lgc "$MAXCLK","$MAXCLK" || true
  trap 'sudo nvidia-smi -rgc || true' EXIT
fi

# ---------------- run ----------------
set +e
python bench_rmsnorm_int8.py --out-dir "$OUT" --residual both "$@"
RC=$?
set -e

nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv >> "$LOG" || true
TAR="${OUT}_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$TAR" "$OUT"
echo
echo "Exit code: $RC   (0 = all cross-checks passed, 2 = some check failed -> see $OUT/correctness.csv)"
echo "Results:   $OUT/bench_results.csv  $OUT/correctness.csv  $OUT/env.json  $OUT/plots/"
echo "Bundle:    $(pwd)/$TAR   <- copy this back (scp / gcloud compute scp)"
exit $RC
