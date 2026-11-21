#!/usr/bin/env bash
# Nsight Systems timelines + kernel summaries (launch gaps, kernel counts, V0's 2 kernels vs V1's 1).
# Usage: ./scripts/run_nsys.sh [HIDDEN]     env: TOKENS="8 1024 4096" VARIANTS="v0 v1"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HIDDEN="${1:-2048}"; TOKENS="${TOKENS:-8 1024 4096}"; VARIANTS="${VARIANTS:-v0 v1}"
PY="${PY:-python}"; OUT="${OUT:-reports/nsys}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')
mkdir -p "$OUT"
for v in $VARIANTS; do for T in $TOKENS; do
  name="${v}_${GPU}_H${HIDDEN}_T${T}"
  nsys profile -t cuda,nvtx -o "$OUT/$name" -f true \
       $PY harness/profile_target.py --variant "$v" --tokens "$T" --hidden "$HIDDEN" --iters 20
  nsys stats -r cuda_gpu_kern_sum --format csv "$OUT/$name.nsys-rep" > "$OUT/$name.kern_sum.csv"
done; done
echo "Done. Reports in $OUT/"
