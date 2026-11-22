#!/usr/bin/env bash
# Part 3 on the L4 VM: V0/V1/V2 + all V3 configs in ONE run (same inputs, timing, tolerances as Part 1),
# with the SM clock locked and clocks/power logged, then a V3 summary.
#   cd triton-v3 && bash run_part3_l4.sh                        # ~10-15 min
#   QUICK=1 bash run_part3_l4.sh                                # ~2 min smoke test
#   CLOCK=2040 bash run_part3_l4.sh                             # other locked clock (default 1200)
set -euo pipefail
cd "$(dirname "$0")"
CLOCK="${CLOCK:-1200}"   # 1200 MHz: below the L4 72 W power cap (used for results/l4_part3)
OUT="${OUT:-results_l4_part3}"
if [ "${QUICK:-0}" = 1 ]; then
  TOK="256 4096"; HID="2048"; REPS=30; OUT="${OUT}_quick"
else
  TOK="8 64 256 1024 2048 4096 8192 16384"; HID="2048 4096"; REPS=100
fi
# Python env: ./.venv if present, else $VENV, else the Part 1 VM env (~/rmsnorm_int8_bench/.venv)
VENV="${VENV:-$( [ -d .venv ] && echo .venv || echo "$HOME/rmsnorm_int8_bench/.venv" )}"
source "$VENV/bin/activate"
mkdir -p "$OUT"
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -lgc "$CLOCK","$CLOCK"
trap 'sudo nvidia-smi -rgc >/dev/null 2>&1 || true; kill $LOGPID 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
           --format=csv -lms 500 > "$OUT/gpu_log.csv" &
LOGPID=$!
set +e
python bench_rmsnorm_int8.py --variants V0 V1 V2 V3 --residual on --dtypes bf16 \
       --hidden $HID --tokens $TOK --reps "$REPS" --out-dir "$OUT"
RC=$?
set -e
kill $LOGPID 2>/dev/null || true
python summarize_v3.py "$OUT" || true
tar czf "$HOME/${OUT}.tar.gz" "$OUT"
echo; echo "Exit code: $RC (0 = all correctness checks passed)"
echo "Bundle: $HOME/${OUT}.tar.gz   <- download this, then stop the VM"
