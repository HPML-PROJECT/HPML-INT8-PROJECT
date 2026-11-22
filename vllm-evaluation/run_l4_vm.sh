#!/usr/bin/env bash
# Part 4 on the Google Cloud L4 VM, reusing Part 1's vLLM environment (same vLLM 0.34.0 as Parts 1-3, no second
# multi-GB venv). Locks the SM clock (the L4 hits its 72 W power cap at 2040 MHz) and logs clocks/power.
#   cd ~/vllm-evaluation && QUICK=1 bash run_l4_vm.sh      # smoke test (~10-15 min incl. model downloads)
#   cd ~/vllm-evaluation && bash run_l4_vm.sh              # full run (~1-1.5 h)
#   CLOCK=0 bash run_l4_vm.sh                              # do not lock clocks (state it in the report)
set -euo pipefail
cd "$(dirname "$0")"
PART1_DIR="${PART1_DIR:-$HOME/rmsnorm_int8_bench}"
CLOCK="${CLOCK:-1200}"
source "$PART1_DIR/.venv/bin/activate"
python -m pip install -q -r requirements.txt
echo "== disk =="; df -h / | tail -1
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 12 ]; then echo "Only ${FREE_GB} GB free: models + datasets need ~10 GB. Free space or enlarge the disk."; exit 2; fi
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

mkdir -p results logs
if [ "$CLOCK" != 0 ]; then
  sudo nvidia-smi -pm 1 >/dev/null
  sudo nvidia-smi -lgc "$CLOCK","$CLOCK"
fi
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
           --format=csv -l 2 > "results/gpu_log$( [ "${QUICK:-0}" = 1 ] && echo _quick ).csv" &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null || true; [ "$CLOCK" != 0 ] && sudo nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT

set +e
./run_all.sh
RC=$?
set -e
echo "{\"locked_sm_clock_mhz\": \"$CLOCK\"}" > results/clock_setting.json
TAR="$HOME/part4_results$( [ "${QUICK:-0}" = 1 ] && echo _quick )_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$TAR" results logs INTEGRATION_PROPOSAL.md
echo; echo "run_all exit code: $RC (0 = all steps finished)"
echo "Bundle: $TAR   <- download this, then stop the VM"
exit $RC
