#!/usr/bin/env bash
# Run the whole WS2 profiling plan on the Google Cloud L4 VM, aligned with the Part 1 run.
# Reuses Part 1's vLLM environment (~/rmsnorm_int8_bench/.venv), requests a 2040 MHz SM clock like Part 1, and runs
# ncu with sudo (Google Cloud blocks GPU counters for normal users). The L4's 72 W power cap can still pull the
# clock below 2040 MHz under load, so absolute times vary; compare variants by ratio.
#   cd ~/HPML-INT8-PROJECT/"gpu profling - bottleneck" && bash run_l4_vm.sh   # full plan (~1 h)
#   QUICK=1 bash run_l4_vm.sh                # 1 hidden size, 2 token counts (~10 min) to test the setup
set -euo pipefail
cd "$(dirname "$0")"
PART1_DIR="${PART1_DIR:-$HOME/rmsnorm_int8_bench}"
export PATH="/usr/local/cuda/bin:$PATH"
for d in /usr/local/cuda*/nsight-systems*/bin /opt/nvidia/nsight-systems/*/bin; do [ -d "$d" ] && PATH="$PATH:$d"; done
# shellcheck disable=SC1091
source "$PART1_DIR/.venv/bin/activate"

if [ "${QUICK:-0}" = 1 ]; then
  export HIDDENS="${HIDDENS:-2048}" TOKENS="${TOKENS:-256 16384}"
else
  # Part 1 showed: V1 slower at 128-4096 tokens (warm), faster at >= 8192. Profile both sides of the switch.
  export HIDDENS="${HIDDENS:-2048 4096}" TOKENS="${TOKENS:-8 256 1024 4096 16384}"
fi
export CLOCK_CONTROL="${CLOCK_CONTROL:-none}" PART1_DIR

echo "== checks =="
command -v ncu  >/dev/null || { echo "ncu not found. Install the CUDA toolkit: sudo python3 ~/cuda_installer.pyz install_cuda"; exit 2; }
command -v nsys >/dev/null || { echo "nsys not found. Add the nsight-systems bin folder to PATH (ls -d /usr/local/cuda*/nsight-systems*/bin)"; exit 2; }
ncu --version | tail -1; nsys --version
python -c "import torch, vllm, triton; print('torch', torch.__version__, '| vllm', vllm.__version__, '| triton', triton.__version__, '|', torch.cuda.get_device_name())"
echo "HIDDENS=$HIDDENS  TOKENS=$TOKENS  clock-control=$CLOCK_CONTROL"

echo "== request SM clock 2040 MHz (same as Part 1; the power cap may still lower it) =="
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -lgc 2040,2040
# always release the locked clock, even if a later step fails
trap 'sudo nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT

sudo -E env "PATH=$PATH" HIDDENS="$HIDDENS" TOKENS="$TOKENS" CLOCK_CONTROL="$CLOCK_CONTROL" \
     PART1_DIR="$PART1_DIR" ./run_all.sh

echo "== compare with Part 1 =="
if [ -f "$PART1_DIR/results_l4/bench_results.csv" ]; then
  sudo -E env "PATH=$PATH" python analysis/compare_part1.py "$PART1_DIR/results_l4/bench_results.csv" || true
else
  echo "Part 1 results not found in $PART1_DIR/results_l4 - skipped (run compare_part1.py later with the CSV path)"
fi

sudo nvidia-smi -rgc >/dev/null || true
OUTTAR="$HOME/ws2_results_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')_$(date +%Y%m%d_%H%M%S).tar.gz"
sudo tar czf "$OUTTAR" reports tables logs
sudo chown "$(id -u):$(id -g)" "$OUTTAR"
sudo chown -R "$(id -u):$(id -g)" reports tables logs .stamps 2>/dev/null || true
echo; echo "Bundle: $OUTTAR   <- download this (SSH window: Download file), then stop the VM"
