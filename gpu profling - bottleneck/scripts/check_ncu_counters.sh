#!/usr/bin/env bash
# Step 0: verify Nsight Compute can read GPU performance counters on this node.
# Usage: ./scripts/check_ncu_counters.sh   (from the toolkit folder, on the L4 VM)
set -u

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv || { echo "nvidia-smi failed: not on a GPU node?"; exit 2; }

echo; echo "== Tools =="
for t in ncu nsys python; do
  if command -v "$t" >/dev/null; then echo "$t: $(command -v $t)"; else echo "$t: MISSING"; fi
done
command -v ncu >/dev/null || { echo "Add /usr/local/cuda/bin to PATH or install Nsight Compute."; exit 2; }
ncu --version | tail -n 2

echo; echo "== Counter access test =="
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/t.py" << 'PY'
import torch
x = torch.randn(1 << 20, device="cuda")
for _ in range(3):
    y = x * 2 + 1
torch.cuda.synchronize()
PY
OUT=$(ncu --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum \
          --launch-count 1 python "$TMP/t.py" 2>&1)
echo "$OUT" | tail -n 15

if echo "$OUT" | grep -q "ERR_NVGPUCTRPERM"; then
  cat << 'MSG'

RESULT: FAIL (ERR_NVGPUCTRPERM) - the driver blocks counters for non-root users.
Options on the L4 VM:
  1. Run as root, keeping PATH and the venv:  sudo -E env "PATH=$PATH" ./run_all.sh
  2. Allow counters for all users permanently:
       echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
       sudo update-initramfs -u && sudo reboot
MSG
  exit 1
elif echo "$OUT" | grep -q "dram__bytes_read.sum"; then
  echo; echo "RESULT: OK - Nsight Compute can read counters on this GPU."
else
  echo; echo "RESULT: UNKNOWN - inspect the output above."; exit 3
fi
