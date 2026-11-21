#!/usr/bin/env bash
# Step 0: verify Nsight Compute can read GPU performance counters on this node.
# Usage: ./scripts/check_ncu_counters.sh   (from ws2/, on a GPU node)
set -u

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv || { echo "nvidia-smi failed: not on a GPU node?"; exit 2; }

echo; echo "== Tools =="
for t in ncu nsys python; do
  if command -v "$t" >/dev/null; then echo "$t: $(command -v $t)"; else echo "$t: MISSING"; fi
done
command -v ncu >/dev/null || { echo "Load a CUDA module (e.g. 'module load cuda') or install Nsight Compute."; exit 2; }
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

RESULT: FAIL (ERR_NVGPUCTRPERM) - the driver blocks counters for non-admin users.
Options:
  1. Ask Insomnia admins to enable profiling for your account/partition
     (driver option NVreg_RestrictProfilingToAdminUsers=0, or a profiling-enabled node).
  2. FALLBACK: Google Cloud L4 VM (Ada, sm_89 - same architecture as the L40S, 24 GB, 48 MB L2).
     Counters work as root/sudo:  sudo -E env "PATH=$PATH" ./scripts/run_ncu_sweep.sh
     Record in the report that the L4 has a different L2 size / bandwidth than the L40S.
MSG
  exit 1
elif echo "$OUT" | grep -q "dram__bytes_read.sum"; then
  echo; echo "RESULT: OK - Nsight Compute can read counters on this GPU."
else
  echo; echo "RESULT: UNKNOWN - inspect the output above."; exit 3
fi
