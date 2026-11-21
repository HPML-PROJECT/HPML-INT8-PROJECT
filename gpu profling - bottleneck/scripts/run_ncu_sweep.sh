#!/usr/bin/env bash
# Nsight Compute sweep: variants x token counts x cache modes.
# Output: reports/ncu/<variant>_<gpu>_H<hidden>_T<tokens>_<cache>.ncu-rep
#
# Cache modes:
#   cold -> ncu --cache-control all   (ncu default: flush caches before every replay pass)
#   warm -> ncu --cache-control none  (no flush; data written/read by earlier passes may stay in L2)
#
# Usage: ./scripts/run_ncu_sweep.sh [HIDDEN]      env: TOKENS="8 1024 4096" VARIANTS="v0 v1" CACHES="cold warm"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HIDDEN="${1:-2048}"
TOKENS="${TOKENS:-8 1024 4096}"
VARIANTS="${VARIANTS:-v0 v1}"
CACHES="${CACHES:-cold warm}"
PY="${PY:-python}"
OUT="${OUT:-reports/ncu}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--no-residual"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')
mkdir -p "$OUT"

# Extra metrics on top of --set full (load width / coalescing, L2 traffic, occupancy)
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,launch__block_size,launch__grid_size,\
sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,\
smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

for cache in $CACHES; do
  case $cache in cold) CC=all;; warm) CC=none;; *) echo "bad cache mode $cache"; exit 1;; esac
  for v in $VARIANTS; do
    for T in $TOKENS; do
      name="${v}_${GPU}_H${HIDDEN}_T${T}_${cache}"
      echo ">>> $name (cache-control=$CC)"
      ncu --set full --metrics "$METRICS" \
          --cache-control "$CC" --clock-control base \
          --nvtx --nvtx-include "target/" \
          -f -o "$OUT/$name" \
          $PY harness/profile_target.py --variant "$v" --tokens "$T" --hidden "$HIDDEN" --iters 3 $EXTRA
    done
  done
done
echo "Done. Reports in $OUT/. Next: python analysis/parse_ncu.py $OUT"
