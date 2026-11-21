#!/usr/bin/env bash
# End-to-end driver for WS2. Run on the Google Cloud L4 VM (repeat on any other GPU type added later).
# Every step logs to logs/, writes a stamp in .stamps/, and is skipped on re-run (FORCE=1 to redo).
#   ./run_all.sh                         # full plan, hidden 2048 and 4096
#   HIDDENS="2048" TOKENS="8 1024 4096" ./run_all.sh
#   ONLY="verify ncu" ./run_all.sh       # run selected steps only
# Steps: counters -> verify -> bench -> nsys -> ncu -> parse -> analyze
set -euo pipefail
cd "$(dirname "$0")"
HIDDENS="${HIDDENS:-2048 4096}"
export TOKENS="${TOKENS:-8 1024 4096}"
ONLY="${ONLY:-counters verify bench nsys ncu parse analyze}"
FORCE="${FORCE:-0}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')
mkdir -p logs .stamps reports/ncu reports/nsys tables
PY="${PY:-python}"
want() { [[ " $ONLY " == *" $1 "* ]]; }
step() {  # step <name> <command...>
  local name="$1"; shift
  want "$name" || return 0
  local stamp=".stamps/${name}_${GPU}"
  if [[ -f "$stamp" && "$FORCE" != 1 ]]; then echo "[skip] $name (done on $GPU; FORCE=1 to redo)"; return 0; fi
  echo "[run ] $name on $GPU"
  if "$@" 2>&1 | tee "logs/${name}_${GPU}.log"; then touch "$stamp"; else
    echo "[FAIL] $name - see logs/${name}_${GPU}.log"; exit 1; fi
}

# 1. Counter access (aborts the pipeline if blocked; then rerun with sudo -E env "PATH=$PATH" ./run_all.sh)
step counters ./scripts/check_ncu_counters.sh

# 2. Fairness gate: V0 and V1 must do identical work (scales, residual, dequant error, INT8 mismatch)
verify_all() { for H in $HIDDENS; do for T in $TOKENS; do "$PY" harness/profile_target.py --mode verify --tokens "$T" --hidden "$H"; done; done; }
step verify verify_all

# 3. Quick event-timing sanity (NOT authoritative; shows the slowdown direction before the long profiles)
bench_all() { for H in $HIDDENS; do for T in $TOKENS; do echo "T=$T H=$H"; "$PY" harness/profile_target.py --mode bench --tokens "$T" --hidden "$H"; done; done; }
step bench bench_all

# 4. Nsight Systems (timeline, kernel counts, gaps)
nsys_all() { for H in $HIDDENS; do ./scripts/run_nsys.sh "$H"; done; }
step nsys nsys_all

# 5. Nsight Compute: variants x tokens x {cold,warm} x hidden
ncu_all() { for H in $HIDDENS; do ./scripts/run_ncu_sweep.sh "$H"; done; }
step ncu ncu_all

# 6. Tables  7. Bottleneck analysis
FORCE_PARSE=1  # cheap; always regenerate from all GPUs' reports
if want parse;   then echo "[run ] parse";   "$PY" analysis/parse_ncu.py reports/ncu | tee "logs/parse_${GPU}.log"; fi
if want analyze; then echo "[run ] analyze"; "$PY" analysis/analyze_bottlenecks.py | tee "logs/analyze_${GPU}.log"; fi

echo; echo "Done. Deliverables:"
echo "  reports/ncu/*.ncu-rep, reports/nsys/*      (labeled variant_gpu_H_T_cache)"
echo "  tables/memory_table.md, tables/execution_table.md"
echo "  tables/bottleneck_analysis.md"
