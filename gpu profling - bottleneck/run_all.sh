#!/usr/bin/env bash
# End-to-end driver for WS2. Run on a GPU node; repeat once per GPU type (A6000, L40S, H100 if available).
# Every step logs to logs/, writes a stamp in .stamps/, and is skipped on re-run (FORCE=1 to redo).
#   ./run_all.sh                         # full plan, hidden 2048 and 4096
#   HIDDENS="2048" TOKENS="256 16384" ./run_all.sh
# On the Google Cloud L4 VM use run_l4_vm.sh instead (sudo for counters, clocks locked like Part 1).
#   ONLY="verify ncu" ./run_all.sh       # run selected steps only
# Steps: counters -> verify -> bench -> nsys -> ncu -> parse -> analyze
set -euo pipefail
cd "$(dirname "$0")"
HIDDENS="${HIDDENS:-2048 4096}"
export TOKENS="${TOKENS:-8 256 1024 4096 16384}"   # Part 1: V1 slow at 128-4096, fast at >=8192
ONLY="${ONLY:-counters verify bench nsys ncu parse analyze}"
FORCE="${FORCE:-0}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')
mkdir -p logs .stamps reports/ncu reports/nsys tables
PY="${PY:-python}"
# Resume stamps are keyed by GPU AND run configuration, so a QUICK/partial run never makes a later full run skip steps.
CFG=$(printf '%s' "$HIDDENS|$TOKENS|${VARIANTS:-}|${CACHES:-}|${CLOCK_CONTROL:-}|${EXTRA:-}" | cksum | cut -d' ' -f1)
want() { [[ " $ONLY " == *" $1 "* ]]; }
step() {  # step <name> <command...>
  local name="$1"; shift
  want "$name" || return 0
  local stamp=".stamps/${name}_${GPU}_${CFG}"
  if [[ -f "$stamp" && "$FORCE" != 1 ]]; then echo "[skip] $name (done on $GPU with this configuration; FORCE=1 to redo)"; return 0; fi
  echo "[run ] $name on $GPU"
  if "$@" 2>&1 | tee "logs/${name}_${GPU}.log"; then touch "$stamp"; else
    echo "[FAIL] $name - see logs/${name}_${GPU}.log"; exit 1; fi
}

# 1. Counter access (aborts the pipeline if blocked; use the L4 fallback then)
step counters ./scripts/check_ncu_counters.sh

# 2. Fairness gate: V0 and V1 must do identical work, checked with Part 1's inputs, reference and tolerances.
#    Exits on FAIL. Also confirms inputs are bit-identical to Part 1 (if PART1_DIR has bench_rmsnorm_int8.py).
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
# parse/analyze are cheap and always regenerate from every report in reports/ncu
if want parse;   then echo "[run ] parse";   "$PY" analysis/parse_ncu.py reports/ncu | tee "logs/parse_${GPU}.log"; fi
if want analyze; then echo "[run ] analyze"; "$PY" analysis/analyze_bottlenecks.py | tee "logs/analyze_${GPU}.log"; fi

echo; echo "Done. Deliverables:"
echo "  reports/ncu/*.ncu-rep, reports/nsys/*      (labeled variant_gpu_H_T_cache)"
echo "  tables/memory_table.md, tables/execution_table.md"
echo "  tables/bottleneck_analysis.md"
