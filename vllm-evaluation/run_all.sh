#!/usr/bin/env bash
# End-to-end driver (NVIDIA L4). Each step is a separate process (GPU memory is freed between models),
# logs to logs/, and is skipped on re-run unless FORCE=1. QUICK=1 = fast smoke test (few repeats / windows).
#   ./run_all.sh                       full run
#   QUICK=1 ./run_all.sh               smoke test
#   ONLY="bench_w8a8 report" ./run_all.sh
# Steps: env selftest bench_w8a8 bench_bf16 ppl_w8a8 ppl_bf16 profile_w8a8 profile_bf16 profile_w8a8_eager amdahl integration report
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
ONLY="${ONLY:-env selftest bench_w8a8 bench_bf16 ppl_w8a8 ppl_bf16 profile_w8a8 profile_bf16 profile_w8a8_eager amdahl integration report}"
FORCE="${FORCE:-0}"
if [[ "${QUICK:-0}" == 1 ]]; then
  REPEATS=2; WARMUP=1; PPL_ARGS="--max-windows 8"; PROF_ARGS="--decode-steps 8 --warmup 1"; Q=_quick
else
  REPEATS=5; WARMUP=3; PPL_ARGS=""; PROF_ARGS="--decode-steps 32"; Q=""
fi
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--allow-non-l4" or "--gpu-mem 0.8"
mkdir -p logs .stamps results
want() { [[ " $ONLY " == *" $1 "* ]]; }
step() {  # step <name> <command...>
  local name="$1"; shift
  want "$name" || return 0
  local stamp=".stamps/${name}${Q}"
  if [[ -f "$stamp" && "$FORCE" != 1 ]]; then echo "[skip] $name (done; FORCE=1 to redo)"; return 0; fi
  echo "[run ] $name"
  # pipefail is set, so a failing command fails the whole pipeline even though it is piped into tee
  if "$@" 2>&1 | tee "logs/${name}${Q}.log"; then touch "$stamp"
  else echo "[FAIL] $name - see logs/${name}${Q}.log"; exit 1; fi
}

step env          $PY -m vllm_eval.bench.env_info
step selftest     $PY -m vllm_eval.reporting.selftest
step bench_w8a8   $PY -m vllm_eval.bench.bench_perf --tag w8a8 --repeats $REPEATS --warmup $WARMUP $EXTRA
step bench_bf16   $PY -m vllm_eval.bench.bench_perf --tag bf16 --repeats $REPEATS --warmup $WARMUP $EXTRA
step ppl_w8a8     $PY -m vllm_eval.bench.eval_ppl --tag w8a8 $PPL_ARGS $EXTRA
step ppl_bf16     $PY -m vllm_eval.bench.eval_ppl --tag bf16 $PPL_ARGS $EXTRA
step profile_w8a8 $PY -m vllm_eval.profile.profile_breakdown --tag w8a8 $PROF_ARGS $EXTRA
step profile_bf16 $PY -m vllm_eval.profile.profile_breakdown --tag bf16 $PROF_ARGS $EXTRA
step profile_w8a8_eager $PY -m vllm_eval.profile.profile_breakdown --tag w8a8 --enforce-eager $PROF_ARGS $EXTRA
step amdahl       $PY -m vllm_eval.reporting.amdahl --profile w8a8
step integration  $PY -m vllm_eval.integration.inspect_integration
if want report; then echo "[run ] report"; $PY -m vllm_eval.reporting.make_report > logs/report.log; fi

echo; echo "Done. Main outputs: results/REPORT.md, results/amdahl.md, results/integration_map.md, INTEGRATION_PROPOSAL.md"
