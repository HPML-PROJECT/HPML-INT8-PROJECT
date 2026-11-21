#!/usr/bin/env python3
"""
Alignment check: Part 2 (Nsight Compute kernel time) vs Part 1 (CUDA-graph benchmark) for the same shapes.
  python analysis/compare_part1.py [PART1_bench_results.csv]
Default Part 1 CSV: $PART1_CSV or ~/rmsnorm_int8_bench/results_l4/bench_results.csv
Writes tables/part1_vs_ncu.md. The absolute times will differ (ncu replays kernels one at a time, Part 1 runs
calls back to back), but the V1/V0 ratio should point the same way; where it does not, say so in the report.
"""
import csv, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return math.nan


def main():
    p1 = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "PART1_CSV", os.path.expanduser("~/rmsnorm_int8_bench/results_l4/bench_results.csv"))
    mt = os.path.join(TABLES, "memory_table.csv")
    if not os.path.exists(p1): sys.exit(f"Part 1 CSV not found: {p1}")
    if not os.path.exists(mt): sys.exit("tables/memory_table.csv missing: run analysis/parse_ncu.py first")

    part1 = {}
    for r in csv.DictReader(open(p1)):
        if r["dtype"] == "bf16" and r["residual"] == "1":
            part1[(int(r["hidden_size"]), int(r["num_tokens"]), r["cache_mode"], r["variant"].lower())] = f(r["median_us"])
    ncu = {}
    for r in csv.DictReader(open(mt)):
        if r["kernel"] == "TOTAL":
            ncu[(r["gpu"], int(r["H"]), int(r["T"]), r["cache"], r["variant"])] = f(r["time_us"])

    L = ["# Part 1 vs Part 2 alignment (BF16, residual add)\n",
         "Part 1 = CUDA-graph median (µs; 2040 MHz requested, power-capped). Part 2 = Nsight Compute kernel time (µs, sum of V0's two kernels).",
         "Ratio = V1 / V0 (>1: fused is slower). 'agree' = both ratios on the same side of 1 (±10%).\n",
         "| GPU | H | T | cache | P1 V0 | P1 V1 | P1 ratio | ncu V0 | ncu V1 | ncu ratio | agree |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    side = lambda x: 0 if math.isnan(x) else (1 if x > 1.1 else -1 if x < 0.9 else 0)
    n = bad = 0
    for (gpu, H, T, cache, v) in sorted(ncu):
        if v != "v0": continue
        a0, a1 = part1.get((H, T, cache, "v0"), math.nan), part1.get((H, T, cache, "v1"), math.nan)
        b0, b1 = ncu.get((gpu, H, T, cache, "v0"), math.nan), ncu.get((gpu, H, T, cache, "v1"), math.nan)
        ra = a1 / a0 if a0 else math.nan
        rb = b1 / b0 if b0 else math.nan
        ag = "–" if math.isnan(ra) or math.isnan(rb) else ("yes" if side(ra) == side(rb) else "**NO**")
        n += 1; bad += ag == "**NO**"
        L.append(f"| {gpu} | {H} | {T} | {cache} | {a0:.1f} | {a1:.1f} | {ra:.2f} | {b0:.1f} | {b1:.1f} | {rb:.2f} | {ag} |")
    L += ["", f"{n - bad}/{n} configurations agree on direction.",
          "Disagreements usually mean the profiler's replay changed the cache state (most likely in 'warm' mode, "
          "where V0's second kernel may lose the L2 reuse it gets in a real back-to-back run). Check L2 hit rates."]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "part1_vs_ncu.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote tables/part1_vs_ncu.md")


if __name__ == "__main__":
    main()
