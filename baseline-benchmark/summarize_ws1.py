#!/usr/bin/env python3
"""
WS1 summary: L2-crossover table, run-to-run reproducibility and timing noise, plus the clock/power log.

  python summarize_ws1.py                                   # default paths below
  python summarize_ws1.py --main results/l4_20261121 --rerun results/l4_nearL2 --out results/ws1_summary.md

Inputs are the folders written by bench_rmsnorm_int8.py (bench_results.csv, optional gpu_log.csv).
Only the Python standard library is used.
"""
import argparse, csv, math, os, statistics
from collections import defaultdict

KEY = ("dtype", "num_tokens", "hidden_size", "residual", "cache_mode")


def load(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "bench_results.csv"))))
    d = defaultdict(dict)
    for r in rows:
        k = (r["dtype"], int(r["num_tokens"]), int(r["hidden_size"]), int(r["residual"]), r["cache_mode"])
        d[k][r["variant"]] = {c: float(r[c]) for c in ("median_us", "p10_us", "p90_us", "min_us")}
    return d


def ratio(d, k, v, stat="median_us"):
    try:
        return d[k][v][stat] / d[k]["V0"][stat]
    except KeyError:
        return math.nan


def crossover(main, L, dtype="bf16", res=1):
    for cache in ("warm", "cold"):
        hs = sorted({k[2] for k in main if k[0] == dtype and k[3] == res and k[4] == cache})
        L += [f"\n### V1 / V0 time ratio ({dtype}, residual={res}, {cache} cache; >1 = fused V1 slower)\n",
              "| T | " + " | ".join(f"H={h}" for h in hs) + " |", "|---|" + "---|" * len(hs)]
        for T in sorted({k[1] for k in main if k[0] == dtype and k[3] == res and k[4] == cache}):
            cells = []
            for h in hs:
                r = ratio(main, (dtype, T, h, res, cache), "V1")
                cells.append("–" if math.isnan(r) else (f"**{r:.2f}**" if r > 1.1 else f"{r:.2f}"))
            L.append(f"| {T} | " + " | ".join(cells) + " |")


def reproducibility(main, rerun, L):
    L.append("\n## Run-to-run reproducibility (main run vs rerun, shapes present in both)\n")
    L += ["| variant | configurations | same side of 1 | median abs. change in ratio | flips |", "|---|---|---|---|---|"]
    for v in ("V1", "V2"):
        n = same = 0; diffs, flips = [], []
        for k in sorted(set(main) & set(rerun)):
            a, b = ratio(main, k, v), ratio(rerun, k, v)
            if math.isnan(a) or math.isnan(b): continue
            n += 1; s = (a > 1) == (b > 1); same += s
            diffs.append(abs(b / a - 1))
            if not s: flips.append(f"{k[0]} T={k[1]} H={k[2]} res={k[3]} {k[4]}: {a:.2f} vs {b:.2f}")
        if n:
            L.append(f"| {v} | {n} | {same}/{n} | {100 * statistics.median(diffs):.1f}% | {'; '.join(flips) or 'none'} |")


def noise(d, L, title, thresh=0.10):
    L.append(f"\n## Timing spread in {title}: (p90 − p10) / median > {int(thresh * 100)}% (warm cache)\n")
    L += ["| variant | dtype | T | H | residual | median µs | p10 | p90 | spread |", "|---|---|---|---|---|---|---|---|---|"]
    hits = 0
    for k in sorted(d):
        if k[4] != "warm": continue
        for v, m in sorted(d[k].items()):
            s = (m["p90_us"] - m["p10_us"]) / m["median_us"]
            if s > thresh:
                hits += 1
                L.append(f"| {v} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | {m['median_us']:.1f} | {m['p10_us']:.1f} | {m['p90_us']:.1f} | {100 * s:.0f}% |")
    if not hits: L.append("| none | | | | | | | | |")


def gpu_log(path, L):
    if not os.path.exists(path):
        L.append(f"\n(no clock log at {path})"); return
    rows = [r for r in list(csv.reader(open(path)))[1:] if len(r) >= 5]
    busy = [r for r in rows if r[4].strip() != "0x0000000000000001"]      # 0x1 = idle
    if not busy:
        L.append("\n(clock log has no busy samples)"); return
    flags = [int(r[4].strip(), 16) for r in busy]
    mhz = [float(r[1].split()[0]) for r in busy]
    w = [float(r[2].split()[0]) for r in busy]
    capped = sum(1 for f in flags if f & 0x4)
    thermal = sum(1 for f in flags if f & 0x60)
    L += ["\n## Clock and power during the rerun (nvidia-smi every 0.5 s)\n",
          f"- busy samples: {len(busy)}; SM clock median {statistics.median(mhz):.0f} MHz, min {min(mhz):.0f} MHz",
          f"- power: median {statistics.median(w):.1f} W, max {max(w):.1f} W (L4 limit 72 W)",
          f"- **software power cap active in {capped}/{len(busy)} samples ({100 * capped / len(busy):.0f}%)**; thermal slowdown in {thermal}",
          "- Locking the clock with `nvidia-smi -lgc` does not override the power cap."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/l4_20261121")
    ap.add_argument("--rerun", default="results/l4_nearL2")
    ap.add_argument("--out", default="results/ws1_summary.md")
    a = ap.parse_args()
    main_d, rerun_d = load(a.main), load(a.rerun)
    L = [f"# WS1 summary\n\nmain run: `{a.main}`  |  rerun: `{a.rerun}`"]
    L.append("\n## L2 crossover (main run)")
    crossover(main_d, L)
    reproducibility(main_d, rerun_d, L)
    noise(rerun_d, L, "the rerun (200 samples)")
    gpu_log(os.path.join(a.rerun, "gpu_log.csv"), L)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
