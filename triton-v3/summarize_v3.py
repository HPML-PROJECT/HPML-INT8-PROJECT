#!/usr/bin/env python3
"""Summarize a Part 3 run: correctness, best V3 config per shape, V3 vs V0/V1/V2, clock stability.
   python summarize_v3.py results_l4_part3   -> results_l4_part3/part3_summary.md"""
import csv, math, os, statistics, sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "results_l4_part3"
rows = list(csv.DictReader(open(os.path.join(d, "bench_results.csv"))))
checks = list(csv.DictReader(open(os.path.join(d, "correctness.csv"))))
f = lambda x: float(x) if x not in ("", None) else math.nan
L = [f"# Part 3 summary ({d})\n"]

# correctness
bad = [c for c in checks if c["passed"] not in ("True", "1")]
v3c = [c for c in checks if c["variant"].startswith("V3")]
L.append(f"**Correctness:** {len(checks) - len(bad)}/{len(checks)} comparisons passed "
         f"(V3: {sum(c['passed'] in ('True','1') for c in v3c)}/{len(v3c)}).")
for c in bad[:10]:
    L.append(f"- FAIL {c['variant']} vs {c['compared_to']} T={c['num_tokens']} H={c['hidden_size']}: "
             f"q_diff={c['q_max_abs_diff']} mism={c['q_mismatch_frac']} lsb={c['dequant_max_err_lsb']} {c['note']}")
lsb = [f(c["dequant_max_err_lsb"]) for c in v3c if c["compared_to"] == "ref_fp32"]
mis = [f(c["q_mismatch_frac"]) for c in v3c if c["compared_to"] == "V0"]
if lsb: L.append(f"V3 max dequant error vs FP32: {max(lsb):.2f} INT8 steps; V3 vs V0 INT8 mismatch: max {max(mis)*100:.1f}%.\n")

# timing tables
t = defaultdict(dict)   # (cache, H, T) -> variant -> (median, p10)
for r in rows:
    t[(r["cache_mode"], int(r["hidden_size"]), int(r["num_tokens"]))][r["variant"]] = (f(r["median_us"]), f(r["p10_us"]))
v3names = sorted({r["variant"] for r in rows if r["variant"].startswith("V3")})
for cache in ("warm", "cold"):
    keys = sorted(k for k in t if k[0] == cache)
    if not keys: continue
    L += [f"## {cache} cache: median µs (best V3 config per shape)\n",
          "| H | T | V0 | V1 | V2 | best V3 | config | V3 vs V0 | V3 vs V1 | V3 vs V2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        v = t[k]
        best = min((n for n in v3names if n in v), key=lambda n: v[n][0], default=None)
        if not best: continue
        b = v[best][0]
        sp = lambda n: f"{v[n][0] / b:.2f}×" if n in v else "–"
        L.append(f"| {k[1]} | {k[2]} | {v.get('V0',(math.nan,))[0]:.1f} | {v.get('V1',(math.nan,))[0]:.1f} | "
                 f"{v.get('V2',(math.nan,))[0]:.1f} | {b:.1f} | {best[3:]} | {sp('V0')} | {sp('V1')} | {sp('V2')} |")
    L.append("")
    # config ablation: geo-mean speedup vs V0 per config
    L += [f"### {cache}: each V3 config, geo-mean speedup vs V0 over all shapes\n", "| config | geo-mean vs V0 | geo-mean vs V2 | shapes where best |", "|---|---|---|---|"]
    for n in v3names:
        s0 = [t[k]["V0"][0] / t[k][n][0] for k in keys if n in t[k] and "V0" in t[k]]
        s2 = [t[k]["V2"][0] / t[k][n][0] for k in keys if n in t[k] and "V2" in t[k]]
        nb = sum(1 for k in keys if n in t[k] and n == min((m for m in v3names if m in t[k]), key=lambda m: t[k][m][0]))
        gm = lambda s: math.exp(sum(map(math.log, s)) / len(s)) if s else math.nan
        L.append(f"| {n[3:]} | {gm(s0):.2f}× | {gm(s2):.2f}× | {nb} |")
    L.append("")

# clock stability
gl = os.path.join(d, "gpu_log.csv")
if os.path.exists(gl):
    g = list(csv.reader(open(gl)))[1:]
    busy = [r for r in g if len(r) >= 5 and r[4].strip() != "0x0000000000000001"]
    capped = [r for r in busy if int(r[4].strip(), 16) & 0x4]
    mhz = [float(r[1].split()[0]) for r in busy]
    if busy:
        L.append(f"**Clock log:** {len(busy)} busy samples, SM clock median {statistics.median(mhz):.0f} MHz "
                 f"(min {min(mhz):.0f}); power-capped in {len(capped)} ({100*len(capped)/len(busy):.0f}%). "
                 + ("Clock was stable." if not capped else "Power capping present: treat small differences with care."))
open(os.path.join(d, "part3_summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
