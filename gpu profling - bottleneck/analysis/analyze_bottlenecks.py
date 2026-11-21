#!/usr/bin/env python3
"""
Initial bottleneck analysis from tables/*.csv  ->  tables/bottleneck_analysis.md
Compares V1 (fused) against V0 (unfused total) per (GPU, hidden, cache, tokens) and lists measured evidence
for/against H1 (data-movement limited) and H2 (cache residency). Flags are heuristics: read the numbers.
"""
import csv, math, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")

TH = dict(slow=1.10, fast=0.90, dram_low=60.0, scalar_B=16.0, inst_ratio=1.2)
LATENCY_STALLS = ("long_scoreboard", "lg_throttle", "short_scoreboard", "mio_throttle", "tex_throttle")


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return math.nan


def rd(name):
    p = os.path.join(TABLES, name)
    if not os.path.exists(p): sys.exit(f"missing {p}: run analysis/parse_ncu.py first")
    return list(csv.DictReader(open(p)))


def top_stall(s):
    return s.split(",")[0].split("=")[0].strip() if s else ""


def main():
    mem, exe = rd("memory_table.csv"), rd("execution_table.csv")
    K = lambda r: (r["gpu"], int(r["H"]), r["cache"], int(r["T"]))
    tot, kern_m, kern_e = {}, defaultdict(list), defaultdict(list)
    for r in mem:
        (tot.__setitem__((K(r), r["variant"]), r) if r["kernel"] == "TOTAL" else kern_m[(K(r), r["variant"])].append(r))
    for r in exe:
        if r["kernel"] != "TOTAL": kern_e[(K(r), r["variant"])].append(r)
    keys = sorted({k for k, _ in tot})

    L = ["# Initial bottleneck analysis (auto-generated; verify against the raw tables)\n",
         "Ratios are V1 (fused CUDA) / V0 (unfused pair, summed). >1 means the fused kernel is worse.\n",
         "| GPU | H | cache | T | V0 us | V1 us | time ratio | DRAM MB ratio | V1 DRAM %peak | V1 L2 hit % | V1 regs | V1 occ % | V0 occ % | V1 top stall |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for k in keys:
        t0, t1 = tot.get((k, "v0")), tot.get((k, "v1"))
        if not (t0 and t1): continue
        m1 = kern_m[(k, "v1")][0] if kern_m[(k, "v1")] else {}
        e1 = kern_e[(k, "v1")][0] if kern_e[(k, "v1")] else {}
        e0 = max(kern_e[(k, "v0")], key=lambda r: f(r["inst"]), default={})
        tr = f(t1["time_us"]) / f(t0["time_us"])
        dr = f(t1["dram_MB"]) / f(t0["dram_MB"]) if f(t0["dram_MB"]) else math.nan
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {t0['time_us']} | {t1['time_us']} | {tr:.2f} | {dr:.2f} | "
                 f"{m1.get('DRAM_%peak','')} | {m1.get('L2_hit_%','')} | {e1.get('regs/thr','')} | "
                 f"{e1.get('occ_ach_%','')} | {e0.get('occ_ach_%','')} | {top_stall(e1.get('top_stalls',''))} |")
        tag = f"**{k[0]}, H={k[1]}, T={k[3]}, {k[2]}**"
        verdict = "fused SLOWER" if tr > TH["slow"] else "fused FASTER" if tr < TH["fast"] else "about equal"
        ev = [f"time ratio {tr:.2f} ({verdict})"]
        # H1 evidence: how data is moved
        if f(m1.get("DRAM_%peak")) < TH["dram_low"]: ev.append(f"V1 uses only {m1['DRAM_%peak']}% of peak DRAM throughput -> not bandwidth-saturated")
        if top_stall(e1.get("top_stalls", "")) in LATENCY_STALLS: ev.append(f"top V1 stall is {top_stall(e1['top_stalls'])} (memory-latency/LSU pressure)")
        b1, b0 = f(e1.get("approx_B/ld_inst")), f(e0.get("approx_B/ld_inst"))
        if b1 < TH["scalar_B"] and (math.isnan(b0) or b0 > b1): ev.append(f"V1 ~{b1:.0f} B per global-load instruction vs V0 ~{b0:.0f}: consistent with narrower/scalar loads")
        i1, i0 = f(e1.get("inst")), f(e0.get("inst"))
        if i0 and i1 / i0 > TH["inst_ratio"]: ev.append(f"V1 executes {i1/i0:.1f}x the instructions of V0's heaviest kernel")
        if f(e1.get("occ_ach_%")) < f(e0.get("occ_ach_%")) - 10: ev.append(f"V1 achieved occupancy {e1['occ_ach_%']}% < V0 {e0['occ_ach_%']}% (regs/thread {e1['regs/thr']})")
        if dr > 1.1: ev.append(f"V1 moves {dr:.2f}x the DRAM bytes of V0")
        notes.append(f"- {tag}: " + "; ".join(ev))

    # H2: cold vs warm sensitivity
    L += ["", "## Observations per configuration", *notes, "", "## H2: cache residency (warm/cold time per variant)",
          "| GPU | H | T | V0 warm/cold | V1 warm/cold | V0 L2 hit warm % | V1 L2 hit warm % |", "|---|---|---|---|---|---|---|"]
    seen = set()
    for k in keys:
        base = (k[0], k[1], k[3])
        if base in seen: continue
        seen.add(base)
        c = lambda v, cache, col: f(tot.get(((k[0], k[1], cache, k[3]), v), {}).get(col))
        row = [f"{c(v,'warm','time_us')/c(v,'cold','time_us'):.2f}" for v in ("v0", "v1")]
        hit = [" / ".join(f"{r['kernel'][:24]}: {r['L2_hit_%']}" for r in kern_m.get(((k[0], k[1], 'warm', k[3]), v), [])) for v in ("v0", "v1")]
        L.append(f"| {k[0]} | {k[1]} | {k[3]} | {row[0]} | {row[1]} | {hit[0]} | {hit[1]} |")
    L += ["", "## Caveats", "- Heuristic thresholds: " + str(TH),
          "- Clock setting: see logs/ncu_*.log (base on clusters; locked 2040 MHz + clock-control none on the L4 VM, as in Part 1).",
          "- For V0, the L2 hit rate that matters for H2 is the quantization kernel's (it re-reads the normalized intermediate).",
          "- Low DRAM %peak alone does not prove scalar loads (see proposal H1); confirm with stalls and the load-width columns.",
          "- Warm mode means 'not flushed'; trust the L2 hit rate, not the label.",
          "- Per-kernel detail: tables/memory_table.md and tables/execution_table.md."]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "bottleneck_analysis.md"), "w").write("\n".join(L) + "\n")
    print("wrote tables/bottleneck_analysis.md")


if __name__ == "__main__":
    main()
