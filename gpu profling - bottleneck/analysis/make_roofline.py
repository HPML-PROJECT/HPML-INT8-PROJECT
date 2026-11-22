#!/usr/bin/env python3
"""
Roofline for a WS2 results folder (V0 vs V1, from Nsight Compute's measured time and bytes).
  python analysis/make_roofline.py results/l4_20261121
Reads   <dir>/tables/memory_table.csv
Writes  <dir>/plots/roofline.png, <dir>/tables/roofline_table.{csv,md}

Model
- Work: the same arithmetic in every variant, counted analytically per element:
  residual add, square, sum, x*rstd, *w, abs, max, /scale, round = FLOPS_PER_ELEM (per-row terms are negligible).
- Cold runs (L2 flushed): DRAM roofline. Intensity = FLOPs / measured DRAM bytes.
- Warm runs (data in L2): L2 roofline. Intensity = FLOPs / measured L2 bytes. The L4's L2 bandwidth is not
  published, so the L2 roof is the highest L2 throughput measured by any kernel in this run.
V0 = sum of its two kernels (time and bytes).
"""
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NVIDIA L4 specs
DRAM_BW = 300e9          # bytes/s
FP32_PEAK = 30.3e12      # FLOP/s
FLOPS_PER_ELEM = 9       # residual-add mode (all WS2 runs)
L2_RESIDENT_FRAC = 0.05  # warm run is "L2-resident" if DRAM bytes < 5% of L2 bytes

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK,
})


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def main(d):
    M = list(csv.DictReader(open(os.path.join(d, "tables", "memory_table.csv"))))
    # empirical L2 roof: highest L2 throughput of any single kernel in the run
    l2_roof = max(f(r["L2_MB"]) * 1e6 / (f(r["time_us"]) * 1e-6) for r in M if r["kernel"] != "TOTAL" and f(r["time_us"]) > 0)

    pts = []
    for r in M:
        if r["kernel"] != "TOTAL": continue
        T, H, t = int(r["T"]), int(r["H"]), f(r["time_us"]) * 1e-6
        flops = FLOPS_PER_ELEM * T * H
        # Which memory level serves the data: a warm run counts as L2-resident when its DRAM traffic is under
        # L2_RESIDENT_FRAC of its L2 traffic; otherwise (cold, or warm with a working set larger than L2) DRAM.
        l2_resident = r["cache"] == "warm" and f(r["dram_MB"]) < L2_RESIDENT_FRAC * f(r["L2_MB"])
        level = "L2" if l2_resident else "DRAM"
        nbytes = (f(r["L2_MB"]) if l2_resident else f(r["dram_MB"])) * 1e6
        roof_bw = l2_roof if l2_resident else DRAM_BW
        ai = flops / nbytes if nbytes > 0 else float("nan")
        perf = flops / t
        roof = min(FP32_PEAK, ai * roof_bw)
        pts.append(dict(variant=r["variant"].upper(), H=H, T=T, cache=r["cache"],
                        level=level,
                        time_us=round(t * 1e6, 2), GFLOP=round(flops / 1e9, 4),
                        bytes_MB=round(nbytes / 1e6, 3), AI_flop_per_byte=round(ai, 3),
                        GFLOPs_per_s=round(perf / 1e9, 1), achieved_GBps=round(nbytes / t / 1e9, 1),
                        pct_of_roof=round(100 * perf / roof, 1)))

    out_t = os.path.join(d, "tables"); out_p = os.path.join(d, "plots"); os.makedirs(out_p, exist_ok=True)
    pts.sort(key=lambda p: (p["cache"], p["H"], p["T"], p["variant"]))
    cols = list(pts[0].keys())
    with open(os.path.join(out_t, "roofline_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(pts)
    with open(os.path.join(out_t, "roofline_table.md"), "w", encoding="utf-8") as fh:
        fh.write(f"Roofs: DRAM {DRAM_BW/1e9:.0f} GB/s (spec), L2 {l2_roof/1e9:.0f} GB/s (highest measured), "
                 f"FP32 {FP32_PEAK/1e12:.1f} TFLOP/s (spec). Work = {FLOPS_PER_ELEM} FLOPs per element.\n\n")
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for p in pts: fh.write("| " + " | ".join(str(p[c]) for c in cols) + " |\n")

    # plot: rows = cache mode (cold -> DRAM roofline, warm -> L2 roofline), cols = hidden size
    Hs = sorted({p["H"] for p in pts})
    fig, axes = plt.subplots(2, len(Hs), figsize=(5.4 * len(Hs), 8.2), sharex=False, sharey=True)
    def roof_line(ax, bw, **kw):
        ridge = FP32_PEAK / bw                      # include the ridge point so the roof bends to flat
        xs = [0.1, ridge, 1000]
        ax.plot(xs, [min(FP32_PEAK, x * bw) / 1e9 for x in xs], color=INK2, zorder=2, **kw)

    for i, (cache, level, bw) in enumerate((("cold", "DRAM", DRAM_BW), ("warm", "L2", l2_roof))):
        for j, H in enumerate(Hs):
            ax = axes[i][j]
            ax.set_xscale("log"); ax.set_yscale("log")
            # roofs, labeled along the sloped part away from the data (data sits at intensity ~0.4-2.5)
            roof_line(ax, bw, linewidth=1.5)
            ax.text(6, 6 * bw / 1e9 * 1.35, f"{level} {bw/1e9:,.0f} GB/s" + (" (measured)" if level == "L2" else " (spec)"),
                    color=INK2, fontsize=8, ha="center")
            if level == "L2":
                roof_line(ax, DRAM_BW, linewidth=1, linestyle=(0, (4, 3)))
                ax.text(25, 25 * DRAM_BW / 1e9 * 0.55, f"DRAM {DRAM_BW/1e9:.0f} GB/s", color=INK2, fontsize=8, ha="center")
            ax.text(250, FP32_PEAK / 1e9 * 1.3, f"FP32 {FP32_PEAK/1e12:.1f} TFLOP/s", color=INK2, fontsize=8, ha="center")
            for v, color, mk in (("V0", BLUE, "o"), ("V1", ORANGE, "s")):
                sel = [p for p in pts if p["variant"] == v and p["H"] == H and p["cache"] == cache
                       and p["level"] == level and p["AI_flop_per_byte"] == p["AI_flop_per_byte"]]
                ax.scatter([p["AI_flop_per_byte"] for p in sel], [p["GFLOPs_per_s"] for p in sel], s=52,
                           color=color, marker=mk, edgecolors=SURFACE, linewidths=1.2, zorder=4,
                           label=f"{v} ({'unfused' if v == 'V0' else 'fused'})")
                # cold points for T >= 256 cluster near the roof; label only the T = 8 outliers there
                for p in sel:
                    if level == "DRAM" and p["T"] > 8: continue
                    left = v == "V0"
                    ax.annotate(f"T={p['T']:,}", (p["AI_flop_per_byte"], p["GFLOPs_per_s"]),
                                xytext=(-7 if left else 7, 0 if left else {256: -6, 1024: 6}.get(p["T"], 0)),
                                textcoords="offset points", fontsize=7.5,
                                color=INK2, ha="right" if left else "left", va="center")
            ax.set_title(("Cold L2: DRAM roofline" if level == "DRAM" else "Warm, data in L2: L2 roofline") + f", H = {H}",
                         loc="left")
            ax.set_xlim(0.1, 1000); ax.set_ylim(5, 1e5)
            ax.set_xlabel(f"Arithmetic intensity (FLOP per {level} byte)")
            if j == 0: ax.set_ylabel("Achieved GFLOP/s")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Roofline: both variants are memory-bound; V1 falls well below the L2 roof when data is in L2",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02,
             f"Nsight Compute, NVIDIA L4, BF16, residual add. Unlabeled cold points are T = 256–16,384. Work = {FLOPS_PER_ELEM} FLOPs per element "
             "(same for both variants), so intensity differences come only from bytes moved.\n"
             "Cold: intensity per DRAM byte vs. the 300 GB/s spec. Warm: only runs whose data stayed in L2 (DRAM traffic < 5% of L2 traffic), "
             f"intensity per L2 byte vs. the highest L2 throughput measured in this run ({l2_roof/1e9:,.0f} GB/s).\n"
             "Warm runs whose working set exceeds L2 are DRAM-bound and listed in tables/roofline_table.md. "
             "V0 = sum of its two kernels.", color=INK2, fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_p, "roofline.png"); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", p, "and tables/roofline_table.{csv,md}")
    print(f"L2 roof (measured) = {l2_roof/1e9:.0f} GB/s")
    for p in pts: print(p)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "l4_20261121"))
