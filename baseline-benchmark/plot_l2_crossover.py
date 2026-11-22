#!/usr/bin/env python3
"""
One-figure summary of WS1: V1/V0 time against the working set, for every hidden size.
  python plot_l2_crossover.py results/l4_20261121
Reads   <dir>/bench_results.csv
Writes  <dir>/plots/l2_crossover.png
Needs only matplotlib.

Working set = distinct bytes one call touches (residual add mode, 2-byte activations):
  x (2 B) + residual, read and written in place (2 B) + INT8 output (1 B) = 5 bytes per element, i.e. 5 * T * H.
"""
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

L2_MB = 48.0                                   # NVIDIA L4
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4 (light)
MARKERS = ["o", "s", "D", "^"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK,
})


def main(d, dtype="bf16"):
    rows = list(csv.DictReader(open(os.path.join(d, "bench_results.csv"))))
    t = {(r["variant"], r["dtype"], r["residual"], r["cache_mode"], int(r["hidden_size"]), int(r["num_tokens"])):
         float(r["median_us"]) for r in rows}
    Hs = sorted({k[4] for k in t})
    Ts = sorted({k[5] for k in t})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, cache in zip(axes, ("warm", "cold")):
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(L2_MB, color=INK2, linewidth=1, linestyle=":", zorder=1)
        ax.text(L2_MB * 1.08, 2.85, "L2 = 48 MB", color=INK2, fontsize=8, va="top")
        for i, H in enumerate(Hs):
            pts = [(5 * T * H / 1e6, t[("V1", dtype, "1", cache, H, T)] / t[("V0", dtype, "1", cache, H, T)])
                   for T in Ts if ("V1", dtype, "1", cache, H, T) in t and ("V0", dtype, "1", cache, H, T) in t]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLORS[i], linewidth=2, marker=MARKERS[i], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"H = {H}", zorder=3)
            if cache == "warm":   # direct label at each line's peak
                k = max(range(len(ys)), key=lambda j: ys[j])
                dx, ha = {4096: (-6, "right"), 5120: (6, "left")}.get(H, (0, "center"))
                ax.annotate(f"H={H}", (xs[k], ys[k]), xytext=(dx, 7), textcoords="offset points",
                            ha=ha, fontsize=8, color=INK2)
        ax.set_xscale("log")
        ticks = [0.01, 0.1, 1, 10, 48, 100, 1000]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Working set of one call (MB) = 5 bytes × T × H")
        ax.set_title(f"{cache.capitalize()} L2", loc="left")
        ax.set_ylim(0.4, 2.9)
        if cache == "warm":
            ax.text(0.012, 1.04, "V1 slower ↑", color=INK2, fontsize=8, va="bottom")
            ax.text(0.012, 0.96, "V1 faster ↓", color=INK2, fontsize=8, va="top")
    axes[0].set_ylabel("V1 / V0 time (median)")
    axes[1].legend(loc="upper left")
    fig.suptitle("The fused kernel's slowdown ends where the working set outgrows the L2 cache, at every hidden size",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.06, f"WS1 main run, NVIDIA L4, {dtype.upper()}, residual add. Warm: CUDA-graph median over 50 replays; "
             "cold: median of 100 single calls after an L2 flush. Cold ratios at small working sets are limited by the "
             "~0.5 µs timer resolution.", color=INK2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(d, "plots"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "l2_crossover.png"); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "l4_20261121"))
