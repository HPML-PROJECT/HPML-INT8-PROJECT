#!/usr/bin/env python3
"""Plots for triton-v3 (WS3) from bench_results.csv. Needs only matplotlib.

  python plot_results.py results/l4_part3/bench_results.csv [out_dir]

Writes, per (dtype, residual, cache mode):
  latency_<dtype>_res<r>_<cache>.png   median latency vs tokens: V0, V1, V2 and V3
  speedup_<dtype>_res<r>_<cache>.png   speedup over V0: V1, V2 and V3
and once:
  v3_configs.png                       geo-mean speedup of each V3 config over V2, warm and cold

V3 is plotted as the best of its configs at each shape (as in part3_summary.md); the shaded band spans all
V3 configs, so the spread between configs stays visible. bench_rmsnorm_int8.py imports make_plots from here.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

# Validated categorical slots 1-4 (light), fixed order, same as the WS1/WS2 plots; markers as secondary encoding.
STYLE = {
    "V0": dict(color="#2a78d6", marker="o", label="V0 vLLM unfused"),
    "V1": dict(color="#eb6834", marker="s", label="V1 vLLM fused CUDA"),
    "V2": dict(color="#1baf7a", marker="^", label="V2 Triton baseline"),
    "V3": dict(color="#eda100", marker="D", label="V3 ours (best config)"),
}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK,
})


def _load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["T"], r["H"], r["us"] = int(r["num_tokens"]), int(r["hidden_size"]), float(r["median_us"])
    return rows


def _series(grp, H):
    """{variant: {T: median us}} with all V3_* configs folded into 'V3' (best) plus 'V3_worst'."""
    out = defaultdict(dict)
    for r in grp:
        if r["H"] != H:
            continue
        v = r["variant"]
        if v.startswith("V3"):
            out["V3"][r["T"]] = min(out["V3"].get(r["T"], math.inf), r["us"])
            out["V3_worst"][r["T"]] = max(out["V3_worst"].get(r["T"], 0.0), r["us"])
        else:
            out[v][r["T"]] = r["us"]
    return out


def _xaxis(ax, Ts):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(Ts))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v) // 1024}K" if v >= 1024 else f"{int(v)}"))
    ax.set_xlabel("Tokens (T)")


def _line(ax, xs, ys, v):
    st = STYLE[v]
    ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=6, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=st["label"], zorder=3)


def _footer(fig, cm, clock):
    timing = ("Warm: median over CUDA-graph replays (100 samples)" if cm == "warm"
              else "Cold: median of 100 single calls, each after a 384 MB L2-evicting write")
    fig.text(0.01, 0.01, f"WS3 run, NVIDIA L4{clock}, BF16, residual add. {timing}. "
             "Band: fastest to slowest V3 config at each shape.\n"
             "Triton timings are bimodal at H=2048, T=4K and H=4096, T=2K, so V3's gap to V2 there is within noise.",
             fontsize=8, color=INK2, ha="left", va="bottom")


def make_plots(csv_path, out_dir, clock=" at 1200 MHz"):
    os.makedirs(out_dir, exist_ok=True)
    rows = _load(csv_path)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dtype"], r["residual"], r["cache_mode"])].append(r)
    written = []

    for (dt, res, cm), grp in sorted(groups.items()):
        Hs = sorted({r["H"] for r in grp})
        Ts = sorted({r["T"] for r in grp})
        tag = f"{dt}_res{res}_{cm}"
        for kind in ("latency", "speedup"):
            fig, axes = plt.subplots(1, len(Hs), figsize=(5.6 * len(Hs), 4.4), squeeze=False)
            for ax, H in zip(axes[0], Hs):
                s = _series(grp, H)
                v0 = s.get("V0", {})
                if kind == "latency":
                    if "V3" in s:
                        xs = sorted(s["V3"])
                        ax.fill_between(xs, [s["V3"][t] for t in xs], [s["V3_worst"][t] for t in xs],
                                        color=STYLE["V3"]["color"], alpha=0.18, linewidth=0, zorder=2)
                    for v in ("V0", "V1", "V2", "V3"):
                        if s.get(v):
                            xs = sorted(s[v])
                            _line(ax, xs, [s[v][t] for t in xs], v)
                    ax.set_yscale("log")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    ax.set_ylabel("Median time per call (µs)")
                else:
                    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                    ax.text(Ts[0], 1.0, " V0 = 1×", color=INK2, fontsize=8, va="bottom")
                    if "V3" in s:
                        xs = [t for t in sorted(s["V3"]) if t in v0]
                        ax.fill_between(xs, [v0[t] / s["V3_worst"][t] for t in xs],
                                        [v0[t] / s["V3"][t] for t in xs],
                                        color=STYLE["V3"]["color"], alpha=0.18, linewidth=0, zorder=2)
                    for v in ("V1", "V2", "V3"):
                        if s.get(v):
                            xs = [t for t in sorted(s[v]) if t in v0]
                            _line(ax, xs, [v0[t] / s[v][t] for t in xs], v)
                    ax.set_ylabel("Speedup over V0 (×, higher is faster)")
                _xaxis(ax, Ts)
                ax.set_title(f"H = {H}", loc="left")
            h, l = axes[0][0].get_legend_handles_labels()
            if any(k.startswith("V3_") for k in {r["variant"] for r in grp}):
                h.append(Patch(color=STYLE["V3"]["color"], alpha=0.18, linewidth=0))
                l.append("V3 range over its configs")
            fig.legend(h, l, loc="upper left", ncol=len(l), bbox_to_anchor=(0.01, 0.925), fontsize=9,
                       handlelength=2.2, columnspacing=1.6)
            head = {"latency": f"Time per call vs. tokens ({cm} L2)",
                    "speedup": f"Speedup over the unfused vLLM pair V0 ({cm} L2)"}[kind]
            fig.suptitle(head, x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold")
            _footer(fig, cm, clock)
            fig.tight_layout(rect=(0, 0.06, 1, 0.87))
            p = os.path.join(out_dir, f"{kind}_{tag}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    p = _config_plot(rows, out_dir, clock)
    if p:
        written.append(p)
    print("Plots:\n  " + "\n  ".join(written))
    return written


def _config_plot(rows, out_dir, clock):
    """Dot plot: geo-mean speedup of each V3 config over V2 (all shapes), warm and cold."""
    t = {(r["variant"], r["cache_mode"], r["H"], r["T"]): r["us"] for r in rows}
    cfgs = sorted({r["variant"] for r in rows if r["variant"].startswith("V3_")})
    if not cfgs or not any(k[0] == "V2" for k in t):
        return None
    caches = [c for c in ("warm", "cold") if any(k[1] == c for k in t)]
    gm = {}
    for c in cfgs:
        for cm in caches:
            ratios = [t[("V2", cm, H, T)] / us for (v, m, H, T), us in t.items()
                      if v == c and m == cm and ("V2", cm, H, T) in t]
            gm[(c, cm)] = math.exp(sum(map(math.log, ratios)) / len(ratios))
    cfgs.sort(key=lambda c: gm[(c, "warm")] if "warm" in caches else gm[(c, caches[0])])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = range(len(cfgs))
    ax.axvline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    for i, c in enumerate(cfgs):
        vals = [gm[(c, cm)] for cm in caches]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=2, zorder=2)
    cache_style = {"warm": dict(color=INK2, marker="o"), "cold": dict(facecolors="none", edgecolors=INK2, marker="s")}
    for cm in caches:   # neutral ink, filled vs hollow: V0-V3 hues keep their meaning across all figures
        kw = dict(edgecolors=SURFACE, linewidths=1.2) if cm == "warm" else dict(linewidths=1.6)
        ax.scatter([gm[(c, cm)] for c in cfgs], list(ys), s=64, zorder=3, label=f"{cm} L2",
                   **kw, **cache_style[cm])
    for i, c in enumerate(cfgs):   # direct value label for the warm point (the regime where configs differ)
        if "warm" in caches:
            ax.annotate(f"{gm[(c, 'warm')]:.2f}×", (gm[(c, "warm")], i), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
    labels = {"V3_r1w8": "r1w8 (V2's design)", "V3_r1w4p": "r1w4p (persistent)"}
    ax.set_yticks(list(ys), [labels.get(c, c[3:]) for c in cfgs])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(cfgs) - 0.4)
    ax.set_xlabel("Geo-mean speedup over V2, all shapes (×; 1 = same as V2)")
    ax.text(1.0, len(cfgs) - 0.45, " V2", color=INK2, fontsize=8, va="top")
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("No V3 config beats V2; fewer warps per row makes Triton slower", x=0.01, y=0.98,
                 ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, f"WS3 run, NVIDIA L4{clock}, BF16, residual add; H = 2048 and 4096, T = 8–16,384. "
             "Config r<rows per program>w<warps per row>.", fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = os.path.join(out_dir, "v3_configs.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/bench_results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path), "plots")
    make_plots(csv_path, out)
