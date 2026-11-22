#!/usr/bin/env python3
"""Plots from bench_results.csv: latency vs token count and speedup vs V0.

One figure pair per (dtype, residual, cache_mode); one panel per hidden size (small multiples).
Usage: python plot_results.py results/bench_results.csv results/plots
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Fixed categorical order (validated reference palette slots 1-3) + marker as secondary encoding
STYLE = {
    "V0": dict(color="#2a78d6", marker="o", label="V0 vLLM unfused"),
    "V1": dict(color="#eb6834", marker="s", label="V1 vLLM fused CUDA"),
    "V2": dict(color="#1baf7a", marker="^", label="V2 Triton fused"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_ax(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _load(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_tokens"] = int(r["num_tokens"])
        r["hidden_size"] = int(r["hidden_size"])
        for k in ("median_us", "p10_us", "p90_us"):
            r[k] = float(r[k])
        r["speedup_vs_v0"] = float(r["speedup_vs_v0"]) if r["speedup_vs_v0"] not in ("", None) else None
    return rows


def _grid(n):
    cols = min(n, 2)
    return (n + cols - 1) // cols, cols


def make_plots(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = _load(csv_path)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dtype"], r["residual"], r["cache_mode"])].append(r)
    gpu = rows[0]["gpu"] if rows else ""
    written = []

    for (dt, res, cm), grp in sorted(groups.items()):
        hiddens = sorted({r["hidden_size"] for r in grp})
        variants = [v for v in STYLE if any(r["variant"] == v for r in grp)]
        tag = f"{dt}_res{res}_{cm}"
        title_ctx = f"{gpu} | {dt} | {'residual-add + ' if res == '1' else ''}RMSNorm + INT8 | {cm} cache"

        for kind in ("latency", "speedup"):
            nr, nc = _grid(len(hiddens))
            fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 3.6 * nr), squeeze=False,
                                     sharex=True)
            fig.patch.set_facecolor("#fcfcfb")
            for i, H in enumerate(hiddens):
                ax = axes[i // nc][i % nc]
                _style_ax(ax)
                for v in variants:
                    pts = sorted((r for r in grp if r["variant"] == v and r["hidden_size"] == H),
                                 key=lambda r: r["num_tokens"])
                    if kind == "speedup":
                        if v == "V0":
                            continue  # baseline is the dashed 1.0 line
                        pts = [p for p in pts if p["speedup_vs_v0"] is not None]
                    if not pts:
                        continue
                    xs = [p["num_tokens"] for p in pts]
                    st = STYLE[v]
                    if kind == "latency":
                        ys = [p["median_us"] for p in pts]
                        ax.fill_between(xs, [p["p10_us"] for p in pts], [p["p90_us"] for p in pts],
                                        color=st["color"], alpha=0.15, linewidth=0)
                    else:
                        ys = [p["speedup_vs_v0"] for p in pts]
                    ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=5,
                            linewidth=2, label=st["label"])
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
                if kind == "latency":
                    ax.set_yscale("log")
                    ax.set_ylabel("median latency (µs)", color=MUTED, fontsize=9)
                else:
                    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
                    ax.set_ylabel("speedup vs V0 (×)", color=MUTED, fontsize=9)
                ax.set_title(f"hidden = {H}", color=INK, fontsize=10, loc="left")
                if i // nc == nr - 1:
                    ax.set_xlabel("num tokens", color=MUTED, fontsize=9)
            for j in range(len(hiddens), nr * nc):
                axes[j // nc][j % nc].set_visible(False)
            h, l = axes[0][0].get_legend_handles_labels()
            if kind == "speedup":
                h = [plt.Line2D([], [], color=MUTED, linestyle="--")] + h
                l = ["V0 baseline (1×)"] + l
            fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9,
                       bbox_to_anchor=(0.5, 0.945))
            name = "Latency vs token count" if kind == "latency" else "Speedup relative to V0 (higher is better)"
            fig.suptitle(f"{name}\n{title_ctx}", color=INK, fontsize=11, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.80 if nr == 1 else 0.90))
            p = os.path.join(out_dir, f"{kind}_{tag}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    print("Plots:\n  " + "\n  ".join(written))
    return written


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/bench_results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path), "plots")
    make_plots(csv_path, out)
