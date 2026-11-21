#!/usr/bin/env python3
"""
Plots for a WS2 results folder (tables + Nsight Systems summaries).
  python analysis/make_plots.py results/l4_20261121
Reads   <dir>/tables/memory_table.csv, execution_table.csv, part1_vs_ncu.md, <dir>/nsys/*.kern_sum.csv
Writes  <dir>/plots/*.png
Needs only matplotlib (no GPU, no Nsight tools).
"""
import csv, io, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Colors (validated categorical slots 1-3, light mode) and ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
L2_BYTES = 48e6          # NVIDIA L4 L2 size
BYTES_PER_TOKEN = 5      # x read, residual read + write, int8 out per element (BF16): ~5 bytes x H

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


def load(d):
    M = list(csv.DictReader(open(os.path.join(d, "tables", "memory_table.csv"))))
    E = list(csv.DictReader(open(os.path.join(d, "tables", "execution_table.csv"))))
    tot = {(r["variant"], int(r["H"]), int(r["T"]), r["cache"]): r for r in M if r["kernel"] == "TOTAL"}
    kern = {}
    for r in E:
        if r["kernel"] != "TOTAL":
            kern.setdefault((r["variant"], int(r["H"]), int(r["T"]), r["cache"]), []).append(r)
    # Part 1 ratios from part1_vs_ncu.md: | GPU | H | T | cache | P1 V0 | P1 V1 | P1 ratio | ...
    p1 = {}
    path = os.path.join(d, "tables", "part1_vs_ncu.md")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 7 and c[1].isdigit():
                p1[(int(c[1]), int(c[2]), c[3])] = f(c[6])
    # Nsight Systems: sum of per-kernel medians
    ns = {}
    nd = os.path.join(d, "nsys")
    for fn in os.listdir(nd) if os.path.isdir(nd) else []:
        m = re.match(r"(v\d)_.+_H(\d+)_T(\d+)\.kern_sum\.csv$", fn)
        if not m: continue
        txt = open(os.path.join(nd, fn)).read(); i = txt.find("Time (%)")
        if i < 0: continue
        rows = list(csv.DictReader(io.StringIO(txt[i:])))
        if rows: ns[(m.group(1), int(m.group(2)), int(m.group(3)))] = sum(f(r["Med (ns)"]) for r in rows) / 1e3
    return tot, kern, p1, ns


def token_axis(ax, Ts):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(Ts))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Tokens (T)")


def ref_line(ax):
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)


def save(fig, out, name):
    p = os.path.join(out, name); fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); print("wrote", p)


def plot_ratio(tot, out, Hs, Ts):
    fig, axes = plt.subplots(1, len(Hs), figsize=(5.2 * len(Hs), 3.9), sharey=True)
    for ax, H in zip(axes, Hs):
        ref_line(ax)
        fit = L2_BYTES / (BYTES_PER_TOKEN * H)
        ax.axvline(fit, color=INK2, linewidth=1, linestyle=":", zorder=1)
        ax.text(fit * 1.06, 2.55, "working set\n> 48 MB L2 →", color=INK2, fontsize=8, va="top")
        for cache, color, mk in (("warm", BLUE, "o"), ("cold", ORANGE, "s")):
            xs = [T for T in Ts if ("v0", H, T, cache) in tot and ("v1", H, T, cache) in tot]
            ys = [f(tot[("v1", H, T, cache)]["time_us"]) / f(tot[("v0", H, T, cache)]["time_us"]) for T in xs]
            ax.plot(xs, ys, color=color, linewidth=2, marker=mk, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{cache} L2", zorder=3)
            i256 = xs.index(256) if 256 in xs else 1
            ax.annotate(f"{cache} L2", (xs[i256], ys[i256]), xytext=(-8, 9 if cache == "warm" else -13),
                        textcoords="offset points", ha="right", color=INK2, fontsize=9)
            top = max(range(len(ys)), key=lambda i: ys[i])
            if ys[top] > 1.1:
                ax.annotate(f"{ys[top]:.2f}×", (xs[top], ys[top]), xytext=(0, 8), textcoords="offset points",
                            ha="center", color=INK, fontsize=9)
        ax.text(Ts[0], 1.03, "V1 slower ↑", color=INK2, fontsize=8, va="bottom")
        ax.text(Ts[0], 0.97, "V1 faster ↓", color=INK2, fontsize=8, va="top")
        token_axis(ax, Ts); ax.set_title(f"H = {H}")
        ax.set_ylim(0.4, 2.6)
    axes[0].set_ylabel("V1 / V0 kernel time")
    axes[0].legend(loc="upper left")
    fig.suptitle("Fused kernel (V1) vs. unfused pair (V0): slower only while data is in L2",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, -0.04, "Nsight Compute kernel time, NVIDIA L4, BF16, residual add. Cold = L2 flushed before each replay; "
             "warm = not flushed. Clocks power-capped (not fixed).", color=INK2, fontsize=8)
    save(fig, out, "time_ratio_vs_tokens.png")


def plot_bytes(tot, out, shapes):
    labels = [f"H={H}\nT={T}" for H, T in shapes]
    dram = [f(tot[("v1", H, T, "cold")]["dram_MB"]) / f(tot[("v0", H, T, "cold")]["dram_MB"]) for H, T in shapes]
    l2 = [f(tot[("v1", H, T, "cold")]["L2_MB"]) / f(tot[("v0", H, T, "cold")]["L2_MB"]) for H, T in shapes]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = range(len(shapes)); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], dram, w, color=BLUE, label="DRAM bytes", zorder=3)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], l2, w, color=ORANGE, label="L2 bytes", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ref_line(ax)
    ax.text(len(shapes) - 0.5, 1.01, "V0 = 1.0", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.12); ax.set_ylabel("V1 bytes / V0 bytes")
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("V1 always moves fewer bytes than V0 (about 35% less)", loc="left")
    fig.text(0.01, -0.06, "Nsight Compute, cold L2, NVIDIA L4, BF16, residual add. V0 = sum of its two kernels.",
             color=INK2, fontsize=8)
    save(fig, out, "bytes_v1_vs_v0.png")


def plot_exec(kern, out, H, T, cache):
    v0, v1 = kern[("v0", H, T, cache)], kern[("v1", H, T, cache)][0]
    norm = next(k for k in v0 if "rms" in k["kernel"].lower() or "enable_if" in k["kernel"])
    panels = [
        ("Instructions (millions)", sum(f(k["inst"]) for k in v0) / 1e6, f(v1["inst"]) / 1e6, "{:.1f}"),
        ("Load instructions (thousands)", sum(f(k["gld_inst"]) for k in v0) / 1e3, f(v1["gld_inst"]) / 1e3, "{:.0f}"),
        ("Bytes per warp load", f(norm["approx_B/ld_inst"]), f(v1["approx_B/ld_inst"]), "{:.0f}"),
        ("Achieved occupancy (%)", f(norm["occ_ach_%"]), f(v1["occ_ach_%"]), "{:.0f}%"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3), gridspec_kw={"wspace": 0.35})
    for ax, (title, a, b, fmt) in zip(axes, panels):
        bars = ax.bar([0, 1], [a, b], 0.6, color=[BLUE, ORANGE], zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02, fmt.format(bar.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["V0", "V1"]); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5, loc="left"); ax.set_ylim(0, max(a, b) * 1.22)
        if "occupancy" in title:
            ax.axhline(66.7, color=INK2, linewidth=1, linestyle=":", zorder=4)
            ax.text(2.05, 68.5, "V1 cap 66.7%", color=INK2, fontsize=8, va="bottom", ha="right")
            ax.set_xlim(-0.5, 2.1)
    fig.suptitle(f"Where V1's extra time goes (H = {H}, T = {T}): more, narrower loads and lower occupancy",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.06)
    fig.text(0.01, -0.12, f"Nsight Compute, {cache} L2, NVIDIA L4, BF16, residual add. V1 uses 1,024-thread blocks: "
             "one block per SM (1,536-thread limit), so occupancy is capped at 2/3.\n"
             "V0 bytes per load and occupancy are for its norm kernel. Ratios are the same at T = 16,384.",
             color=INK2, fontsize=8)
    save(fig, out, "execution_v0_vs_v1.png")


def plot_agreement(tot, p1, ns, out, shapes):
    labels = [f"H={H}\nT={T}" for H, T in shapes]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    series = {
        "warm": [("Nsight Compute", BLUE, "o",
                  [f(tot[("v1", H, T, "warm")]["time_us"]) / f(tot[("v0", H, T, "warm")]["time_us"]) for H, T in shapes]),
                 ("Nsight Systems", ORANGE, "s",
                  [ns[("v1", H, T)] / ns[("v0", H, T)] if ("v0", H, T) in ns and ("v1", H, T) in ns else float("nan")
                   for H, T in shapes]),
                 ("Part 1 benchmark", AQUA, "D", [p1.get((H, T, "warm"), float("nan")) for H, T in shapes])],
        "cold": [("Nsight Compute", BLUE, "o",
                  [f(tot[("v1", H, T, "cold")]["time_us"]) / f(tot[("v0", H, T, "cold")]["time_us"]) for H, T in shapes]),
                 ("Part 1 benchmark", AQUA, "D", [p1.get((H, T, "cold"), float("nan")) for H, T in shapes])],
    }
    offs = {"Nsight Compute": -0.15, "Nsight Systems": 0.0, "Part 1 benchmark": 0.15}
    for ax, cache in zip(axes, ("warm", "cold")):
        ref_line(ax)
        for name, color, mk, ys in series[cache]:
            ax.scatter([i + offs[name] for i in range(len(shapes))], ys, s=48, color=color, marker=mk,
                       edgecolors=SURFACE, linewidths=1.2, label=name, zorder=3)
        ax.set_ylabel("V1 / V0 time"); ax.set_ylim(0.4, 2.6)
        ax.set_title(f"{cache.capitalize()} L2", loc="left"); ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", ncols=3, fontsize=8.5)
    axes[1].set_xticks(range(len(shapes))); axes[1].set_xticklabels(labels, fontsize=8)
    fig.suptitle("Three measurement methods agree on which shapes are slow",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.03, "NVIDIA L4, BF16, residual add. Nsight Systems runs are warm only. "
             "Part 1 = CUDA-graph benchmark.", color=INK2, fontsize=8)
    fig.tight_layout()
    save(fig, out, "ratio_agreement.png")


def main(d):
    tot, kern, p1, ns = load(d)
    out = os.path.join(d, "plots"); os.makedirs(out, exist_ok=True)
    Hs = sorted({k[1] for k in tot}); Ts = sorted({k[2] for k in tot})
    shapes = [(H, T) for H in Hs for T in Ts]
    plot_ratio(tot, out, Hs, Ts)
    plot_bytes(tot, out, [(H, T) for H in Hs for T in Ts if T >= 256])
    plot_exec(kern, out, Hs[0], 1024 if 1024 in Ts else Ts[len(Ts) // 2], "cold")
    plot_agreement(tot, p1, ns, out, shapes)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "l4_20261121"))
