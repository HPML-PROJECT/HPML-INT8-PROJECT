#!/usr/bin/env python3
"""WS4 figures from a finished results folder. Needs only matplotlib (no GPU, no vLLM).

  python -m vllm_eval.reporting.make_plots <results_dir> [--ws3-csv <bench_results.csv>]

Reads   <results_dir>/{profile_w8a8,amdahl,bench_w8a8,bench_bf16,ppl_w8a8,ppl_bf16}.json
        the WS3 kernel benchmark CSV (for the measured V3-vs-V0 kernel speedups)
Writes  <results_dir>/../plots/time_breakdown_amdahl.png   where W8A8 GPU time goes + expected end-to-end gain
        <results_dir>/../plots/w8a8_vs_bf16.png            W8A8 speedup over BF16 per workload
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Validated categorical slots 1-4 (light), same order as the WS1-WS3 plots; gray for the remainder.
BLUE, ORANGE, GREEN, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b5b3ad"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK,
})

PHASES = [("prefill_8x512", "Prefill 8×512"), ("decode_b1", "Decode b=1"),
          ("decode_b8", "Decode b=8"), ("decode_b64", "Decode b=64")]
# Token count per norm+quant call in each phase, matched to the nearest WS3 benchmark shape (H = 2048).
PHASE_TOKENS = {"prefill_8x512": 4096, "decode_b1": 8, "decode_b8": 8, "decode_b64": 64}
DEFAULT_WS3 = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "triton-v3", "results", "l4_part3", "bench_results.csv")


def _load(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def kernel_speedups(ws3_csv, H=2048):
    """{T: (cold, warm)} speedup of the fastest V3 config over V0, from the WS3 run."""
    t = {}
    for r in csv.DictReader(open(ws3_csv)):
        if int(r["hidden_size"]) == H:
            t[(r["variant"], r["cache_mode"], int(r["num_tokens"]))] = float(r["median_us"])
    out = {}
    for T in set(PHASE_TOKENS.values()):
        s = []
        for cm in ("cold", "warm"):
            best = min(v for (var, m, tt), v in t.items() if var.startswith("V3") and m == cm and tt == T)
            s.append(t[("V0", cm, T)] / best)
        out[T] = (min(s), max(s))
    return out


def amdahl(f, s):
    return 1.0 / ((1.0 - f) + f / s)


def plot_breakdown(d, out, ws3_csv):
    prof = _load(d, "profile_w8a8.json")["phases"]
    amd = _load(d, "amdahl.json")
    ks = kernel_speedups(ws3_csv)
    labels = [lab for _, lab in PHASES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1.35, 1]))
    ys = list(range(len(PHASES)))[::-1]

    # Left: 100% stacked share of GPU kernel time.
    segs = [("INT8 GEMM", BLUE), ("Fusable RMSNorm → quant", ORANGE), ("Other norm + quant", GREEN),
            ("SiLU·mul (activation)", YELLOW), ("Attention + other", GRAY)]
    for y, (key, _) in zip(ys, PHASES):
        sh = prof[key]["shares"]
        vals = [sh["gemm"], sh["fusable_region"], sh["norm_plus_quant"] - sh["fusable_region"], sh["activation"]]
        vals.append(max(0.0, 1.0 - sum(vals)))
        left = 0.0
        for (name, col), v in zip(segs, vals):
            ax1.barh(y, v * 100, left=left * 100, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
            left += v
        ax1.text(sh["gemm"] * 50, y, f"GEMM {sh['gemm'] * 100:.0f}%", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
        ax1.annotate(f"fusable {sh['fusable_region'] * 100:.1f}%", (sh["gemm"] * 100, y),
                     xytext=(0, 17), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)
    ax1.set_xlim(0, 100)
    ax1.set_yticks(ys, labels)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("Share of GPU kernel time, W8A8 (%)")
    ax1.set_title("Where the time goes", loc="left")
    ax1.legend(handles=[Patch(color=c, label=n) for n, c in segs], loc="upper left",
               bbox_to_anchor=(0, -0.2), ncol=3, fontsize=8.5, handlelength=1.2)

    # Right: expected end-to-end gain = Amdahl on the fusable share.
    for y, (key, _) in zip(ys, PHASES):
        f = amd[key]["fractions"]["f_fusable_gpu"]
        lo_s, hi_s = ks[PHASE_TOKENS[key]]
        lo, hi, cap = [(amdahl(f, s) - 1) * 100 for s in (lo_s, hi_s, float("inf"))]
        ax2.plot([lo, hi], [y, y], color=YELLOW, linewidth=7, solid_capstyle="round", zorder=3)
        ax2.plot([hi, cap], [y, y], color=GRID, linewidth=1.5, linestyle=(0, (2, 2)), zorder=2)
        ax2.scatter([cap], [y], s=60, facecolors=SURFACE, edgecolors=INK2, linewidths=1.6, zorder=4)
        ax2.annotate(f"{lo:.1f}–{hi:.1f}%", ((lo + hi) / 2, y), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8.5, color=INK)
        ax2.annotate(f"≤{cap:.1f}%", (cap, y), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK2)
    ax2.set_xlim(0, 10)
    ax2.set_xlabel("Expected end-to-end speedup from fusing norm + quant (%)")
    ax2.set_title("What fusion can buy (Amdahl)", loc="left")
    ax2.grid(axis="y", visible=False)
    ax2.legend(handles=[Line2D([], [], color=YELLOW, linewidth=7, label="with measured V3 kernel speedup (cold–warm)"),
                        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK2,
                               markersize=7, label="ceiling: fused region takes zero time")],
               loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8.5)

    fig.suptitle("INT8 GEMMs dominate; fusing RMSNorm + quant can save at most ~8% (prefill) and ~3% (decode)",
                 x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "WS4, NVIDIA L4 at 1200 MHz, vLLM 0.34.0 (torch.compile + CUDA graphs), Qwen3-1.7B W8A8. "
             "PyTorch Profiler, GPU kernels only.\nKernel speedups: fastest V3 config vs V0 from WS3 at H = 2048, "
             "T = 4096 (prefill) and 8–64 (decode).\nEstimates, not measurements: in the compiled model the norm is "
             "an Inductor kernel, not V0's CUDA kernel, and V3 is not yet integrated.",
             fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    p = os.path.join(out, "time_breakdown_amdahl.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_precision(d, out):
    w, b = _load(d, "bench_w8a8.json"), _load(d, "bench_bf16.json")
    pw, pb = _load(d, "ppl_w8a8.json")["ppl"], _load(d, "ppl_bf16.json")["ppl"]
    rows = []
    for k in ("8x512", "1x2048"):
        bw, bb = w["prefill"][k]["throughput_tok_s"]["median"], b["prefill"][k]["throughput_tok_s"]["median"]
        rows.append((f"Prefill {k.replace('x', '×')}", bw / bb, f"{bb / 1e3:.1f}K → {bw / 1e3:.1f}K tok/s"))
    for k in ("b1", "b8", "b64"):
        tw, tb = w["decode"][k]["tpot_ms"]["median"], b["decode"][k]["tpot_ms"]["median"]
        rows.append((f"Decode b={k[1:]}", tb / tw, f"{tb:.1f} → {tw:.1f} ms/token"))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = range(len(rows))
    ax.bar(xs, [r[1] for r in rows], width=0.56, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for x, (_, s, _) in zip(xs, rows):
        ax.text(x, s + 0.015, f"{s:.2f}×", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    ax.set_xticks(list(xs), [f"{n}\n{raw}" for n, _, raw in rows], fontsize=8.5)
    ax.set_ylim(0, 1.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("W8A8 speedup over BF16 (×; BF16 = 1)")
    fig.suptitle("W8A8 INT8 is 1.20–1.42× faster than BF16 end to end", x=0.01, y=0.98, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, f"WS4, NVIDIA L4 at 1200 MHz, vLLM 0.34.0, Qwen3-1.7B. Prefill: throughput; decode: time per "
             f"output token\n(128-token prompts, 128 output tokens). Median of 5 repeats after 3 warm-ups.\n"
             f"WikiText-2 perplexity: BF16 {pb:.2f}, W8A8 {pw:.2f} (2048-token windows, 146 windows).",
             fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    p = os.path.join(out, "w8a8_vs_bf16.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir")
    ap.add_argument("--ws3-csv", default=DEFAULT_WS3, help="WS3 bench_results.csv (V3 kernel speedups)")
    ap.add_argument("--out", default=None, help="output folder (default: <results_dir>/../plots)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results_dir, "..", "plots")
    os.makedirs(out, exist_ok=True)
    for p in (plot_breakdown(a.results_dir, out, a.ws3_csv), plot_precision(a.results_dir, out)):
        print(os.path.normpath(p))


if __name__ == "__main__":
    main()
