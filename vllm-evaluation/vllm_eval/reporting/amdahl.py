#!/usr/bin/env python3
"""
Amdahl's-law bound on end-to-end gain from fusing RMSNorm + INT8 quantization (W8A8 profile).
Two fractions per phase, on GPU-kernel-time and on wall-clock basis:
  f_all     = all norm + quant (+ already-fused) kernels             -> optimistic upper bound
  f_fusable = norm->quant pairs adjacent in the kernel stream        -> what a fused norm+quant kernel can replace
Speedup of the fused region s: a grid of values, plus the bandwidth-model bound s = 11/7 (BF16; see analysis.py).
Usage: python -m vllm_eval.reporting.amdahl [--profile w8a8] [--s 1.25 1.5 2 3]
"""
import argparse, math
from vllm_eval import analysis as A
from vllm_eval import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="w8a8", help="tag / file stem results/profile_<this>.json")
    ap.add_argument("--s", type=float, nargs="+", default=[1.25, 1.5, 2.0, 3.0])
    a = ap.parse_args()
    prof = C.load_json(f"profile_{a.profile}.json")
    if prof is None:
        raise SystemExit(f"results/profile_{a.profile}.json not found: run profile_breakdown.py first")
    s_list = a.s + [round(A.S_BW_BOUND, 3), math.inf]
    rows, md = {}, []
    md.append(f"# Amdahl bound ({a.profile}, mode={prof['mode']}, GPU={prof['gpu']})\n")
    md.append("f = fraction of time in the accelerated region; S = 1 / ((1-f) + f/s). "
              f"Bandwidth-model bound s = 11/7 = {A.S_BW_BOUND:.2f} (BF16 add+norm+quant vs fused).\n")
    for phase, p in prof["phases"].items():
        m, s_ = p["metrics"], p["shares"]
        wall_us = p["wall_s"] * 1e6
        fr = {"f_all_gpu": s_["norm_plus_quant"], "f_fusable_gpu": s_["fusable_region"],
              "f_all_wall": (m["by_cat_us"]["norm"] + m["by_cat_us"]["quant"] + m["by_cat_us"]["norm_quant_fused"]) / wall_us if wall_us > 0 else float("nan"),
              "f_fusable_wall": (m["fusable"]["norm_us"] + m["fusable"]["quant_us"]) / wall_us if wall_us > 0 else float("nan")}
        res = {name: {("inf" if s == math.inf else f"{s:g}"): A.amdahl(f, s) for s in s_list} for name, f in fr.items()}
        rows[phase] = {"fractions": fr, "speedup": res, "gpu_busy_frac_of_wall": p["gpu_busy_frac_of_wall"]}
        md.append(f"\n## {phase}\nGPU busy fraction of wall time: {p['gpu_busy_frac_of_wall']:.2f}\n")
        md.append("| basis | f | " + " | ".join("s=inf" if s == math.inf else f"s={s:g}" for s in s_list) + " |")
        md.append("|---|---|" + "---|" * len(s_list))
        for name, f in fr.items():
            md.append(f"| {name} | {f*100:.2f}% | " + " | ".join(f"{A.amdahl(f, s):.4f}x" for s in s_list) + " |")
    md.append("\nNotes: *_gpu uses GPU kernel time (ignores CPU gaps); *_wall divides by profiled wall time (profiler overhead inflates it, "
              "so wall-based f is a lower bound). A kernel-level speedup is never an end-to-end speedup; use f_fusable for a realistic bound.")
    C.save_json(rows, "amdahl.json")
    (C.RESULTS / "amdahl.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
