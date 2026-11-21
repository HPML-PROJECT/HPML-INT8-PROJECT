#!/usr/bin/env python3
"""Aggregate everything in results/ into results/REPORT.md (performance, perplexity, breakdown, Amdahl)."""
import math
from vllm_eval import common as C


def f(x, d=2): return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def main():
    L = ["# End-to-end impact of RMSNorm + quantization: Qwen3-1.7B on NVIDIA L4\n"]
    env = C.load_json("env.json")
    if env:
        L += [f"- GPU: {env['gpu']} (cc {env['compute_capability']}), torch {env['torch']}, CUDA {env['cuda']}, vLLM {env.get('vllm')}",
              f"- nvidia-smi snapshot: `{env['smi']}`\n"]
    b = {t: C.load_json(f"bench_{t}.json") for t in ("w8a8", "bf16")}
    p = {t: C.load_json(f"ppl_{t}.json") for t in ("w8a8", "bf16")}

    L.append("## 1. Prefill throughput (tokens/s, median; higher is better)\n")
    L += ["| workload | BF16 | W8A8 | W8A8 / BF16 |", "|---|---|---|---|"]
    for w in ("8x512", "1x2048"):
        v = {t: b[t]["prefill"][w]["throughput_tok_s"]["median"] if b[t] and w in b[t]["prefill"] else None for t in b}
        r = v["w8a8"] / v["bf16"] if v["w8a8"] and v["bf16"] else None
        L.append(f"| {w} | {f(v['bf16'],0)} | {f(v['w8a8'],0)} | {f(r)}x |")

    L.append("\n## 2. Decode latency (128 output tokens)\n")
    L += ["| batch | BF16 TPOT ms | W8A8 TPOT ms | W8A8/BF16 speedup | BF16 e2e s | W8A8 e2e s |", "|---|---|---|---|---|---|"]
    for B in (1, 8, 64):
        k = f"b{B}"
        g = lambda t, key: b[t]["decode"][k][key]["median"] if b[t] and k in b[t]["decode"] else None
        tb, tw = g("bf16", "tpot_ms"), g("w8a8", "tpot_ms")
        L.append(f"| {B} | {f(tb)} | {f(tw)} | {f(tb/tw if tb and tw else None)}x | {f(g('bf16','e2e_s'),3)} | {f(g('w8a8','e2e_s'),3)} |")

    L.append("\n## 3. WikiText-2 perplexity (non-overlapping windows)\n")
    L += ["| model | PPL | tokens scored | windows | partial run |", "|---|---|---|---|---|"]
    for t in ("bf16", "w8a8"):
        if p[t]: L.append(f"| {t} | {f(p[t]['ppl'],4)} | {p[t]['n_tokens_scored']} | {p[t]['n_windows']} | {p[t]['partial_run']} |")
    if p["bf16"] and p["w8a8"]:
        L.append(f"\nW8A8 - BF16 = {p['w8a8']['ppl']-p['bf16']['ppl']:+.4f} PPL ({(p['w8a8']['ppl']/p['bf16']['ppl']-1)*100:+.2f}%)")

    L.append("\n## 4. Model execution breakdown (share of GPU kernel time)\n")
    for t in ("w8a8", "bf16"):
        pr = C.load_json(f"profile_{t}.json")
        if not pr: continue
        L.append(f"\n**{t}** (mode `{pr['mode']}`, quantization `{pr['quantization']}`)\n")
        L += ["| phase | norm | quant | fused n+q | norm+quant | fusable region | GEMM | attention | other | GPU busy / wall |", "|---|---|---|---|---|---|---|---|---|---|"]
        for ph, d in pr["phases"].items():
            s = d["shares"]
            L.append(f"| {ph} | {s['norm']*100:.1f}% | {s['quant']*100:.1f}% | {s['norm_quant_fused']*100:.1f}% | **{s['norm_plus_quant']*100:.1f}%** | "
                     f"{s['fusable_region']*100:.1f}% | {s['gemm']*100:.1f}% | {s['attention']*100:.1f}% | "
                     f"{(s['activation']+s['rope']+s['memory']+s['other'])*100:.1f}% | {f(d['gpu_busy_frac_of_wall'])} |")
    am = C.load_json("amdahl.json")
    if am:
        L.append("\n## 5. Amdahl bound for fusing norm + quant (W8A8)\n")
        L += ["| phase | f_all (GPU) | f_fusable (GPU) | max speedup, fusable (s=inf) | speedup, fusable at s=11/7 (bandwidth bound) |", "|---|---|---|---|---|"]
        for ph, d in am.items():
            L.append(f"| {ph} | {d['fractions']['f_all_gpu']*100:.2f}% | {d['fractions']['f_fusable_gpu']*100:.2f}% | "
                     f"{d['speedup']['f_fusable_gpu']['inf']:.4f}x | {d['speedup']['f_fusable_gpu'].get('1.571', float('nan')):.4f}x |")
        L.append("\nFull tables: `amdahl.md`. Kernel-level speedups are not end-to-end speedups.")
    L.append("\n## 6. Integration\nSee `INTEGRATION_PROPOSAL.md` and `integration_map.md` (hooks found in the installed vLLM).\n")
    (C.RESULTS / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
