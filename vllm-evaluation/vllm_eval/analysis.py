"""
Pure-Python (no GPU / no vLLM needed) analysis helpers:
  * classify(): GPU kernel name -> category (norm, quant, norm_quant_fused, gemm, attention, ...)
  * analyze_events(): per-category GPU time, and the *fusable* norm+quant region found by kernel adjacency
  * subtract(): decode-only metrics = (prefill+decode run) - (prefill-only run)
  * amdahl(): end-to-end speedup bound
Kernel names differ across vLLM versions / compile modes, so the report always lists the top kernels and their
categories; adjust the regexes below if something is misclassified.
"""
import json, math, re
from collections import defaultdict

CATS = ["quant", "norm", "norm_quant_fused", "gemm", "attention", "activation", "rope", "memory", "other"]

_NORM = re.compile(r"rms_?norm|rmsnorm|layer_?norm|rsqrt")            # vLLM CUDA kernels and Inductor (rsqrt) kernels
_QUANT = re.compile(r"int8_quant|scaled_int8|per_token_quant|static_scaled|dynamic_scaled|quantize|quant_kernel")
_GEMM = re.compile(r"cutlass|gemm|cublas|xmma|gemv|nvjet|sm80_|sm89_|ampere_|ada_")
_ATTN = re.compile(r"flash|fmha|attn|attention|paged|reshape_and_cache|cache_kernel|concat_and_cache")
_ACT = re.compile(r"silu|act_and_mul|gelu|swiglu")
_ROPE = re.compile(r"rotary|rope")


def classify(name: str, cat: str = "kernel") -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memory"
    n = name.lower()
    has_norm = bool(_NORM.search(n))
    has_quant = bool(_QUANT.search(n)) or ("round" in n and ("amax" in n or "clamp" in n))
    if has_norm and has_quant: return "norm_quant_fused"
    if has_quant: return "quant"
    if has_norm: return "norm"
    if _GEMM.search(n): return "gemm"
    if _ATTN.search(n): return "attention"
    if _ACT.search(n): return "activation"
    if _ROPE.search(n): return "rope"
    return "other"


def load_gpu_events(trace_path):
    """Kernel + memcpy/memset events from a torch.profiler chrome trace."""
    with open(trace_path) as f:
        data = json.load(f)
    evs = data["traceEvents"] if isinstance(data, dict) else data
    out = []
    for e in evs:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
            args = e.get("args", {}) or {}
            out.append({"name": e.get("name", ""), "cat": e["cat"], "ts": float(e["ts"]), "dur": float(e["dur"]),
                        "stream": args.get("stream", e.get("tid")), "dev": args.get("device", e.get("pid"))})
    return out


def analyze_events(events, top_n=25):
    by_us, by_n = defaultdict(float), defaultdict(int)
    names = defaultdict(lambda: [0, 0.0, ""])
    streams = defaultdict(list)
    for e in events:
        c = classify(e["name"], e["cat"])
        e["_c"] = c
        by_us[c] += e["dur"]; by_n[c] += 1
        rec = names[e["name"]]; rec[0] += 1; rec[1] += e["dur"]; rec[2] = c
        streams[(e["dev"], e["stream"])].append(e)
    # Fusable region: a quant kernel executed immediately after a norm kernel on the same stream
    # (qkv_proj <- input_layernorm, gate_up_proj <- post_attention_layernorm). o_proj / down_proj quant
    # follow attention / SiLU-and-mul and q_norm/k_norm are followed by RoPE, so they are NOT counted.
    f_norm_us = f_quant_us = 0.0; f_norm_n = f_quant_n = 0
    for evs in streams.values():
        evs.sort(key=lambda x: x["ts"])
        for prev, cur in zip(evs, evs[1:]):
            if cur["_c"] == "quant" and prev["_c"] == "norm":
                f_quant_us += cur["dur"]; f_quant_n += 1
                f_norm_us += prev["dur"]; f_norm_n += 1
    total = sum(e["dur"] for e in events)
    span = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) if events else 0.0
    top = sorted(([k, v[2], v[0], v[1]] for k, v in names.items()), key=lambda r: -r[3])[:top_n]
    return {"total_us": total, "span_us": span, "n_kernels": len(events),
            "by_cat_us": {c: by_us.get(c, 0.0) for c in CATS}, "by_cat_n": {c: by_n.get(c, 0) for c in CATS},
            "fusable": {"norm_us": f_norm_us, "norm_n": f_norm_n, "quant_us": f_quant_us, "quant_n": f_quant_n},
            "top": [{"name": n[:140], "cat": c, "n": k, "us": u} for n, c, k, u in top]}


def subtract(b, a):
    """Numeric dict difference b - a (used for decode-only = (prefill+decode) - prefill)."""
    if isinstance(b, dict):
        return {k: subtract(b[k], a[k]) if k in a else b[k] for k in b if k != "top"} | ({"top": b["top"]} if "top" in b else {})
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return b - a
    return b


def shares(m):
    """Fractions of total GPU kernel time for the quantities used in Amdahl."""
    tot = m["total_us"] or float("nan")
    c = m["by_cat_us"]
    nq = c["norm"] + c["quant"] + c["norm_quant_fused"]
    fus = m["fusable"]["norm_us"] + m["fusable"]["quant_us"]
    return {"norm": c["norm"] / tot, "quant": c["quant"] / tot, "norm_quant_fused": c["norm_quant_fused"] / tot,
            "norm_plus_quant": nq / tot, "fusable_region": fus / tot,
            **{k: v / tot for k, v in c.items() if k not in ("norm", "quant", "norm_quant_fused")}}


def amdahl(f, s):
    """End-to-end speedup when a fraction f of time is accelerated by factor s."""
    if s == math.inf: return 1.0 / (1.0 - f)
    return 1.0 / ((1.0 - f) + f / s)


# Bandwidth-model bound for fusing (BF16, per element): unfused = add+rmsnorm (read x, r; write x', r' = 8 B)
# + quant (read 2 B, write 1 B) = 11 B; fused = read x, r (4 B), write r' (2 B) + int8 (1 B) = 7 B.
S_BW_BOUND = 11.0 / 7.0
