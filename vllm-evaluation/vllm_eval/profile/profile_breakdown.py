#!/usr/bin/env python3
"""
Profile real vLLM inference with torch.profiler (GPU kernels only) and locate RMSNorm + quantization kernels.

Phases: prefill (8x512 prompts) and decode (batch 1/8/64, --decode-steps decode steps).
Decode-only time = profile(prefill + N decode steps) - profile(prefill only) on identical prompts.
Output: results/profile_<tag>.json  (+ results/profile_<tag>_<phase>_kernels.csv with the top kernels)
Tip: run once normally (CUDA graphs on = how vLLM really runs) and once with --enforce-eager for clearer attribution.
Usage: python -m vllm_eval.profile.profile_breakdown --tag w8a8 [--decode-steps 32]
"""
import argparse, csv, os, time
from vllm_eval import analysis as A
from vllm_eval import common as C


def profile_once(llm, prompts, sp, trace_path, keep):
    import torch
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    prof.export_chrome_trace(str(trace_path))
    m = A.analyze_events(A.load_gpu_events(trace_path))
    if not keep:
        os.remove(trace_path)
    return m, wall


def write_kernels_csv(m, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kernel", "category", "calls", "total_us", "share_of_gpu_time"])
        for r in m["top"]:
            w.writerow([r["name"], r["cat"], r["n"], round(r["us"], 1), round(r["us"] / m["total_us"], 5) if m["total_us"] else ""])


def main():
    ap = C.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--decode-batches", type=int, nargs="+", default=C.DECODE_BATCHES)
    ap.add_argument("--decode-steps", type=int, default=32, help="decode steps per profiled run (keeps traces small)")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--keep-trace", action="store_true")
    a = ap.parse_args()

    gpu = C.require_l4(a.allow_non_l4)
    model = a.model or C.MODELS[a.tag]
    llm = C.make_llm(model, a.max_model_len, a.gpu_mem, a.enforce_eager, a.custom_ops)
    mode = ("eager" if a.enforce_eager else "graphs") + (f"+{a.custom_ops}" if a.custom_ops else "")
    suffix = "" if mode == "graphs" else "_" + mode.replace("+", "").replace(",", "_")
    out = {"tag": a.tag, "model": model, "gpu": gpu, "mode": mode, "quantization": C.detected_quantization(llm),
           "env": C.env_info(), "phases": {}}
    tdir = C.RESULTS / "traces"; tdir.mkdir(parents=True, exist_ok=True)
    sp1 = C.sampling(1)

    def record(phase, m, wall, extra=None):
        s = A.shares(m)
        out["phases"][phase] = {"metrics": m, "shares": s, "wall_s": wall,
                                "gpu_busy_frac_of_wall": (m["total_us"] / 1e6) / wall if wall else None, **(extra or {})}
        write_kernels_csv(m, C.RESULTS / f"profile_{a.tag}{suffix}_{phase}_kernels.csv")
        print(f"[{phase}] GPU kernel time {m['total_us']/1e3:.2f} ms | norm {s['norm']*100:.1f}% quant {s['quant']*100:.1f}% "
              f"fused {s['norm_quant_fused']*100:.1f}% -> norm+quant {s['norm_plus_quant']*100:.1f}% "
              f"(fusable region {s['fusable_region']*100:.1f}%) | gemm {s['gemm']*100:.1f}% attn {s['attention']*100:.1f}%")

    # prefill: 8 x 512
    prompts = C.make_prompts(*C.PREFILL_WORKLOADS["8x512"])
    for _ in range(a.warmup): llm.generate(prompts, sp1, use_tqdm=False)
    m, wall = profile_once(llm, prompts, sp1, tdir / f"{a.tag}_prefill.json", a.keep_trace)
    record("prefill_8x512", m, wall)

    # decode: B in batches
    spN = C.sampling(a.decode_steps + 1)
    for B in a.decode_batches:
        prompts = C.make_prompts(B, a.prompt_len)
        for _ in range(a.warmup):
            llm.generate(prompts, sp1, use_tqdm=False); llm.generate(prompts, spN, use_tqdm=False)
        mA, wA = profile_once(llm, prompts, sp1, tdir / f"{a.tag}_b{B}_A.json", a.keep_trace)
        mB, wB = profile_once(llm, prompts, spN, tdir / f"{a.tag}_b{B}_B.json", a.keep_trace)
        dec = A.subtract(mB, mA)
        record(f"decode_b{B}", dec, wB - wA, {"decode_steps": a.decode_steps})

    C.save_json(out, f"profile_{a.tag}{suffix}.json")


if __name__ == "__main__":
    main()
