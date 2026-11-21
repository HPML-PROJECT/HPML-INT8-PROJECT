#!/usr/bin/env python3
"""
Prefill throughput and decode latency with stock vLLM on the L4 (W8A8 or BF16).

Prefill workloads : 8 x 512-token prompts (one batch), 1 x 2048-token prompt. max_tokens=1.
                    throughput = prompt tokens / wall time of generate()  (includes the 1 sampled token).
Decode workloads  : batch 1, 8, 64; 128 output tokens (prompt length --prompt-len, default 128).
                    Each repeat runs max_tokens=1 and max_tokens=128 on the same prompts:
                    tpot = (t_128 - t_1) / 127  -> decode latency per output token (prefill removed).
Usage: python -m vllm_eval.bench.bench_perf --tag w8a8 [--repeats 5 --warmup 3]
"""
import argparse, sys
from vllm_eval import common as C


def main():
    ap = C.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=128, help="prompt length for decode workloads")
    ap.add_argument("--out-len", type=int, default=C.DECODE_OUT_LEN)
    ap.add_argument("--decode-batches", type=int, nargs="+", default=C.DECODE_BATCHES)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    a = ap.parse_args()

    gpu = C.require_l4(a.allow_non_l4)
    model = a.model or C.MODELS[a.tag]
    llm = C.make_llm(model, a.max_model_len, a.gpu_mem, a.enforce_eager, a.custom_ops)
    res = {"tag": a.tag, "model": model, "gpu": gpu, "quantization": C.detected_quantization(llm),
           "config": vars(a), "env": C.env_info(), "smi_before": C.smi_snapshot(), "prefill": {}, "decode": {}}
    print(f"[info] detected quantization = {res['quantization']}")

    # ---------------- prefill ----------------
    if not a.skip_prefill:
        sp1 = C.sampling(1)
        for name, (n, L) in C.PREFILL_WORKLOADS.items():
            prompts = C.make_prompts(n, L)
            for _ in range(a.warmup):
                C.timed_generate(llm, prompts, sp1)
            ts = [C.timed_generate(llm, prompts, sp1)[0] for _ in range(a.repeats)]
            tokens = n * L
            thr = [tokens / t for t in ts]
            res["prefill"][name] = {"prompts": n, "prompt_len": L, "tokens": tokens, "times_s": ts,
                                    "time_s": C.summarize(ts), "throughput_tok_s": C.summarize(thr)}
            print(f"[prefill {name}] {tokens} tok  median {C.summarize(ts)['median']*1e3:.1f} ms  "
                  f"-> {C.summarize(thr)['median']:.0f} tok/s")

    # ---------------- decode ----------------
    if not a.skip_decode:
        sp1, spN = C.sampling(1), C.sampling(a.out_len)
        for B in a.decode_batches:
            prompts = C.make_prompts(B, a.prompt_len)
            for _ in range(a.warmup):
                C.timed_generate(llm, prompts, sp1); C.timed_generate(llm, prompts, spN)
            t1s, tNs, tpots = [], [], []
            for _ in range(a.repeats):
                t1, _ = C.timed_generate(llm, prompts, sp1)
                tN, outs = C.timed_generate(llm, prompts, spN)
                got = {len(o.outputs[0].token_ids) for o in outs}
                if got != {a.out_len}:
                    sys.exit(f"Expected {a.out_len} generated tokens per request, got {got}")
                t1s.append(t1); tNs.append(tN); tpots.append((tN - t1) / (a.out_len - 1))
            tp = C.summarize([x * 1e3 for x in tpots])
            res["decode"][f"b{B}"] = {
                "batch": B, "prompt_len": a.prompt_len, "out_len": a.out_len,
                "prefill_s": C.summarize(t1s), "e2e_s": C.summarize(tNs), "tpot_ms": tp,
                "decode_tok_s": B / (tp["median"] / 1e3),                 # decode-only throughput
                "e2e_tok_s": B * a.out_len / C.summarize(tNs)["median"],  # end-to-end throughput
                "raw": {"t1_s": t1s, "tN_s": tNs, "tpot_ms": [x * 1e3 for x in tpots]}}
            print(f"[decode b={B}] TPOT {tp['median']:.2f} ms (+-{tp['stdev']:.2f})  "
                  f"e2e({a.out_len} tok) {C.summarize(tNs)['median']:.3f} s  decode {B/(tp['median']/1e3):.0f} tok/s")

    res["smi_after"] = C.smi_snapshot()
    C.save_json(res, f"bench_{a.tag}.json")


if __name__ == "__main__":
    main()
