#!/usr/bin/env python3
"""
Baseline WikiText-2 (raw, test) perplexity through the SAME vLLM engine/kernels used for benchmarking.

Method: test text joined with "\n\n" -> tokenized -> split into non-overlapping windows of --seq-len tokens
(remainder dropped, the common GPTQ/SmoothQuant-style protocol). vLLM returns the log-prob of every prompt token
(prompt_logprobs=0); NLL over tokens 1..L-1 of each window; PPL = exp(mean NLL).
Usage: python -m vllm_eval.bench.eval_ppl --tag w8a8 [--seq-len 2048 --batch 2 --max-windows 0]
"""
import argparse, math, sys, time
from vllm_eval import common as C


def main():
    ap = C.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2, help="windows per generate() call (prompt logprobs are memory hungry)")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all windows (quick tests: 8)")
    a = ap.parse_args()
    if a.gpu_mem == 0.85: a.gpu_mem = 0.80    # leave headroom for full-vocab logits of prompt tokens

    gpu = C.require_l4(a.allow_non_l4)
    model = a.model or C.MODELS[a.tag]
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), add_special_tokens=False)["input_ids"]
    n_win = len(ids) // a.seq_len
    if a.max_windows: n_win = min(n_win, a.max_windows)
    windows = [ids[i * a.seq_len:(i + 1) * a.seq_len] for i in range(n_win)]
    print(f"[data] {len(ids)} tokens -> {n_win} windows of {a.seq_len}")

    llm = C.make_llm(model, max(a.max_model_len, a.seq_len + 8), a.gpu_mem, a.enforce_eager, a.custom_ops)
    sp = C.sampling(1, prompt_logprobs=0)
    nll, cnt, t0 = 0.0, 0, time.perf_counter()
    for i in range(0, n_win, a.batch):
        batch = windows[i:i + a.batch]
        outs = llm.generate([{"prompt_token_ids": w} for w in batch], sp, use_tqdm=False)
        for w, o in zip(batch, outs):
            pl = o.prompt_logprobs
            if pl is None: sys.exit("vLLM returned no prompt_logprobs (is prefix caching disabled?)")
            for t in range(1, len(w)):
                lp = pl[t].get(w[t])
                if lp is None: sys.exit(f"token {w[t]} missing from prompt_logprobs at position {t}")
                nll -= lp.logprob; cnt += 1
        if (i // a.batch) % 10 == 0:
            print(f"  window {min(i + a.batch, n_win)}/{n_win}  running ppl {math.exp(nll / cnt):.4f}")
    ppl = math.exp(nll / cnt)
    print(f"[ppl] {a.tag}: {ppl:.4f}  ({cnt} tokens, {time.perf_counter() - t0:.0f}s)")
    C.save_json({"tag": a.tag, "model": model, "gpu": gpu, "quantization": C.detected_quantization(llm),
                 "dataset": "wikitext-2-raw-v1/test", "seq_len": a.seq_len, "n_windows": n_win,
                 "n_tokens_scored": cnt, "total_tokens": len(ids), "mean_nll": nll / cnt, "ppl": ppl,
                 "partial_run": bool(a.max_windows), "env": C.env_info()}, f"ppl_{a.tag}.json")


if __name__ == "__main__":
    main()
