#!/usr/bin/env python3
"""
Independent harness for V0 (unfused) and V1 (vLLM fused CUDA kernel) RMSNorm + dynamic per-token INT8 quant.
No dependency on WS1 scripts. Same shapes/GPU as WS1: (tokens T, hidden H), BF16 activations.

  V0: fused_add_rms_norm (in-place: x<-norm(x+r), r<-x+r)  ->  scaled_int8_quant (dynamic per-token)
  V1: rms_norm_dynamic_per_token_quant(x, w, eps, int8, residual=r)   (single fused CUDA kernel)

Modes:
  profile : (default) fresh inputs before every call, call wrapped in NVTX range "target"
            so ncu/nsys can filter to exactly the kernels of interest.
  verify  : numerical comparison V0 vs V1 vs FP32 reference.
  bench   : quick CUDA-event timing sanity check (NOT the authoritative number; use ncu gpu__time_duration).
Examples:
  python harness/profile_target.py --variant v1 --tokens 4096 --hidden 2048
  python harness/profile_target.py --mode verify --tokens 4096 --hidden 2048
"""
import argparse, statistics, sys
import torch

try:
    from vllm import _custom_ops as ops
except Exception as e:  # pragma: no cover
    sys.exit(f"Could not import vllm._custom_ops ({e}). Activate the pinned vLLM env.")

EPS = 1e-6
NVTX_RANGE = "target"


def make_inputs(T, H, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x0 = torch.randn(T, H, device="cuda", dtype=dtype, generator=g)
    r0 = torch.randn(T, H, device="cuda", dtype=dtype, generator=g)
    w = (torch.rand(H, device="cuda", dtype=dtype, generator=g) + 0.5).contiguous()
    return x0, r0, w


def call_v0(x, r, w, residual):
    """Unfused pair. Mutates x (and r when residual=True)."""
    if residual:
        ops.fused_add_rms_norm(x, r, w, EPS)
        q, s, _ = ops.scaled_int8_quant(x)
    else:
        out = torch.empty_like(x)
        ops.rms_norm(out, x, w, EPS)
        q, s, _ = ops.scaled_int8_quant(out)
    return q, s


def call_v1(x, r, w, residual):
    """vLLM fused CUDA kernel. Mutates r when residual=True."""
    q, s = ops.rms_norm_dynamic_per_token_quant(
        x, w, EPS, torch.int8, residual=r if residual else None)
    return q, s


CALLS = {"v0": call_v0, "v1": call_v1}


def run_profile(a, dtype):
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype, a.seed)
    x, r = x0.clone(), r0.clone()
    fn = CALLS[a.variant]

    def fresh():  # restore inputs OUTSIDE the NVTX range (in-place kernels need fresh data)
        x.copy_(x0); r.copy_(r0)

    for _ in range(a.warmup):
        fresh(); fn(x, r, w, a.residual)
    torch.cuda.synchronize()
    for _ in range(a.iters):
        fresh()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(NVTX_RANGE)
        fn(x, r, w, a.residual)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    print(f"[profile] variant={a.variant} T={a.tokens} H={a.hidden} residual={a.residual} "
          f"iters={a.iters} gpu={torch.cuda.get_device_name()}")


def run_verify(a, dtype):
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype, a.seed)
    res = {}
    for name, fn in CALLS.items():
        x, r = x0.clone(), r0.clone()
        q, s = fn(x, r, w, a.residual)
        res[name] = (q.clone(), s.flatten().clone(), r.clone())
    # FP32 reference
    xf, rf, wf = x0.float(), r0.float(), w.float()
    h = xf + rf if a.residual else xf
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * wf
    ref_scale = ref.abs().amax(-1) / 127.0
    print(f"FP32 ref: T={a.tokens} H={a.hidden} residual={a.residual}")
    for name, (q, s, r) in res.items():
        deq = q.float() * s[:, None]
        print(f"{name}: scale rel.err(max)={((s-ref_scale).abs()/ref_scale).max():.3e}  "
              f"dequant abs.err(max)={(deq-ref).abs().max():.3e}  "
              f"dequant rel.err(mean)={((deq-ref).abs().mean()/ref.abs().mean()):.3e}"
              + (f"  residual max|diff|={(r.float()-h).abs().max():.3e}" if a.residual else ""))
    q0, q1 = res["v0"][0], res["v1"][0]
    d = (q0.int() - q1.int()).abs()
    print(f"v0 vs v1 INT8: mismatch rate={(d>0).float().mean()*100:.3f}%  max step diff={int(d.max())}  "
          f"scale max rel.diff={((res['v0'][1]-res['v1'][1]).abs()/res['v0'][1]).max():.3e}")


def run_bench(a, dtype):
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype, a.seed)
    x, r = x0.clone(), r0.clone()
    for name, fn in CALLS.items():
        ts = []
        for i in range(a.warmup + 50):
            x.copy_(x0); r.copy_(r0)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(x, r, w, a.residual); e.record(); e.synchronize()
            if i >= a.warmup: ts.append(s.elapsed_time(e) * 1e3)
        print(f"{name}: median {statistics.median(ts):.1f} us (event timing; includes launch overhead)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=CALLS, default="v1")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--no-residual", dest="residual", action="store_false",
                   help="plain RMSNorm+quant without fused residual add")
    p.add_argument("--mode", choices=["profile", "verify", "bench"], default="profile")
    p.add_argument("--iters", type=int, default=3, help="profiled calls (ncu replays each)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    {"profile": run_profile, "verify": run_verify, "bench": run_bench}[a.mode](a, dtype)


if __name__ == "__main__":
    main()
