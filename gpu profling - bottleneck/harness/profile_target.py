#!/usr/bin/env python3
"""
Independent harness for V0 (unfused) and V1 (vLLM fused CUDA kernel) RMSNorm + dynamic per-token INT8 quant.
ALIGNED WITH PART 1 (rmsnorm_int8_bench / bench_rmsnorm_int8.py):
  - identical inputs: CPU generator, per-shape seed (shape_seed), default seed 1234,
    x ~ N(0,1), w = 1 + 0.1*N(0,1), residual ~ N(0,1), generated in fp32 then cast
  - same eps (1e-6), BF16 default, residual add on by default (Part 1's main setting)
  - same reference (fp32 math with vLLM's activation-dtype rounding points) and same BF16 tolerances
  - same vLLM kernels (vllm._custom_ops wrappers call the torch.ops._C kernels Part 1 times)

  V0: fused_add_rms_norm (in-place: x<-norm(x+r), r<-x+r)  ->  scaled_int8_quant (dynamic per-token)
  V1: rms_norm_dynamic_per_token_quant(x, w, eps, int8, residual=r)   (single fused CUDA kernel)

Modes:
  profile : (default) fresh inputs before every call, call wrapped in NVTX range "target"
            so ncu/nsys can filter to exactly the kernels of interest.
  verify  : V0 and V1 vs reference and vs each other with Part 1's tolerances. Exit code 1 on failure.
            Also checks that the inputs are bit-identical to Part 1's if Part 1's script can be imported.
  bench   : quick CUDA-event timing sanity check (NOT authoritative; Part 1 has the headline timings).
"""
import argparse, os, statistics, sys, zlib
import torch

try:
    from vllm import _custom_ops as ops
except Exception as e:  # pragma: no cover
    sys.exit(f"Could not import vllm._custom_ops ({e}). Activate the pinned vLLM env.")

EPS = 1e-6
NVTX_RANGE = "target"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
PART1_DIR = os.environ.get("PART1_DIR", os.path.expanduser("~/rmsnorm_int8_bench"))

# Part 1's BF16 tolerances (bench_rmsnorm_int8.py, as used for results_l4_20261121_213823)
TOL = dict(q_max_abs=3, q_mismatch_frac=0.15, scale_rtol=1.6e-2, scale_atol=1e-6,
           dequant_lsb=2.5, res_atol=1e-2, res_rtol=1e-3)


# ---- inputs: same generator as Part 1 -------------------------------------------------------
def shape_seed(base, T, H, dtype, tag):
    return (base * 1_000_003 + zlib.crc32(f"{T}x{H}:{dtype}:{tag}".encode())) % (2**63 - 1)


def make_inputs(T, H, dtype_name, seed, residual=True):
    """Returns cuda tensors x0 [T,H], r0 [T,H] (None if no residual), w [H] in the activation dtype."""
    dt = DTYPES[dtype_name]
    g = torch.Generator()  # CPU, so inputs are bit-identical on every machine
    g.manual_seed(shape_seed(seed, T, H, dtype_name, "x"))
    x = torch.randn(T, H, generator=g, dtype=torch.float32)
    g.manual_seed(shape_seed(seed, 0, H, dtype_name, "w"))
    w = 1.0 + 0.1 * torch.randn(H, generator=g, dtype=torch.float32)
    r = None
    if residual:
        g.manual_seed(shape_seed(seed, T, H, dtype_name, "r"))
        r = torch.randn(T, H, generator=g, dtype=torch.float32)
    cv = lambda t: None if t is None else t.to(dt).cuda().contiguous()
    return cv(x), cv(r), cv(w)


def check_part1_inputs(T, H, dtype_name, seed, residual):
    """Import Part 1's make_inputs and confirm our tensors are bit-identical. Returns True/False/None."""
    if not os.path.exists(os.path.join(PART1_DIR, "bench_rmsnorm_int8.py")):
        print(f"[inputs] Part 1 script not found in {PART1_DIR}; skipped identity check")
        return None
    sys.path.insert(0, PART1_DIR)
    try:
        import bench_rmsnorm_int8 as p1
        theirs = p1.make_inputs(T, H, dtype_name, residual, seed)
    except Exception as e:
        print(f"[inputs] could not call Part 1 make_inputs ({e}); skipped identity check")
        return None
    flat = []
    def collect(o):
        if torch.is_tensor(o): flat.append(o)
        elif isinstance(o, (tuple, list)): [collect(i) for i in o]
        elif isinstance(o, dict): [collect(i) for i in o.values()]
        elif hasattr(o, "__dict__"): [collect(i) for i in vars(o).values()]
    collect(theirs)
    ours = [t for t in make_inputs(T, H, dtype_name, seed, residual) if t is not None]
    ok = all(any(o.shape == t.shape and torch.equal(o.cpu(), t.to(o.dtype).cpu()) for t in flat) for o in ours)
    print(f"[inputs] identical to Part 1 for T={T} H={H} {dtype_name} residual={residual}: {'YES' if ok else 'NO'}")
    return ok


# ---- variants ----------------------------------------------------------------------------
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


# ---- reference: same rounding points as Part 1 / vLLM --------------------------------------
def reference(x0, r0, w, residual):
    dt = x0.dtype
    h = (x0.float() + r0.float()).to(dt) if residual else x0       # residual add rounded to activation dtype
    hf = h.float()
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    y = ((hf * rstd).to(dt) * w).float()                            # rounded after norm and after * weight
    scale = y.abs().amax(-1) / 127.0
    return y, scale, h


def run_profile(a, dtype_name):
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype_name, a.seed, a.residual)
    x = x0.clone(); r = r0.clone() if r0 is not None else None
    fn = CALLS[a.variant]

    def fresh():  # restore inputs OUTSIDE the NVTX range (in-place kernels need fresh data)
        x.copy_(x0)
        if r is not None: r.copy_(r0)

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
    print(f"[profile] variant={a.variant} T={a.tokens} H={a.hidden} {dtype_name} residual={a.residual} "
          f"seed={a.seed} iters={a.iters} gpu={torch.cuda.get_device_name()}")


def run_verify(a, dtype_name):
    same = check_part1_inputs(a.tokens, a.hidden, dtype_name, a.seed, a.residual)
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype_name, a.seed, a.residual)
    res = {}
    for name, fn in CALLS.items():
        x = x0.clone(); r = r0.clone() if r0 is not None else None
        q, s = fn(x, r, w, a.residual)
        torch.cuda.synchronize()
        res[name] = (q.clone(), s.flatten().float().clone(), r.clone() if r is not None else None)
    y, ref_scale, h = reference(x0, r0, w, a.residual)
    ok_all = same is not False
    print(f"[verify] T={a.tokens} H={a.hidden} {dtype_name} residual={a.residual} tolerances={TOL}")
    for name, (q, s, r) in res.items():
        safe = s.clamp_min(1e-12)[:, None]
        lsb = ((q.float() * s[:, None] - y).abs() / safe).max().item()
        scale_ok = torch.allclose(s, ref_scale, rtol=TOL["scale_rtol"], atol=TOL["scale_atol"])
        ok = lsb <= TOL["dequant_lsb"] and scale_ok
        msg = f"  {name} vs ref: dequant max err = {lsb:.3f} INT8 steps, scales {'ok' if scale_ok else 'BAD'}"
        if r is not None:
            rdiff = (r.float() - h.float()).abs().max().item()
            ok = ok and torch.allclose(r.float(), h.float(), atol=TOL["res_atol"], rtol=TOL["res_rtol"])
            msg += f", residual max|diff| = {rdiff:.3e}"
        print(msg + ("  PASS" if ok else "  FAIL")); ok_all &= ok
    q0, s0, _ = res["v0"]; q1, s1, _ = res["v1"]
    d = (q0.int() - q1.int()).abs()
    mism, dmax = (d > 0).float().mean().item(), int(d.max())
    sc_rel = ((s1 - s0).abs() / s0.clamp_min(1e-12)).max().item()
    ok = dmax <= TOL["q_max_abs"] and mism <= TOL["q_mismatch_frac"] and \
        torch.allclose(s1, s0, rtol=TOL["scale_rtol"], atol=TOL["scale_atol"])
    print(f"  v1 vs v0: INT8 mismatch {mism*100:.2f}%, max step diff {dmax}, scale max rel diff {sc_rel:.2e}"
          + ("  PASS" if ok else "  FAIL"))
    ok_all &= ok
    print("[verify] RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


def run_bench(a, dtype_name):
    x0, r0, w = make_inputs(a.tokens, a.hidden, dtype_name, a.seed, a.residual)
    x = x0.clone(); r = r0.clone() if r0 is not None else None
    for name, fn in CALLS.items():
        ts = []
        for i in range(a.warmup + 50):
            x.copy_(x0)
            if r is not None: r.copy_(r0)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(x, r, w, a.residual); e.record(); e.synchronize()
            if i >= a.warmup: ts.append(s.elapsed_time(e) * 1e3)
        print(f"{name}: median {statistics.median(ts):.1f} us (event timing, no graph; sanity only - use Part 1 numbers)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=CALLS, default="v1")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--no-residual", dest="residual", action="store_false",
                   help="plain RMSNorm+quant without fused residual add")
    p.add_argument("--mode", choices=["profile", "verify", "bench"], default="profile")
    p.add_argument("--iters", type=int, default=3, help="profiled calls (ncu replays each)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234, help="same default as Part 1")
    a = p.parse_args()
    {"profile": run_profile, "verify": run_verify, "bench": run_bench}[a.mode](a, a.dtype)


if __name__ == "__main__":
    main()
