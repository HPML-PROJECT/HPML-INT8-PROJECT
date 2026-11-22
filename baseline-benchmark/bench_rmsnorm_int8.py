#!/usr/bin/env python3
"""
Baseline benchmark: RMSNorm + dynamic per-token INT8 quantization.

Variants
  V0  vLLM unfused : torch.ops._C.rms_norm (or fused_add_rms_norm)  ->  torch.ops._C.dynamic_scaled_int8_quant
  V1  vLLM fused   : torch.ops._C.rms_norm_dynamic_per_token_quant   (single CUDA kernel)
  V2  Triton fused : built-in Triton kernel below, or an external one via --v2-impl module:function

Pipeline per variant
  1. deterministic inputs (CPU torch.Generator, seeded per shape -> identical across machines)
  2. cross-check V1/V2 against V0 and all against an fp32 PyTorch reference (shared tolerances)
  3. timing with CUDA events, kernels replayed through CUDA graphs
       warm : graph of N back-to-back calls, replayed R times, latency = elapsed / N
       cold : L2 flushed (write a buffer >> L2) before every call, each call timed individually
  4. results CSV + correctness CSV + env.json, then plots (latency vs tokens, speedup vs V0)

Usage
  python bench_rmsnorm_int8.py                       # full default grid, warm + cold
  python bench_rmsnorm_int8.py --quick               # small grid smoke test
  python bench_rmsnorm_int8.py --plot-only --out-dir results   # re-plot an existing CSV
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import json
import math
import os
import platform
import statistics
import sys
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------------------
# Shared configuration (change here or via CLI so every team member uses the same values)
# --------------------------------------------------------------------------------------
DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_HIDDEN = [2048, 4096, 5120, 8192]
DEFAULT_DTYPES = ["fp16", "bf16"]
DEFAULT_EPS = 1e-6
DEFAULT_SEED = 1234

# Shared tolerances for the cross-check
TOL_Q_MAX_ABS = 3          # max |q_a - q_b| in int8 LSBs
TOL_Q_MISMATCH_FRAC = 0.15  # fraction of int8 elements allowed to differ (by <= TOL_Q_MAX_ABS)
TOL_SCALE_RTOL = 1.6e-2      # relative tolerance on per-token scales
TOL_SCALE_ATOL = 1e-6
TOL_DEQUANT_LSB = 2.5      # |q*scale - ref| <= TOL_DEQUANT_LSB * scale   (vs fp32 reference)
TOL_RES_ATOL = 1e-2        # residual output (residual mode only)
TOL_RES_RTOL = 1e-3

CSV_FIELDS = [
    "variant", "impl", "gpu", "dtype", "num_tokens", "hidden_size", "residual",
    "cache_mode", "timing_method", "inner_iters", "samples",
    "median_us", "mean_us", "std_us", "min_us", "p10_us", "p90_us",
    "eff_bandwidth_GBps", "speedup_vs_v0", "check_passed",
    "torch_version", "vllm_version", "triton_version", "timestamp",
]
CHECK_FIELDS = [
    "variant", "compared_to", "dtype", "num_tokens", "hidden_size", "residual",
    "q_max_abs_diff", "q_mismatch_frac", "scale_max_rel_diff", "dequant_max_err_lsb",
    "residual_max_abs_diff", "passed", "note",
]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# --------------------------------------------------------------------------------------
# V2: Triton fused RMSNorm (+ optional residual add) + dynamic per-token INT8 quant
# Numerics mirror vLLM: fp32 accumulation, normalized value rounded to the activation
# dtype before multiplying by the weight, scale = absmax / 127, round-to-nearest-even.
# --------------------------------------------------------------------------------------
_TRITON_OK = False
_TRITON_ERR = ""
try:
    import triton
    import triton.language as tl

    _rint = None
    for _mod in ("triton.language.extra.cuda.libdevice", "triton.language.extra.libdevice",
                 "triton.language.math"):
        try:
            _m = importlib.import_module(_mod)
            if hasattr(_m, "rint"):
                _rint = _m.rint
                break
        except Exception:
            pass
    if _rint is None:
        raise ImportError("no libdevice.rint found in this Triton version")

    @triton.jit
    def _rms_norm_dyn_int8_kernel(
        x_ptr, r_ptr, w_ptr, q_ptr, s_ptr,
        stride_x, stride_r, stride_q,
        H, eps,
        HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
        if HAS_RES:
            r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0)
            z = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
            tl.store(r_ptr + row * stride_r + cols, z, mask=mask)
            xf = z.to(tl.float32)
        else:
            xf = x.to(tl.float32)
        var = tl.sum(xf * xf, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = ((xf * rstd).to(w.dtype) * w).to(tl.float32)
        absmax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(absmax / 127.0, 1e-10)
        q = _rint(y / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(q_ptr + row * stride_q + cols, q.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, scale)

    def triton_rms_norm_dyn_int8(x, w, eps, q_out, s_out, residual=None):
        T, H = x.shape
        BLOCK = triton.next_power_of_2(H)
        nw = 16 if BLOCK >= 8192 else (8 if BLOCK >= 2048 else 4)
        _rms_norm_dyn_int8_kernel[(T,)](
            x, residual if residual is not None else x, w, q_out, s_out,
            x.stride(0), residual.stride(0) if residual is not None else 0, q_out.stride(0),
            H, eps, HAS_RES=residual is not None, BLOCK=BLOCK, num_warps=nw,
        )

    _TRITON_OK = True
except Exception as e:  # pragma: no cover
    _TRITON_ERR = f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------------------
# Variant wrappers. Every variant exposes the same interface:
#   run(x, w, eps, q_out, s_out, residual, scratch) -> None
# Outputs: q_out (int8 [T,H]), s_out (fp32 [T,1]); in residual mode `residual` is updated
# in place to x + residual (vLLM semantics). V0 may also clobber `x` (fused_add_rms_norm).
# --------------------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    impl: str
    run: Optional[Callable] = None
    available: bool = False
    reason: str = ""
    needs_scratch: bool = False


def _load_vllm():
    try:
        import vllm  # noqa: F401
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
        return vllm.__version__, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _probe(run, needs_scratch, residual):
    """Call a candidate wrapper on a tiny GPU problem; True if it runs and gives sane output."""
    T, H = 4, 256
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(T, H, generator=g).half().cuda()
    w = torch.ones(H).half().cuda()
    r = torch.randn(T, H, generator=g).half().cuda() if residual else None
    q = torch.zeros(T, H, dtype=torch.int8, device="cuda")
    s = torch.zeros(T, 1, dtype=torch.float32, device="cuda")
    scratch = torch.empty_like(x) if needs_scratch else None
    run(x, w, 1e-6, q, s, r, scratch)
    torch.cuda.synchronize()
    return bool((s > 0).all().item()) and int(q.abs().max().item()) >= 100


def _pick(v: Variant, candidates):
    """candidates: list of (label, run). Keep the first that passes the probe with and without residual."""
    errs = []
    for label, run in candidates:
        try:
            if _probe(run, v.needs_scratch, False) and _probe(run, v.needs_scratch, True):
                v.run, v.available = run, True
                v.impl += f" [{label}]"
                return v
            errs.append(f"{label}: probe gave bad output")
        except Exception as e:
            errs.append(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
    v.reason = " | ".join(errs) or "no candidate implementation found"
    return v


def make_v0() -> Variant:
    v = Variant("V0", "vllm rms_norm/fused_add_rms_norm + dynamic_scaled_int8_quant (unfused)",
                needs_scratch=True)
    ver, err = _load_vllm()
    if ver is None:
        v.reason = "vLLM import failed: " + err
        return v
    C = torch.ops._C
    from vllm import _custom_ops as ops
    cands = []
    if all(hasattr(C, n) for n in ("rms_norm", "fused_add_rms_norm", "dynamic_scaled_int8_quant")):
        def run_c(x, w, eps, q_out, s_out, residual, scratch):
            if residual is None:
                C.rms_norm(scratch, x, w, eps)
                C.dynamic_scaled_int8_quant(q_out, scratch, s_out, None)
            else:
                # in place: residual <- x + residual ; x <- rmsnorm(residual) * w
                C.fused_add_rms_norm(x, residual, w, eps)
                C.dynamic_scaled_int8_quant(q_out, x, s_out, None)
        cands.append(("torch.ops._C", run_c))

    def run_py(x, w, eps, q_out, s_out, residual, scratch):
        if residual is None:
            ops.rms_norm(scratch, x, w, eps)
            src = scratch
        else:
            ops.fused_add_rms_norm(x, residual, w, eps)
            src = x
        q, s, _ = ops.scaled_int8_quant(src, None, None, True)
        q_out.copy_(q)
        s_out.copy_(s.view_as(s_out))
    cands.append(("vllm._custom_ops python wrappers", run_py))
    return _pick(v, cands)


def make_v1() -> Variant:
    v = Variant("V1", "vllm rms_norm_dynamic_per_token_quant (fused CUDA)")
    ver, err = _load_vllm()
    if ver is None:
        v.reason = "vLLM import failed: " + err
        return v
    C = torch.ops._C
    from vllm import _custom_ops as ops
    cands = []
    if hasattr(C, "rms_norm_dynamic_per_token_quant"):
        def run_c(x, w, eps, q_out, s_out, residual, scratch):
            C.rms_norm_dynamic_per_token_quant(q_out, x, w, s_out, eps, None, residual)
        cands.append(("torch.ops._C", run_c))
    if hasattr(ops, "rms_norm_dynamic_per_token_quant"):
        # python wrapper allocates its outputs; copied into the preallocated buffers
        def run_py(x, w, eps, q_out, s_out, residual, scratch):
            q, s = ops.rms_norm_dynamic_per_token_quant(x, w, eps, torch.int8, None, residual)
            q_out.copy_(q)
            s_out.copy_(s.view_as(s_out))
        cands.append(("python wrapper fallback, includes output copy", run_py))
    return _pick(v, cands)


def make_v2(spec: Optional[str]) -> Variant:
    if spec:
        # external implementation: module:function with signature
        #   fn(x, w, eps, residual=None) -> (q_int8 [T,H], scale_fp32 [T] or [T,1])
        v = Variant("V2", f"external Triton: {spec}")
        try:
            mod_name, fn_name = spec.split(":")
            fn = getattr(importlib.import_module(mod_name), fn_name)
        except Exception as e:
            v.reason = f"cannot load {spec}: {e}"
            return v

        def run(x, w, eps, q_out, s_out, residual, scratch):
            q, s = fn(x, w, eps, residual=residual)
            q_out.copy_(q)
            s_out.copy_(s.view_as(s_out))
        v.run, v.available = run, True
        return v

    v = Variant("V2", "Triton fused rmsnorm + per-token int8 (built-in)")
    if not _TRITON_OK:
        v.reason = "Triton unavailable: " + _TRITON_ERR
        return v

    def run(x, w, eps, q_out, s_out, residual, scratch):
        triton_rms_norm_dyn_int8(x, w, eps, q_out, s_out, residual)
    return _pick(v, [("triton " + getattr(triton, "__version__", "?"), run)])


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def shape_seed(base: int, T: int, H: int, dtype: str, tag: str) -> int:
    return (base * 1_000_003 + zlib.crc32(f"{T}x{H}:{dtype}:{tag}".encode())) % (2**63 - 1)


def make_inputs(T: int, H: int, dtype_name: str, residual: bool, seed: int):
    """Deterministic: generated on CPU with a per-shape seed, then copied to the GPU."""
    dt = DTYPES[dtype_name]
    g = torch.Generator(device="cpu")
    g.manual_seed(shape_seed(seed, T, H, dtype_name, "x"))
    x = torch.randn(T, H, generator=g, dtype=torch.float32)
    g.manual_seed(shape_seed(seed, 0, H, dtype_name, "w"))
    w = 1.0 + 0.1 * torch.randn(H, generator=g, dtype=torch.float32)
    r = None
    if residual:
        g.manual_seed(shape_seed(seed, T, H, dtype_name, "r"))
        r = torch.randn(T, H, generator=g, dtype=torch.float32)
    to = lambda t: None if t is None else t.to(dt).cuda().contiguous()
    return to(x), to(w), to(r)


def reference(x, w, eps, residual):
    """fp32 reference with the same activation-dtype rounding points as vLLM."""
    dt = x.dtype
    if residual is not None:
        z = (x.float() + residual.float()).to(dt)
        res_out = z
        xf = z.float()
    else:
        res_out = None
        xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = ((xf * rstd).to(dt) * w).float()
    return y, res_out


# --------------------------------------------------------------------------------------
# Cross-check
# --------------------------------------------------------------------------------------
@dataclass
class Buffers:
    x0: torch.Tensor
    w: torch.Tensor
    r0: Optional[torch.Tensor]
    x: torch.Tensor = None
    r: Optional[torch.Tensor] = None
    q: torch.Tensor = None
    s: torch.Tensor = None
    scratch: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, x0, w, r0, needs_scratch):
        b = cls(x0=x0, w=w, r0=r0)
        b.x = x0.clone()
        b.r = None if r0 is None else r0.clone()
        T, H = x0.shape
        b.q = torch.empty(T, H, dtype=torch.int8, device=x0.device)
        b.s = torch.empty(T, 1, dtype=torch.float32, device=x0.device)
        b.scratch = torch.empty_like(x0) if needs_scratch else None
        return b

    def reset(self):
        self.x.copy_(self.x0)
        if self.r is not None:
            self.r.copy_(self.r0)

    def call(self, v: Variant, eps):
        v.run(self.x, self.w, eps, self.q, self.s, self.r, self.scratch)


def run_once(v: Variant, x0, w, r0, eps):
    b = Buffers.create(x0, w, r0, v.needs_scratch)
    b.call(v, eps)
    torch.cuda.synchronize()
    return b.q.clone(), b.s.clone(), (None if b.r is None else b.r.clone())


def compare(name, other, outs_a, outs_b, ref_y, ref_res, meta, tol) -> dict:
    qa, sa, ra = outs_a
    row = dict(variant=name, compared_to=other, **meta)
    note = []
    if outs_b is not None:
        qb, sb, rb = outs_b
        d = (qa.int() - qb.int()).abs()
        row["q_max_abs_diff"] = int(d.max().item())
        row["q_mismatch_frac"] = float((d > 0).float().mean().item())
        rel = ((sa - sb).abs() / sb.abs().clamp_min(1e-12))
        row["scale_max_rel_diff"] = float(rel.max().item())
        ok = (row["q_max_abs_diff"] <= tol.q_max_abs
              and row["q_mismatch_frac"] <= tol.q_mismatch_frac
              and torch.allclose(sa, sb, rtol=tol.scale_rtol, atol=tol.scale_atol))
        if ra is not None and rb is not None:
            row["residual_max_abs_diff"] = float((ra.float() - rb.float()).abs().max().item())
            ok = ok and torch.allclose(ra.float(), rb.float(), atol=tol.res_atol, rtol=tol.res_rtol)
        row["dequant_max_err_lsb"] = ""
    else:  # vs fp32 reference
        deq = qa.float() * sa
        err_lsb = ((deq - ref_y).abs() / sa).max().item()
        row["dequant_max_err_lsb"] = float(err_lsb)
        row["q_max_abs_diff"] = row["q_mismatch_frac"] = row["scale_max_rel_diff"] = ""
        ref_scale = ref_y.abs().amax(dim=-1, keepdim=True) / 127.0
        scale_ok = torch.allclose(sa, ref_scale, rtol=tol.scale_rtol, atol=tol.scale_atol)
        if not scale_ok:
            note.append("scale!=ref")
        ok = err_lsb <= tol.dequant_lsb and scale_ok
        if ra is not None:
            row["residual_max_abs_diff"] = float((ra.float() - ref_res.float()).abs().max().item())
            ok = ok and torch.allclose(ra.float(), ref_res.float(), atol=tol.res_atol, rtol=tol.res_rtol)
        if not torch.isfinite(deq).all():
            ok = False
            note.append("non-finite")
    row.setdefault("residual_max_abs_diff", "")
    row["passed"] = bool(ok)
    row["note"] = ";".join(note)
    return row


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def _stats(samples_us: List[float]) -> dict:
    s = sorted(samples_us)
    n = len(s)
    q = lambda p: s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return dict(
        median_us=statistics.median(s), mean_us=statistics.fmean(s),
        std_us=statistics.pstdev(s) if n > 1 else 0.0, min_us=s[0],
        p10_us=q(0.10), p90_us=q(0.90), samples=n,
    )


def _capture(fn: Callable, n: int) -> Callable:
    """Capture n back-to-back calls of fn into a CUDA graph; return graph.replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    return g.replay


def _estimate_us(fn: Callable, reset: Callable, n: int = 20) -> float:
    reset()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reset()
    st.record()
    for _ in range(n):
        fn()
    en.record()
    en.synchronize()
    return st.elapsed_time(en) * 1000.0 / n


def time_warm(fn, reset, needs_reset, use_graph, reps, target_ms, max_inner):
    est = _estimate_us(fn, reset)
    inner = int(max(1, min(max_inner, round(target_ms * 1000.0 / max(est, 0.5)))))
    if use_graph:
        runner = _capture(fn, inner)
    else:
        def runner():
            for _ in range(inner):
                fn()
    reset()
    runner()  # warm up the graph / caches
    torch.cuda.synchronize()
    out = []
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        if needs_reset:
            reset()        # not timed; also leaves inputs resident in L2 (warm)
        st.record()
        runner()
        en.record()
        en.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / inner)
    return out, inner


def time_cold(fn, reset, needs_reset, use_graph, iters, flush_buf):
    runner = _capture(fn, 1) if use_graph else fn
    reset()
    runner()
    torch.cuda.synchronize()
    evs = []
    for _ in range(iters):
        if needs_reset:
            reset()
        flush_buf.zero_()  # evict L2 (buffer is several x L2 size); not timed
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        runner()
        en.record()
        evs.append((st, en))
    torch.cuda.synchronize()
    return [a.elapsed_time(b) * 1000.0 for a, b in evs], 1


def ideal_bytes(T, H, elem, residual):
    b = T * H * elem + H * elem + T * H * 1 + T * 4  # read x, w ; write q, scale
    if residual:
        b += 2 * T * H * elem                          # read + write residual
    return b


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------
@dataclass
class Tol:
    q_max_abs: int = TOL_Q_MAX_ABS
    q_mismatch_frac: float = TOL_Q_MISMATCH_FRAC
    scale_rtol: float = TOL_SCALE_RTOL
    scale_atol: float = TOL_SCALE_ATOL
    dequant_lsb: float = TOL_DEQUANT_LSB
    res_atol: float = TOL_RES_ATOL
    res_rtol: float = TOL_RES_RTOL


def env_info():
    ver, _ = _load_vllm()
    p = torch.cuda.get_device_properties(0)
    info = dict(
        gpu=torch.cuda.get_device_name(0),
        sm=f"{p.major}.{p.minor}",
        sm_count=p.multi_processor_count,
        total_mem_GB=round(p.total_memory / 2**30, 2),
        l2_cache_MB=round(getattr(p, "L2_cache_size", 0) / 2**20, 2),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        vllm_version=ver or "n/a",
        triton_version=getattr(sys.modules.get("triton"), "__version__", "n/a"),
        python=platform.python_version(),
        host=platform.node(),
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
    )
    return info


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    ap.add_argument("--hidden", type=int, nargs="+", default=DEFAULT_HIDDEN)
    ap.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES, choices=list(DTYPES))
    ap.add_argument("--residual", choices=["off", "on", "both"], default="off",
                    help="benchmark plain RMSNorm (off), fused residual-add RMSNorm (on), or both")
    ap.add_argument("--variants", nargs="+", default=["V0", "V1", "V2"])
    ap.add_argument("--v2-impl", default=None,
                    help="external Triton impl 'module:function', fn(x,w,eps,residual=None)->(q,scale)")
    ap.add_argument("--cache-modes", nargs="+", default=["warm", "cold"], choices=["warm", "cold"])
    ap.add_argument("--reps", type=int, default=50, help="warm: graph replays per measurement")
    ap.add_argument("--target-ms", type=float, default=2.0, help="warm: target GPU time per replay")
    ap.add_argument("--max-inner", type=int, default=500, help="warm: max calls captured per graph")
    ap.add_argument("--cold-iters", type=int, default=100, help="cold: individually timed calls")
    ap.add_argument("--flush-mb", type=int, default=0, help="cold: L2 flush buffer MB (0 = 8 x L2)")
    ap.add_argument("--no-graph", action="store_true", help="time eager launches instead of CUDA graphs")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--bench-failed", action="store_true",
                    help="still benchmark a variant that failed the cross-check for a shape")
    ap.add_argument("--quick", action="store_true", help="small grid smoke test")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    # tolerances
    ap.add_argument("--tol-q-max-abs", type=int, default=TOL_Q_MAX_ABS)
    ap.add_argument("--tol-q-mismatch-frac", type=float, default=TOL_Q_MISMATCH_FRAC)
    ap.add_argument("--tol-scale-rtol", type=float, default=TOL_SCALE_RTOL)
    ap.add_argument("--tol-dequant-lsb", type=float, default=TOL_DEQUANT_LSB)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "bench_results.csv")
    check_path = os.path.join(args.out_dir, "correctness.csv")

    if args.plot_only:
        from plot_results import make_plots
        make_plots(csv_path, os.path.join(args.out_dir, "plots"))
        return

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required.")
    if args.quick:
        args.tokens = [1, 16, 256, 4096]
        args.hidden = [4096]
        args.dtypes = ["fp16"]
        args.reps, args.cold_iters = 20, 30

    tol = Tol(args.tol_q_max_abs, args.tol_q_mismatch_frac, args.tol_scale_rtol,
              TOL_SCALE_ATOL, args.tol_dequant_lsb)
    env = env_info()
    with open(os.path.join(args.out_dir, "env.json"), "w") as f:
        json.dump(dict(env=env, args=vars(args), tolerances=vars(tol)), f, indent=2)
    print(json.dumps(env, indent=2))

    builders = {"V0": make_v0, "V1": make_v1, "V2": lambda: make_v2(args.v2_impl)}
    variants: List[Variant] = []
    for name in args.variants:
        v = builders[name]()
        print(f"[{name}] {'OK  ' if v.available else 'SKIP'} {v.impl}" + ("" if v.available else f"  ({v.reason})"))
        if v.available:
            variants.append(v)
    if not variants:
        sys.exit("No variants available.")
    if "V0" not in [v.name for v in variants]:
        print("WARNING: V0 unavailable -> speedups will be blank.")

    l2 = int(env["l2_cache_MB"] * 2**20) or 48 * 2**20
    flush_bytes = (args.flush_mb * 2**20) if args.flush_mb else max(8 * l2, 256 * 2**20)
    flush_buf = torch.empty(flush_bytes // 4, dtype=torch.int32, device="cuda") \
        if "cold" in args.cache_modes else None

    res_modes = {"off": [False], "on": [True], "both": [False, True]}[args.residual]
    timing_method = "cuda_events+" + ("eager" if args.no_graph else "cuda_graph")

    rows: List[dict] = []
    check_rows: List[dict] = []
    n_fail = 0
    total = len(args.dtypes) * len(res_modes) * len(args.hidden) * len(args.tokens)
    k = 0
    for dtn in args.dtypes:
        elem = torch.empty((), dtype=DTYPES[dtn]).element_size()
        for res in res_modes:
            for H in args.hidden:
                for T in args.tokens:
                    k += 1
                    meta = dict(dtype=dtn, num_tokens=T, hidden_size=H, residual=int(res))
                    x0, w, r0 = make_inputs(T, H, dtn, res, args.seed)

                    # ---------------- cross-check ----------------
                    passed: Dict[str, bool] = {v.name: True for v in variants}
                    if not args.skip_check:
                        ref_y, ref_res = reference(x0, w, args.eps, r0)
                        outs = {}
                        for v in variants:
                            try:
                                outs[v.name] = run_once(v, x0, w, r0, args.eps)
                            except Exception as e:
                                passed[v.name] = False
                                check_rows.append(dict(variant=v.name, compared_to="run", **meta,
                                                       passed=False, note=f"error: {e}"))
                        for name, o in outs.items():
                            r = compare(name, "ref_fp32", o, None, ref_y, ref_res, meta, tol)
                            check_rows.append(r)
                            passed[name] &= r["passed"]
                            # pairwise: V1 vs V0, V2 vs V0, V2 vs V1
                            order = [vv.name for vv in variants]
                            for other in order[:order.index(name)]:
                                if other in outs:
                                    r = compare(name, other, o, outs[other], None, None, meta, tol)
                                    check_rows.append(r)
                                    passed[name] &= r["passed"]
                        del outs, ref_y, ref_res
                        n_fail += sum(1 for p in passed.values() if not p)

                    # ---------------- timing ----------------
                    line = [f"[{k}/{total}] {dtn} res={int(res)} H={H:5d} T={T:6d}"]
                    for v in variants:
                        if not passed[v.name] and not args.bench_failed:
                            line.append(f"{v.name}=FAIL(check)")
                            continue
                        b = Buffers.create(x0, w, r0, v.needs_scratch)
                        fn = lambda v=v, b=b: b.call(v, args.eps)
                        for cm in args.cache_modes:
                            try:
                                if cm == "warm":
                                    samples, inner = time_warm(fn, b.reset, res, not args.no_graph,
                                                               args.reps, args.target_ms, args.max_inner)
                                else:
                                    samples, inner = time_cold(fn, b.reset, res, not args.no_graph,
                                                               args.cold_iters, flush_buf)
                            except Exception as e:
                                traceback.print_exc()
                                line.append(f"{v.name}/{cm}=ERR")
                                continue
                            st = _stats(samples)
                            rows.append(dict(
                                variant=v.name, impl=v.impl, gpu=env["gpu"], **meta,
                                cache_mode=cm, timing_method=timing_method, inner_iters=inner,
                                **st,
                                eff_bandwidth_GBps=ideal_bytes(T, H, elem, res) / (st["median_us"] * 1e3),
                                speedup_vs_v0="", check_passed=int(passed[v.name]),
                                torch_version=env["torch_version"], vllm_version=env["vllm_version"],
                                triton_version=env["triton_version"], timestamp=env["timestamp"],
                            ))
                            line.append(f"{v.name}/{cm}={st['median_us']:8.2f}us")
                        del b
                    print("  ".join(line), flush=True)
                    del x0, w, r0
                    torch.cuda.empty_cache()

    # ---------------- speedups vs V0 ----------------
    base = {(r["dtype"], r["num_tokens"], r["hidden_size"], r["residual"], r["cache_mode"]): r["median_us"]
            for r in rows if r["variant"] == "V0"}
    for r in rows:
        key = (r["dtype"], r["num_tokens"], r["hidden_size"], r["residual"], r["cache_mode"])
        if key in base:
            r["speedup_vs_v0"] = base[key] / r["median_us"]

    def _fmt(v):
        return f"{v:.6g}" if isinstance(v, float) else v

    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _fmt(r.get(k, "")) for k in CSV_FIELDS})
    with open(check_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        wr.writeheader()
        for r in check_rows:
            wr.writerow({k: _fmt(r.get(k, "")) for k in CHECK_FIELDS})

    print(f"\nWrote {csv_path} ({len(rows)} rows) and {check_path} ({len(check_rows)} rows)")
    if not args.skip_check:
        bad = [r for r in check_rows if not r.get("passed")]
        print(f"Cross-check: {len(check_rows) - len(bad)}/{len(check_rows)} comparisons passed")
        for r in bad[:20]:
            print("  FAIL", {k: r.get(k) for k in CHECK_FIELDS})

    # summary: geometric-mean speedup per variant / cache mode
    print("\nGeo-mean speedup vs V0 (median latency):")
    for cm in args.cache_modes:
        for v in variants:
            sp = [r["speedup_vs_v0"] for r in rows
                  if r["variant"] == v.name and r["cache_mode"] == cm and r["speedup_vs_v0"] != ""]
            if sp:
                gm = math.exp(sum(math.log(s) for s in sp) / len(sp))
                print(f"  {cm:4s} {v.name}: {gm:.3f}x  (min {min(sp):.2f}x, max {max(sp):.2f}x, n={len(sp)})")

    if not args.no_plot:
        try:
            from plot_results import make_plots
            make_plots(csv_path, os.path.join(args.out_dir, "plots"))
        except Exception as e:
            print(f"Plotting failed ({e}); rerun with --plot-only")

    if n_fail and not args.skip_check:
        sys.exit(2)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
