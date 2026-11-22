#!/usr/bin/env python3
"""
Part 3 edge-case correctness test for V3 (all configs), with V0/V1/V2 alongside for reference.
Uses the benchmark's own inputs, reference, tolerances and compare() (bench_rmsnorm_int8.py), so the
checks are exactly the ones used for the timing runs.

Cases (BF16 and FP16, residual on and off):
  T = 1, 3, 7, 1000, 1023 (not powers of two)          H = 2048
  H = 5120 and 3000 (not powers of two -> masked tail)   T = 1, 257
  all-zero rows (x + residual == 0 on some rows)         H = 2048, 4096
  very large values (x * 1000, one 3e4 outlier per row)  H = 2048, 4096
Usage (from ~/rmsnorm_int8_bench, venv active):  python test_edge_cases.py [--out edge_cases.csv]
Exit code 0 = every V3 config passed every case.
"""
import argparse, csv, sys
import torch
import bench_rmsnorm_int8 as B

EPS = B.DEFAULT_EPS


def build_variants():
    vs = []
    for name, mk in (("V0", B.make_v0), ("V1", B.make_v1), ("V2", lambda: B.make_v2(None)),
                     ("V3", lambda: B.make_v3(B.DEFAULT_V3_CONFIGS))):
        built = mk()
        for v in (built if isinstance(built, list) else [built]):
            print(f"[{v.name}] {'OK  ' if v.available else 'SKIP'} {v.impl}" + ("" if v.available else f" ({v.reason})"))
            if v.available:
                vs.append(v)
    return vs


def cases():
    for dt in ("bf16", "fp16"):
        for res in (True, False):
            for T in (1, 3, 7, 1000, 1023):
                yield dict(name=f"odd_T", T=T, H=2048, dt=dt, res=res, mod=None)
            for H in (5120, 3000):
                for T in (1, 257):
                    yield dict(name="odd_H", T=T, H=H, dt=dt, res=res, mod=None)
            for H in (2048, 4096):
                yield dict(name="zero_rows", T=64, H=H, dt=dt, res=res, mod="zero")
                yield dict(name="large_vals", T=64, H=H, dt=dt, res=res, mod="large")


def make(case):
    x0, w, r0 = B.make_inputs(case["T"], case["H"], case["dt"], case["res"], B.DEFAULT_SEED)
    zero_rows = []
    if case["mod"] == "zero":
        zero_rows = [0, 5, case["T"] - 1]
        for i in zero_rows:
            if r0 is not None:
                x0[i] = -r0[i]           # x + residual == 0 exactly
            else:
                x0[i] = 0
    elif case["mod"] == "large":
        big = 1000.0 if case["dt"] == "bf16" else 50.0     # keep fp16 below its 65504 limit after the add
        x0 = (x0.float() * big).to(x0.dtype)
        x0[:, 17] = 3e4 if case["dt"] == "bf16" else 3e3
    return x0, w, r0, zero_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edge_cases.csv")
    a = ap.parse_args()
    tol = B.Tol()
    variants = build_variants()
    rows, v3_fail, other_fail = [], 0, 0
    for c in cases():
        x0, w, r0, zr = make(c)
        ref_y, ref_res = B.reference(x0, w, EPS, r0)
        keep = torch.ones(c["T"], dtype=torch.bool, device=x0.device)
        keep[zr] = False
        meta = dict(dtype=c["dt"], num_tokens=c["T"], hidden_size=c["H"], residual=int(c["res"]))
        outs = {}
        for v in variants:
            try:
                q, s, r = B.run_once(v, x0, w, r0, EPS)
                outs[v.name] = (q, s, r)
            except Exception as e:
                rows.append(dict(case=c["name"], variant=v.name, compared_to="run", **meta, passed=False,
                                 note=f"error: {type(e).__name__}: {str(e)[:120]}"))
        for v in variants:
            if v.name not in outs:
                if v.name.startswith("V3"): v3_fail += 1
                else: other_fail += 1
                continue
            q, s, r = outs[v.name]
            notes, ok = [], True
            # zero rows: q must be all 0 and the scale finite (no NaN/inf)
            for i in zr:
                if not (bool((q[i] == 0).all()) and bool(torch.isfinite(s[i]).all())):
                    ok = False; notes.append(f"zero-row {i}: q!=0 or scale not finite")
            # everything else: the benchmark's own compare() vs FP32 reference and vs V0
            sel = (q[keep], s[keep], None if r is None else r[keep])
            r1 = B.compare(v.name, "ref_fp32", sel, None, ref_y[keep], None if ref_res is None else ref_res[keep], meta, tol)
            ok &= bool(r1["passed"]); notes.append(f"lsb={r1['dequant_max_err_lsb']:.2f}" if r1["dequant_max_err_lsb"] != "" else "")
            if r is not None and ref_res is not None and not torch.equal(r, ref_res):
                ok = False; notes.append("residual != x+residual")
            if "V0" in outs and v.name != "V0":
                q0, s0, r0o = outs["V0"]
                r2 = B.compare(v.name, "V0", sel, (q0[keep], s0[keep], None if r0o is None else r0o[keep]), None, None, meta, tol)
                ok &= bool(r2["passed"])
                notes.append(f"vsV0 mism={r2['q_mismatch_frac']*100:.1f}% maxdiff={r2['q_max_abs_diff']}")
            rows.append(dict(case=c["name"], variant=v.name, compared_to="ref+V0", **meta, passed=ok,
                             note="; ".join(n for n in notes if n)))
            if not ok:
                if v.name.startswith("V3"): v3_fail += 1
                else: other_fail += 1
        status = " ".join(f"{r['variant']}:{'ok' if r['passed'] else 'FAIL'}" for r in rows[-len(variants):])
        print(f"{c['name']:10s} {c['dt']} res={int(c['res'])} T={c['T']:5d} H={c['H']:5d}  {status}")
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    n3 = sum(1 for r in rows if r["variant"].startswith("V3"))
    print(f"\nV3: {n3 - v3_fail}/{n3} checks passed.  V0/V1/V2 failures (for information): {other_fail}")
    print(f"Wrote {a.out}")
    sys.exit(0 if v3_fail == 0 else 1)


if __name__ == "__main__":
    main()
