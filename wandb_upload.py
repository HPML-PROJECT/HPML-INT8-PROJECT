#!/usr/bin/env python3
"""
Upload the team's existing results (WS1–WS4) to one Weights & Biases project, so the midpoint report has a
dashboard link. Nothing is rerun: this reads the CSV/JSON/PNG files already in the repo.

One W&B run per result folder:
  ws1-main-L4      baseline-benchmark/results/l4_20261121        (V0/V1/V2, full grid)
  ws1-rerun-L4     baseline-benchmark/results/l4_nearL2           (200 samples + clock/power log)
  ws2-profile-L4   gpu profling - bottleneck/results/l4_20261121 (Nsight tables and plots)
  ws3-v3-L4        triton-v3/results/l4_part3                     (V3 configs, edge cases)
  ws4-vllm-L4      vllm-evaluation/results_l4_20261122 or part4_results_20261122_004608/results (W8A8 vs BF16, PPL, breakdown, Amdahl)
Each run gets: config (from env.json), every CSV as a W&B Table, every PNG as an image, headline numbers in the
run summary, a few native line charts, and the whole folder as a versioned artifact.

Usage (from the repo root, on any machine; no GPU needed):
  pip install wandb pandas pillow
  wandb login                      # paste your API key from https://wandb.ai/authorize
  python wandb_upload.py --entity <team-or-username> --project hpml-int8
  python wandb_upload.py --dry-run # show what would be logged, without W&B
"""
import argparse, glob, json, math, os, sys

FOLDERS = [
    ("ws1-main-L4", "WS1 baseline", "baseline-benchmark/results/l4_20261121"),
    ("ws1-rerun-L4", "WS1 baseline", "baseline-benchmark/results/l4_nearL2"),
    ("ws2-profile-L4", "WS2 profiling", "gpu profling - bottleneck/results/l4_20261121"),
    ("ws3-v3-L4", "WS3 Triton V3", ["triton-v3/results/l4_part3"]),
    ("ws4-vllm-L4", "WS4 vLLM end-to-end", ["vllm-evaluation/results_l4_20261122", "part4_results_20261122_004608/results"]),
]
SKIP_DIRS = ("nsys", "ncu", "traces")          # large binary profiler reports: not uploaded


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = ", ".join(map(str, v))[:500]
        else:
            out[key] = v
    return out


def files(folder, pattern):
    hits = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    return sorted(h for h in hits if not any(f"{os.sep}{s}{os.sep}" in h for s in SKIP_DIRS))


# ----------------------------- headline numbers per folder -----------------------------
def kernel_summary(pd, folder):
    """WS1/WS3: V1/V0 and V2/V0 (and best V3) median-time ratios at the key shapes, BF16 + residual."""
    p = os.path.join(folder, "bench_results.csv")
    if not os.path.exists(p):
        return {}, []
    b = pd.read_csv(p)
    b = b[(b["dtype"] == "bf16") & (b["residual"] == 1)] if "residual" in b else b[b["dtype"] == "bf16"]
    s, charts = {}, []
    piv = b.pivot_table(index=["cache_mode", "hidden_size", "num_tokens"], columns="variant", values="median_us")
    v3 = [c for c in piv.columns if str(c).startswith("V3")]
    if v3:
        piv["V3_best"] = piv[v3].min(axis=1)
    for (cache, H, T), r in piv.iterrows():
        if T in (256, 1024, 2048, 4096, 16384) and H in (2048, 4096) and "V0" in r:
            for v in ("V1", "V2", "V3_best"):
                if v in r and not math.isnan(r[v]):
                    s[f"{cache}/H{H}/T{T}/{v}_over_V0"] = round(float(r[v] / r["V0"]), 4)
    for cache in sorted(set(piv.index.get_level_values(0))):
        for H in sorted(set(piv.loc[cache].index.get_level_values(0))):
            d = piv.loc[(cache, H)]
            xs = [int(t) for t in d.index]
            keys = [v for v in ("V1", "V2", "V3_best") if v in d and "V0" in d]
            if keys:
                charts.append((f"{cache}_H{H}_time_over_V0",
                               f"time / V0 vs tokens (BF16, residual, {cache}, H={H}; >1 = slower than V0)",
                               xs, [[float(x) for x in (d[k] / d["V0"])] for k in keys], keys))
    c = os.path.join(folder, "correctness.csv")
    if os.path.exists(c):
        cc = pd.read_csv(c)
        s["correctness/passed"] = int(cc["passed"].astype(str).isin(["True", "1"]).sum())
        s["correctness/total"] = int(len(cc))
    e = os.path.join(folder, "edge_cases.csv")
    if os.path.exists(e):
        ee = pd.read_csv(e); v3e = ee[ee["variant"].astype(str).str.startswith("V3")]
        s["edge_cases/V3_passed"] = int(v3e["passed"].astype(str).isin(["True", "1"]).sum())
        s["edge_cases/V3_total"] = int(len(v3e))
    return s, charts


def clock_summary(folder):
    p = os.path.join(folder, "gpu_log.csv")
    if not os.path.exists(p):
        return {}
    import csv, statistics
    rows = [r for r in list(csv.reader(open(p)))[1:] if len(r) >= 5]
    busy = [r for r in rows if r[4].strip() != "0x0000000000000001"]
    if not busy:
        return {}
    mhz = [float(r[1].split()[0]) for r in busy]
    capped = sum(1 for r in busy if int(r[4].strip(), 16) & 0x4)
    return {"clock/sm_mhz_median": statistics.median(mhz), "clock/sm_mhz_min": min(mhz),
            "clock/power_capped_frac": round(capped / len(busy), 3)}


def ws4_summary(folder):
    s = {}
    for t in ("w8a8", "bf16"):
        p = os.path.join(folder, f"bench_{t}.json")
        if os.path.exists(p):
            b = json.load(open(p))
            for k, v in b.get("prefill", {}).items():
                s[f"{t}/prefill_{k}/tok_s"] = v["throughput_tok_s"]["median"]
            for k, v in b.get("decode", {}).items():
                s[f"{t}/decode_{k}/tpot_ms"] = v["tpot_ms"]["median"]
        q = os.path.join(folder, f"ppl_{t}.json")
        if os.path.exists(q):
            s[f"{t}/wikitext2_ppl"] = json.load(open(q))["ppl"]
    a = os.path.join(folder, "amdahl.json")
    if os.path.exists(a):
        for phase, p in json.load(open(a)).items():
            if not isinstance(p, dict) or "fractions" not in p:
                continue
            for basis in ("f_all_gpu", "f_fusable_gpu"):
                s[f"amdahl/{phase}/{basis}"] = p["fractions"][basis]
                for sv, val in p.get("speedup", {}).get(basis, {}).items():
                    s[f"amdahl/{phase}/{basis}/speedup_s={sv}"] = val
    return s


# ----------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None, help="W&B team or username (default: your default entity)")
    ap.add_argument("--project", default="hpml-int8")
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--only", nargs="*", help="run names to upload, e.g. ws1-main-L4 ws4-vllm-L4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    wandb = None
    if not a.dry_run:
        import wandb

    project_url = None
    for name, group, rel in FOLDERS:
        if a.only and name not in a.only:
            continue
        cands = rel if isinstance(rel, list) else [rel]
        rel = next((c for c in cands if os.path.isdir(os.path.join(a.root, c))), cands[0])
        folder = os.path.join(a.root, rel)
        if not os.path.isdir(folder):
            print(f"[skip] {name}: {rel} not found (merge that PR first)")
            continue
        env = {}
        for envp in (os.path.join(folder, "env.json"),):
            if os.path.exists(envp):
                env = flat(json.load(open(envp)))
        summary, charts = kernel_summary(pd, folder)
        summary.update(clock_summary(folder))
        if name.startswith("ws4"):
            summary.update(ws4_summary(folder))
        csvs, pngs = files(folder, "*.csv"), files(folder, "*.png")
        print(f"[{name}] {rel}: {len(csvs)} tables, {len(pngs)} images, {len(summary)} summary values, {len(charts)} charts")
        if a.dry_run:
            for k in list(summary)[:12]:
                print(f"    {k} = {summary[k]}")
            continue

        run = wandb.init(entity=a.entity, project=a.project, name=name, group=group, job_type="results-upload",
                         config={"source_folder": rel, "gpu": "NVIDIA L4", **env}, reinit=True)
        for c in csvs:
            key = "table/" + os.path.relpath(c, folder).replace(os.sep, "/").rsplit(".", 1)[0]
            try:
                run.log({key: wandb.Table(dataframe=pd.read_csv(c))})
            except Exception as e:
                print(f"    could not log {c}: {e}")
        for p in pngs:
            run.log({"plot/" + os.path.relpath(p, folder).replace(os.sep, "/").rsplit(".", 1)[0]: wandb.Image(p)})
        for key, title, xs, ys, keys in charts:
            run.log({f"chart/{key}": wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=title, xname="tokens T")})
        run.summary.update(summary)
        art = wandb.Artifact(name.replace("-", "_"), type="results",
                             description=f"Raw result files from {rel} (profiler binaries excluded)")
        for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
            if os.path.isfile(f) and not any(f"{os.sep}{s}{os.sep}" in f for s in SKIP_DIRS):
                art.add_file(f, name=os.path.relpath(f, folder))
        run.log_artifact(art)
        print(f"    -> {run.url}")
        project_url = run.url.split("/runs/")[0]
        run.finish()
    if project_url:
        print(f"\nProject: {project_url}")


if __name__ == "__main__":
    sys.exit(main())
