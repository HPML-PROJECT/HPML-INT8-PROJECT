#!/usr/bin/env python3
"""
Turn ncu reports into comparison tables.
  python analysis/parse_ncu.py reports/ncu   -> tables/memory_table.csv|md, tables/execution_table.csv|md
Filename convention: <variant>_<gpu>_H<hidden>_T<tokens>_<cache>.ncu-rep
Paths are relative to the toolkit directory (the parent of this file's folder).
"""
import csv, glob, io, os, re, statistics, subprocess, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")


def resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)

M = {
    "time_us": "gpu__time_duration.sum",
    "dram_rd": "dram__bytes_read.sum", "dram_wr": "dram__bytes_write.sum",
    "l2_bytes": "lts__t_bytes.sum", "l2_hit": "lts__t_sector_hit_rate.pct",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "regs": "launch__registers_per_thread", "block": "launch__block_size", "grid": "launch__grid_size",
    "occ_ach": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theo": "sm__maximum_warps_per_active_cycle_pct",
    "inst": "smsp__inst_executed.sum", "ld_inst": "smsp__inst_executed_op_global_ld.sum",
    "ld_sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "ld_req": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
}
NAME_RE = re.compile(r"(?P<variant>v\d)_(?P<gpu>.+)_H(?P<hidden>\d+)_T(?P<tokens>\d+)_(?P<cache>cold|warm)\.ncu-rep$")


def num(s):
    try: return float(str(s).replace(",", ""))
    except ValueError: return None


def load_report(path):
    out = subprocess.run(["ncu", "--import", path, "--page", "raw", "--csv", "--print-units", "base"],
                         capture_output=True, text=True, check=True).stdout
    start = out.find('"ID"')                      # skip any ==PROF== preamble
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    return [r for r in rows if (r.get("ID") or "").strip().isdigit()]


def stalls(row, k=3):
    """Top-k warp stall reasons (average warps stalled per issue-active cycle)."""
    items = []
    for key, val in row.items():
        m = re.match(r"smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio", key)
        if m and num(val) is not None: items.append((num(val), m.group(1)))
    items = sorted(items, reverse=True)[:k]
    return ", ".join(f"{n}={v:.2f}" for v, n in items)


def min_bytes(T, H, residual=True):
    """Minimum DRAM bytes for the full fused operation (BF16): x in, r in, r out, weight, int8 out, fp32 scales."""
    return T * H * (2 + (2 + 2 if residual else 0) + 1) + 2 * H + 4 * T


def short(k): return re.sub(r"\(.*", "", k)[:60]


def main(d):
    mem, exe = [], []
    for path in sorted(glob.glob(os.path.join(d, "*.ncu-rep"))):
        m = NAME_RE.search(os.path.basename(path))
        if not m: print("skip (name):", path); continue
        meta = m.groupdict(); T, H = int(meta["tokens"]), int(meta["hidden"])
        byk = defaultdict(list)
        for r in load_report(path): byk[r["Kernel Name"]].append(r)
        agg = []
        for kname, rs in byk.items():
            med = lambda col: statistics.median([num(r.get(col)) for r in rs if num(r.get(col)) is not None] or [float("nan")])
            a = {k: med(c) for k, c in M.items()}
            a["time_us"] = a["time_us"] / 1e3   # raw value is ns (--print-units base) -> us
            a["kernel"] = short(kname); a["stalls"] = stalls(rs[len(rs) // 2])
            agg.append(a)
        tot = {k: sum(a[k] for a in agg) for k in ("time_us", "dram_rd", "dram_wr", "l2_bytes", "inst")}
        base = {**meta, "T": T, "H": H}
        for a in agg + [dict(tot, kernel="TOTAL", l2_hit=float("nan"), dram_pct=float("nan"), mem_pct=float("nan"),
                             regs=float("nan"), block=float("nan"), grid=float("nan"), occ_ach=float("nan"),
                             occ_theo=float("nan"), ld_inst=float("nan"), ld_sec=float("nan"), ld_req=float("nan"), stalls="")]:
            dram = a["dram_rd"] + a["dram_wr"]
            mb = min_bytes(T, H)
            mem.append({**base, "kernel": a["kernel"], "time_us": round(a["time_us"], 2),
                        "dram_MB": round(dram / 1e6, 3), "dram_rd_MB": round(a["dram_rd"] / 1e6, 3),
                        "dram_wr_MB": round(a["dram_wr"] / 1e6, 3), "L2_MB": round(a["l2_bytes"] / 1e6, 3),
                        "L2_hit_%": round(a["l2_hit"], 1), "DRAM_%peak": round(a["dram_pct"], 1),
                        "dram_GBps": round(dram / (a["time_us"] * 1e3), 1) if a["time_us"] else "",
                        "min_bytes_MB": round(mb / 1e6, 3),
                        "dram/min": round(dram / mb, 2) if a["kernel"] == "TOTAL" else ""})
            exe.append({**base, "kernel": a["kernel"], "regs/thr": a["regs"], "block": a["block"], "grid": a["grid"],
                        "occ_ach_%": round(a["occ_ach"], 1), "occ_theo_%": round(a["occ_theo"], 1),
                        "inst": a["inst"], "gld_inst": a["ld_inst"],
                        "ld_sectors/req": round(a["ld_sec"] / a["ld_req"], 2) if a["ld_req"] else "",
                        "approx_B/ld_inst": round(a["ld_sec"] * 32 / a["ld_inst"], 1) if a["ld_inst"] else "",
                        "top_stalls": a["stalls"]})
    os.makedirs(TABLES, exist_ok=True)
    for name, rows in (("memory_table", mem), ("execution_table", exe)):
        if not rows: print("no data for", name); continue
        rows.sort(key=lambda r: (r["gpu"], r["H"], r["cache"], r["T"], r["variant"], r["kernel"]))
        cols = list(rows[0].keys())
        with open(os.path.join(TABLES, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
        with open(os.path.join(TABLES, f"{name}.md"), "w") as f:
            f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in rows: f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        print(f"wrote tables/{name}.csv and .md ({len(rows)} rows)")


if __name__ == "__main__":
    main(resolve(sys.argv[1]) if len(sys.argv) > 1 else resolve("reports/ncu"))
