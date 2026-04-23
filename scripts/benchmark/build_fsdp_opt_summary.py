#!/usr/bin/env python3
"""Aggregate outputs of run_fsdp_opt_sweep.sh into a single comparison table.

For each `<bench_root>/<cfg>/` sub-directory, reads:
  - cfg.json                  (sweep config metadata)
  - train/report.json         (from report_swift_sp.py)
  - train/bench.jsonl         (per-step times, for median)
  - train/dcgm_tc.tsv         (DCGM @ 10 Hz)
  - train/train.log           (for OOM / crash status)
  - profile/analysis.txt      (torch.profiler kernel breakdown, we re-run it
                               from the trace to stay machine-readable)
  - profile/trace_rank0_step3.json  (for TC-eligible %)

Emits `_opt_summary.md` + `_opt_summary.json` with columns:
  name, tier, status, peak_mem_gb, train_step_ms, token_acc_last,
  dcgm_tc_busy20_pct, dcgm_power_peak_w, gemm_ms, flash_ms, nccl_ms, memshuf_ms,
  gpu_busy_ms, wall_ms, tc_eligible_pct, est_real_mfu_pct, compile_flag, notes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from analyze_torch_profile import analyze as analyze_trace  # type: ignore


H100_BF16_PEAK_TFLOPS = 989.0


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def parse_dcgm(tsv: Path):
    if not tsv.exists():
        return None
    series = defaultdict(list)
    with open(tsv) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                ts = float(row["ts"])
                series[ts].append(
                    (
                        float(row["tc_active"] or 0),
                        float(row["power_w"] or 0),
                        float(row["gr_active"] or 0),
                        float(row["dram_active"] or 0),
                    )
                )
            except Exception:
                pass
    if not series:
        return None
    tcs = []
    pws = []
    grs = []
    drs = []
    for _, v in sorted(series.items()):
        tcs.append(sum(x[0] for x in v) / len(v))
        pws.append(sum(x[1] for x in v) / len(v))
        grs.append(sum(x[2] for x in v) / len(v))
        drs.append(sum(x[3] for x in v) / len(v))
    n = len(tcs)
    top_n = max(1, n // 5)
    thr = sorted(tcs, reverse=True)[top_n - 1]
    busy_idx = [i for i, t in enumerate(tcs) if t >= thr]
    def _m(arr, idx):
        return sum(arr[i] for i in idx) / len(idx)
    return {
        "n_samples": n,
        "tc_all_avg_pct": sum(tcs) / n * 100,
        "tc_peak_pct": max(tcs) * 100,
        "tc_busy20_avg_pct": _m(tcs, busy_idx) * 100,
        "power_all_avg_w": sum(pws) / n,
        "power_peak_w": max(pws),
        "power_busy20_avg_w": _m(pws, busy_idx),
        "gr_busy20_avg_pct": _m(grs, busy_idx) * 100,
        "dram_busy20_avg_pct": _m(drs, busy_idx) * 100,
    }


def median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def steady_step_ms(bench_rows, warmup: int = 3):
    """Use `train_speed_s_per_it` at final non-null row (it's an EMA).

    Better: the final "train_speed" reported is already cumulative. We derive
    steady-state from the last measured step.
    """
    steps = [r for r in bench_rows if r.get("step") and r.get("train_speed_s_per_it")]
    if not steps:
        return None
    last = steps[-1]
    return float(last["train_speed_s_per_it"]) * 1000.0


def classify_for_mfu(analysis: dict):
    """Sum GEMM + FlashAttn kernel time (tensor-core-eligible)."""
    if not analysis:
        return None
    cats = {c["category"]: c for c in analysis["by_category"]}
    gemm = cats.get("GEMM", {}).get("ms", 0.0)
    flash = cats.get("FlashAttn", {}).get("ms", 0.0)
    nccl = cats.get("NCCL", {}).get("ms", 0.0)
    memsh = cats.get("Memory", {}).get("ms", 0.0)
    total = analysis["total_gpu_ms"]
    return {
        "gpu_ms": total,
        "gemm_ms": gemm,
        "flash_ms": flash,
        "nccl_ms": nccl,
        "memshuf_ms": memsh,
        "tc_eligible_ms": gemm + flash,
    }


def scan_config(cfg_dir: Path):
    cfg = _read_json(cfg_dir / "cfg.json") or {"name": cfg_dir.name}
    out: dict = {"name": cfg["name"], "tier": cfg.get("tier"), "cfg": cfg}

    train_dir = cfg_dir / "train"
    prof_dir = cfg_dir / "profile"

    # status
    rc_file = cfg_dir / "train.rc"
    train_rc = None
    if rc_file.exists():
        s = rc_file.read_text().strip()
        try:
            train_rc = int(s.split("=")[-1])
        except Exception:
            pass
    out["train_rc"] = train_rc

    log = train_dir / "train.log"
    note_bits = []
    if log.exists():
        text = log.read_text(errors="ignore")
        if "CUDA out of memory" in text or "OutOfMemoryError" in text:
            note_bits.append("OOM")
        if "NaN" in text and "grad_norm" in text.lower():
            note_bits.append("NaN-grad")
        if "cuDNN error" in text or "CUDA error" in text:
            note_bits.append("CUDA-err")
    out["notes"] = ",".join(note_bits) or "-"

    # report.json
    rep = _read_json(train_dir / "report.json") or {}
    out["peak_mem_gb"] = rep.get("peak_mem_gib_from_swift_log") or rep.get("peak_mem_gb")
    out["step_time_ms"] = rep.get("mean_step_time_ms")
    out["tokens_per_sec_per_gpu"] = rep.get("tokens_per_sec_per_gpu")
    out["mfu_pct_swift_formula"] = rep.get("mfu_pct")
    out["avg_power_w"] = rep.get("avg_power_w")
    out["peak_power_w_swift"] = rep.get("peak_power_w")

    # bench.jsonl: token_acc of last step + median step time
    bench_rows = _read_jsonl(train_dir / "bench.jsonl")
    out["token_acc_last"] = None
    out["loss_last"] = None
    for r in reversed(bench_rows):
        if r.get("token_acc") is not None:
            out["token_acc_last"] = float(r["token_acc"])
            out["loss_last"] = float(r.get("loss")) if r.get("loss") is not None else None
            break
    out["steady_step_ms"] = steady_step_ms(bench_rows)

    # DCGM
    dcgm = parse_dcgm(train_dir / "dcgm_tc.tsv")
    if dcgm:
        out["dcgm_tc_busy20_pct"] = dcgm["tc_busy20_avg_pct"]
        out["dcgm_tc_peak_pct"] = dcgm["tc_peak_pct"]
        out["dcgm_power_peak_w"] = dcgm["power_peak_w"]
        out["dcgm_power_busy20_w"] = dcgm["power_busy20_avg_w"]
        out["dcgm_n"] = dcgm["n_samples"]

    # profile trace
    trace = None
    if prof_dir.exists():
        tlist = sorted(prof_dir.glob("trace_rank0_step*.json"))
        if tlist:
            trace = tlist[0]
    if trace:
        analysis = analyze_trace(trace)
        out["profile"] = classify_for_mfu(analysis)
        # Derive TC-eligible % over wall time. wall ≈ steady step ms.
        wall = out.get("steady_step_ms") or analysis["total_gpu_ms"]
        out["wall_ms_for_mfu"] = wall
        out["tc_eligible_pct"] = (
            out["profile"]["tc_eligible_ms"] / wall * 100 if wall else None
        )
        # assume peak efficiency during TC kernels
        out["est_real_mfu_pct"] = (
            out["profile"]["tc_eligible_ms"] / wall * 100 if wall else None
        )

    return out


def fmt(x, spec: str = "", na: str = "—"):
    if x is None:
        return na
    try:
        return format(x, spec)
    except Exception:
        return str(x)


def render_md(rows: list[dict]) -> str:
    lines = []
    lines.append("# FSDP2+SP Optimisation Sweep — Summary\n")
    lines.append("Generated by `build_fsdp_opt_summary.py`. "
                 "Training column is from 10-step run; profile column from 5-step run, step 3 traced on rank 0.\n")
    lines.append("Real MFU proxy = `(GEMM+FlashAttn kernel ms) / steady_step_ms`. "
                 "Assumes those kernels hit bf16 peak; actual is ≤ this number.\n")

    # main table
    hdr = (
        "| cfg | tier | status | peak mem (GB) | step (ms) | token_acc | "
        "TC% busy20 (DCGM) | power peak (W) | "
        "GEMM ms | Flash ms | NCCL ms | MemShuf ms | GPU busy / wall | "
        "TC-eligible % | est real MFU | notes |"
    )
    sep = "|" + "|".join(["---"] * 16) + "|"
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        status = "OOM" if "OOM" in r.get("notes", "") else (
            "FAIL" if r.get("train_rc") not in (0, None) else "ok"
        )
        prof = r.get("profile") or {}
        wall = r.get("steady_step_ms")
        gpu_busy_pct = None
        if prof.get("gpu_ms") and wall:
            gpu_busy_pct = prof["gpu_ms"] / wall * 100
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("name", "-"),
                    str(r.get("tier", "-")),
                    status,
                    fmt(r.get("peak_mem_gb"), ".1f"),
                    fmt(r.get("steady_step_ms"), ".0f"),
                    fmt(r.get("token_acc_last"), ".2f"),
                    fmt(r.get("dcgm_tc_busy20_pct"), ".2f"),
                    fmt(r.get("dcgm_power_peak_w"), ".0f"),
                    fmt(prof.get("gemm_ms"), ".1f"),
                    fmt(prof.get("flash_ms"), ".2f"),
                    fmt(prof.get("nccl_ms"), ".1f"),
                    fmt(prof.get("memshuf_ms"), ".1f"),
                    fmt(gpu_busy_pct, ".1f") + " %"
                    if gpu_busy_pct is not None
                    else "—",
                    fmt(r.get("tc_eligible_pct"), ".2f"),
                    fmt(r.get("est_real_mfu_pct"), ".2f"),
                    r.get("notes", "-"),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("### Swift's own MFU number (for calibration vs. reality)\n")
    lines.append("| cfg | swift MFU formula % | steady step ms | tokens/s/GPU | note |")
    lines.append("|---|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("name", "-"),
                    fmt(r.get("mfu_pct_swift_formula"), ".1f"),
                    fmt(r.get("steady_step_ms"), ".0f"),
                    fmt(r.get("tokens_per_sec_per_gpu"), ".0f"),
                    "swift 6ND/step over-estimates — see real MFU column above",
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_root", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for d in sorted(args.bench_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if not (d / "cfg.json").exists():
            continue
        row = scan_config(d)
        rows.append(row)

    # preserve canonical order from matrix by tier then name
    order = [
        "baseline", "no_ac", "mbs2", "compile",
        "mbs4", "no_reshard", "wrap_large", "sp1_mbs2",
        "combo_easy", "combo_compile",
    ]
    rank = {n: i for i, n in enumerate(order)}
    rows.sort(key=lambda r: rank.get(r["name"], 999))

    args.out_json.write_text(json.dumps(rows, indent=2))
    args.out_md.write_text(render_md(rows))
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    print()
    print(render_md(rows))


if __name__ == "__main__":
    main()
