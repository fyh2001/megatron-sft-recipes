#!/usr/bin/env python3
"""Aggregate P1 sweep results into a single comparison table.

Usage:
    python3 build_p1_summary.py [<p1_root_dir>]

Default root: /home/ubuntu/fyh/megatron_output/gemma4_opt/p1_gbs_sweep

Reads each run_*/report.json, pulls key throughput + memory numbers, and writes:
    <root>/_summary.md   - markdown table
    <root>/_summary.tsv  - tab-separated for further analysis
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/fyh/megatron_output/gemma4_opt/p1_gbs_sweep")
    if not root.exists():
        print(f"[err] root not found: {root}")
        return 1

    rows = []
    for run_dir in sorted(root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        report = run_dir / "report.json"
        status_file = run_dir / "STATUS"
        status = status_file.read_text().strip() if status_file.exists() else "?"

        # Parse label from dirname (run_<TS>_<LABEL>)
        m = re.match(r"run_\d+_\d+_(.+)", run_dir.name)
        label = m.group(1) if m else run_dir.name

        # Extract MBS/GAS/GBS from label
        mbs_m = re.search(r"mbs(\d+)", label)
        gas_m = re.search(r"ga(\d+)", label)
        gbs_m = re.search(r"gbs(\d+)", label)
        mbs = int(mbs_m.group(1)) if mbs_m else None
        gas = int(gas_m.group(1)) if gas_m else None
        gbs = int(gbs_m.group(1)) if gbs_m else None

        offload_flag = "offload" if (gas and gas >= 2) else "native"

        if not report.exists():
            rows.append({
                "label": label, "mbs": mbs, "gas": gas, "gbs": gbs,
                "mode": offload_flag, "status": "FAILED",
                "summary": status,
            })
            continue

        d = json.loads(report.read_text())
        rows.append({
            "label": label,
            "mbs": mbs, "gas": gas, "gbs": gbs,
            "mode": offload_flag,
            "status": "ok",
            "step_ms": d.get("mean_step_time_ms"),
            "tok_per_s_per_gpu": d.get("tokens_per_sec_per_gpu"),
            "peak_mem_gib": d.get("peak_mem_gib_from_swift_log"),
            "achieved_tflops_active": d.get("achieved_tflops_per_gpu_active"),
            "real_mfu_active_pct": d.get("mfu_pct_active_params"),
            "loss_first_step": d.get("loss_first_step"),
            "actual_total_wall_min": d.get("actual_total_wall_min"),
        })

    # Sort by GBS then by MBS
    rows.sort(key=lambda r: (r.get("gbs") or 0, r.get("mbs") or 0))

    # Determine peak (best tokens_per_s_per_gpu among ok runs)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    peak = None
    if ok_rows:
        peak = max(ok_rows, key=lambda r: r.get("tok_per_s_per_gpu") or 0)

    # Write markdown
    md = root / "_summary.md"
    with open(md, "w") as f:
        f.write(f"# P1 GBS/MBS sweep summary\n\n")
        f.write(f"Root: `{root}`  ·  total runs: {len(rows)}  ·  ok: {len(ok_rows)}\n\n")
        f.write(f"## Per-config results\n\n")
        f.write("| MBS | GAS | GBS | mode | step (ms) | tokens/s/GPU | peak mem (GiB) | active TFLOPS/GPU | real MFU% | loss step1 | full-epoch wall* |\n")
        f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r["status"] != "ok":
                f.write(f"| {r.get('mbs')} | {r.get('gas')} | {r.get('gbs')} | {r['mode']} | — | — | — | — | — | — | **{r.get('summary', 'FAIL')}** |\n")
            else:
                # Recompute full-epoch wall: 18819 samples / GBS × step_ms
                gbs = r.get('gbs') or 0
                step_ms = r.get('step_ms') or 0
                wall_min = (18819.0 / gbs * step_ms / 1000 / 60.0) if (gbs > 0 and step_ms > 0) else None
                wall_str = f"{wall_min:.0f} min" if wall_min else "—"
                marker = " ★" if (peak and r['label'] == peak['label']) else ""
                f.write(
                    f"| {r['mbs']} | {r['gas']} | {r['gbs']} | {r['mode']} | "
                    f"{r['step_ms']:.0f} | "
                    f"**{r['tok_per_s_per_gpu']:.0f}**{marker} | "
                    f"{r['peak_mem_gib']:.1f} | "
                    f"{r['achieved_tflops_active']:.0f} | "
                    f"{r['real_mfu_active_pct']:.1f}% | "
                    f"{r['loss_first_step']:.4f} | "
                    f"{wall_str} |\n"
                )
        f.write("\n_*full-epoch wall ≈ (18819 / GBS) × step_ms; theoretical, doesn't include data loader overhead. ★ = peak tokens/s/GPU._\n\n")

        if peak:
            f.write(f"## Peak\n\n")
            f.write(f"- **Best throughput**: MBS={peak['mbs']} GAS={peak['gas']} GBS={peak['gbs']} ({peak['mode']})\n")
            f.write(f"- tokens/s/GPU = **{peak['tok_per_s_per_gpu']:.0f}**\n")
            f.write(f"- step = {peak['step_ms']:.0f} ms · peak mem = {peak['peak_mem_gib']:.1f} GiB · active MFU = {peak['real_mfu_active_pct']:.1f}%\n")

    # TSV (raw numbers for analysis)
    tsv = root / "_summary.tsv"
    cols = ["label", "mbs", "gas", "gbs", "mode", "status", "step_ms",
            "tok_per_s_per_gpu", "peak_mem_gib", "achieved_tflops_active",
            "real_mfu_active_pct", "loss_first_step", "actual_total_wall_min"]
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    print(f"wrote {md}")
    print(f"wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
