#!/usr/bin/env python3
"""Parse swift logging.jsonl, compute steady-state per-step timing.

Strategy:
- elapsed_time field is integer-seconds ("2m 13s" format). Taking delta between
  the last step and step `warmup` gives a stable steady-state per-step time
  that avoids compile-cold-start contamination.
- train_speed(s/it) is a cumulative rolling mean (from step 1), so its "last"
  value underestimates recent speed when cold-start was slow. Use steady_step_s
  as the primary number and tail_speed as a sanity check.

Usage:
    python parse_swift_log.py [<runs_root>]
    # Default runs_root = /home/ubuntu/fyh/megatron_output/nccl_sweep
    # Prints one line per run found under <runs_root>/*/v*-*/logging.jsonl
"""
from __future__ import annotations
import json, sys, glob, os, re


def parse_elapsed(s):
    """'2m 13s' -> 133.0 ; '13s' -> 13.0 ; '1h 2m 3s' -> 3723.0"""
    total = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([hmsd])", s):
        v = float(m.group(1))
        u = m.group(2)
        total += {"s": 1, "m": 60, "h": 3600, "d": 86400}[u] * v
    return total


def parse_run(jsonl_path, warmup=10):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # swift uses either 'global_step/max_steps' (HF backend) or
            # 'iteration' (Megatron backend) — normalise here
            if "global_step/max_steps" in d:
                step = int(str(d["global_step/max_steps"]).split("/")[0])
            elif "iteration" in d:
                step = int(str(d["iteration"]).split("/")[0])
            else:
                continue
            et = d.get("elapsed_time")
            if et is None:
                continue
            elapsed_s = parse_elapsed(str(et)) if isinstance(et, str) else float(et)
            rows.append({
                "step": step,
                "elapsed_s": elapsed_s,
                "speed_s_it": float(d.get("train_speed(s/it)", 0.0) or 0.0),
                "loss": d.get("loss"),
                "mem_gib": d.get("memory(GiB)"),
            })
    if not rows:
        return None
    if rows[-1]["step"] <= warmup:
        return None
    start = next((r for r in rows if r["step"] >= warmup), None)
    end = rows[-1]
    if start is None or end["step"] - start["step"] < 5:
        return None
    steady_per_step = (end["elapsed_s"] - start["elapsed_s"]) / (end["step"] - start["step"])
    cumu_per_step = end["elapsed_s"] / end["step"]
    try:
        mems = [float(r["mem_gib"]) for r in rows if r["mem_gib"] is not None]
        peak_mem = max(mems) if mems else 0.0
    except Exception:
        peak_mem = 0.0
    return {
        "n_steps": rows[-1]["step"],
        "warmup_skipped": warmup,
        "window_steps": end["step"] - start["step"],
        "steady_step_s": steady_per_step,
        "cumu_step_s": cumu_per_step,
        "tail_speed": end["speed_s_it"],
        "loss_first": rows[0]["loss"],
        "loss_last": end["loss"],
        "peak_mem_gib": peak_mem,
        "total_elapsed_s": end["elapsed_s"],
    }


def main():
    runs_root = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/fyh/megatron_output/nccl_sweep"
    paths = sorted(glob.glob(os.path.join(runs_root, "*/v*-*/logging.jsonl")))
    if not paths:
        # fall back: look one level deeper (for phases with run_NN subdirs)
        paths = sorted(glob.glob(os.path.join(runs_root, "*/*/v*-*/logging.jsonl")))
    by_run = {}
    for p in paths:
        run = os.path.relpath(p, runs_root).split(os.sep)[0]
        if "/" in os.path.relpath(p, runs_root):
            parent = os.path.dirname(os.path.dirname(p))
            run = os.path.relpath(parent, runs_root)
        if run not in by_run or p > by_run[run]:
            by_run[run] = p
    header = f"{'run':<40} {'n':>4} {'win':>4} {'steady_s/step':>14} {'tail_avg_s':>11} {'mem_gib':>8} {'total_s':>8}"
    print(header)
    print("-" * len(header))
    base_steady = None
    for run, path in sorted(by_run.items()):
        r = parse_run(path)
        if r is None:
            print(f"{run:<40} (no data)")
            continue
        delta = ""
        if "baseline" in run.lower() and base_steady is None:
            base_steady = r["steady_step_s"]
        elif base_steady:
            d = (r["steady_step_s"] - base_steady) / base_steady * 100
            delta = f"  ({d:+.1f}%)"
        print(f"{run:<40} {r['n_steps']:>4} {r['window_steps']:>4} "
              f"{r['steady_step_s']:>14.3f} {r['tail_speed']:>11.3f} "
              f"{r['peak_mem_gib']:>8.2f} {r['total_elapsed_s']:>8.0f}{delta}")


if __name__ == "__main__":
    main()
