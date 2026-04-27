#!/usr/bin/env python3
"""Extract loss / grad_norm / mem / token_acc curves from swift logging.jsonl.

Outputs:
    <run_dir>/loss_curve.tsv    -- step  loss  grad_norm  mem_gib  token_acc  lr  elapsed_s  tokens_this_step
    <run_dir>/loss_curve.txt    -- ASCII sparkline + summary stats

Usage:
    python extract_loss_curve.py <logging.jsonl> [<output_dir>]
"""
from __future__ import annotations
import json
import math
import re
import sys
from pathlib import Path


def parse_elapsed(s):
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    total = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([hmsd])", str(s)):
        v = float(m.group(1))
        u = m.group(2)
        total += {"s": 1, "m": 60, "h": 3600, "d": 86400}[u] * v
    return total


def main():
    if len(sys.argv) < 2:
        print("usage: extract_loss_curve.py <logging.jsonl> [<output_dir>]")
        sys.exit(1)
    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.parent

    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step/max_steps" not in d:
                continue
            try:
                step = int(str(d["global_step/max_steps"]).split("/")[0])
            except Exception:
                continue
            rows.append({
                "step": step,
                "loss": d.get("loss"),
                "grad_norm": d.get("grad_norm"),
                "mem_gib": d.get("memory(GiB)"),
                "token_acc": d.get("token_acc"),
                "lr": d.get("learning_rate"),
                "elapsed_s": parse_elapsed(d.get("elapsed_time")),
                "tokens_this_step": d.get("tokens_this_step"),
                "tokens_per_gpu_per_sec": d.get("tokens_per_gpu_per_sec"),
                "train_speed_s_per_it": d.get("train_speed(s/it)"),
            })

    if not rows:
        print("[extract_loss_curve] no per-step records found")
        sys.exit(2)

    # TSV dump
    tsv_path = out_dir / "loss_curve.tsv"
    cols = ["step", "elapsed_s", "loss", "grad_norm", "mem_gib", "token_acc",
            "lr", "tokens_this_step", "tokens_per_gpu_per_sec",
            "train_speed_s_per_it"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # ASCII sparkline + stats
    losses = [r["loss"] for r in rows if isinstance(r["loss"], (int, float)) and not math.isnan(r["loss"])]
    txt_path = out_dir / "loss_curve.txt"
    with open(txt_path, "w") as f:
        f.write(f"# Loss curve from {log_path}\n")
        f.write(f"# {len(rows)} steps, {len(losses)} valid loss values\n")
        f.write(f"# elapsed: {rows[-1]['elapsed_s']:.0f}s = {rows[-1]['elapsed_s']/60:.1f}min\n")
        if losses:
            f.write(f"# loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  "
                    f"min={min(losses):.4f}  mean={sum(losses)/len(losses):.4f}\n")
        nans = sum(1 for r in rows if r["loss"] is not None
                   and isinstance(r["loss"], float) and math.isnan(r["loss"]))
        f.write(f"# NaN steps: {nans}\n\n")

        f.write(f"{'step':>6}  {'elapsed':>9}  {'loss':>8}  {'grad_norm':>10}  "
                f"{'mem_gib':>8}  {'tok/step':>10}  spark\n")
        # Sparkline normalise loss to 0..7 range
        spark_chars = " ▁▂▃▄▅▆▇█"
        if losses:
            lo, hi = min(losses), max(losses)
            denom = (hi - lo) or 1.0
            for r in rows:
                lv = r["loss"]
                if isinstance(lv, (int, float)) and not math.isnan(lv):
                    idx = int(((lv - lo) / denom) * 8)
                    sp = spark_chars[max(0, min(8, idx))]
                else:
                    sp = "?"
                f.write(f"{r['step']:>6}  {r['elapsed_s']:>8.0f}s  "
                        f"{(r['loss'] if r['loss'] is not None else 0):>8.4f}  "
                        f"{(r['grad_norm'] if r['grad_norm'] is not None else 0):>10.4f}  "
                        f"{(r['mem_gib'] if r['mem_gib'] is not None else 0):>8.2f}  "
                        f"{r.get('tokens_this_step', 0) or 0:>10}  {sp}\n")

    print(f"wrote {tsv_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
